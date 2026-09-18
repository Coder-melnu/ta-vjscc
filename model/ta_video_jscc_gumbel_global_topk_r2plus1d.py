# -*- coding: utf-8 -*-
"""Gumbel global-Top-K VideoJSCC with a frozen R(2+1)D task model.

The hard forward path transmits exactly K latent spatial locations per GoP.
``spatiotemporal`` ranks T*H'*W' locations jointly. ``spatial_only`` builds
one H'*W' ranking and repeats the same mask for every frame. During training,
Gumbel perturbations encourage exploration and a straight-through relaxation
provides gradients. Evaluation is deterministic.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.jscc import DeepJSCC
from model.temporal import TemporalFusionModule
from downstream.action_recognition.models.r2plus1d_recognizer import (
    KINETICS_MEAN,
    KINETICS_STD,
    R2Plus1DRecognizer,
    load_r2plus1d_checkpoint,
)


class FrozenR2Plus1D(nn.Module):
    """Locked five-frame recognizer; gradients still flow to its input."""

    def __init__(self, checkpoint_path):
        super().__init__()
        self.model = R2Plus1DRecognizer(pretrained=False, num_classes=101)
        self.checkpoint_metadata = load_r2plus1d_checkpoint(
            self.model, checkpoint_path
        )
        if self.checkpoint_metadata.get("gop_size") != 5:
            raise RuntimeError("R(2+1)D checkpoint was not trained with 5 frames")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor(KINETICS_MEAN).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(KINETICS_STD).view(1, 1, 3, 1, 1)
        )
        self.eval()

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[1:3] != (5, 3):
            raise ValueError(
                f"Expected reconstructed clips [B,5,3,H,W], got {tuple(frames.shape)}"
            )
        return self.model((frames.clamp(0, 1) - self.mean) / self.std)


class ImportanceScorer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latent):
        return self.net(latent)


class GumbelGlobalTopK(nn.Module):
    """Exact-budget selector over a complete GoP."""

    def __init__(self, channels, keep_fraction=0.9, tau=1.0, random_seed=1729):
        super().__init__()
        self.scorer = ImportanceScorer(channels)
        self.keep_fraction = 0.0
        self.tau = 0.0
        self.set_keep_fraction(keep_fraction)
        self.set_temperature(tau)
        self.mode = "spatiotemporal"
        self.random_seed = int(random_seed)
        self._random_row_offset = 0

    def set_mode(self, mode):
        valid = {
            "spatiotemporal", "spatial_only", "random_spatiotemporal",
            "random_spatial_only", "bypass",
        }
        if mode not in valid:
            raise ValueError(f"selector mode must be one of {sorted(valid)}")
        self.mode = mode

    def set_keep_fraction(self, value):
        if not 0.0 < value <= 1.0:
            raise ValueError("keep_fraction must be in (0, 1]")
        self.keep_fraction = float(value)

    def set_temperature(self, value):
        if value <= 0:
            raise ValueError("Gumbel temperature must be positive")
        self.tau = float(value)

    def reset_random_sequence(self):
        self._random_row_offset = 0

    @staticmethod
    def _sample_gumbel_like(tensor, eps=1e-6):
        uniform = torch.rand_like(tensor).clamp_(eps, 1.0 - eps)
        return -torch.log(-torch.log(uniform))

    @staticmethod
    def _hard_topk(scores, count):
        indices = scores.topk(count, dim=1, largest=True, sorted=False).indices
        hard = torch.zeros_like(scores)
        hard.scatter_(1, indices, 1.0)
        return hard

    def _random_hard(self, batch, locations, count, device, dtype):
        rows = []
        for row_index in range(batch):
            generator = torch.Generator(device=device)
            generator.manual_seed(
                self.random_seed + self._random_row_offset + row_index
            )
            indices = torch.randperm(
                locations, generator=generator, device=device
            )[:count]
            row = torch.zeros(locations, device=device, dtype=dtype)
            row.scatter_(0, indices, 1.0)
            rows.append(row)
        self._random_row_offset += batch
        return torch.stack(rows)

    def _straight_through(self, hard, ranking_scores, count):
        # A budget-normalised soft relaxation supplies gradients while the
        # forward value remains exactly binary through the STE identity.
        soft = torch.softmax(ranking_scores / self.tau, dim=1) * count
        soft = soft.clamp(max=1.0)
        return hard + soft - soft.detach()

    def forward(self, latent, batch, frames):
        flat_scores = self.scorer(latent)
        scores = flat_scores.reshape(
            batch, frames, 1, flat_scores.shape[-2], flat_scores.shape[-1]
        )
        spatial_locations = scores.shape[-2] * scores.shape[-1]

        if self.mode == "bypass":
            hard = torch.ones_like(scores)
            return hard.reshape_as(flat_scores), flat_scores, hard, spatial_locations * frames

        if self.mode in {"spatial_only", "random_spatial_only"}:
            base_scores = scores.mean(dim=1).flatten(1)
            per_frame_count = max(
                1, min(spatial_locations, round(self.keep_fraction * spatial_locations))
            )
            if self.mode == "random_spatial_only":
                hard_spatial = self._random_hard(
                    batch, spatial_locations, per_frame_count,
                    scores.device, scores.dtype,
                )
                gate_spatial = hard_spatial
            elif self.training:
                ranking = base_scores + self._sample_gumbel_like(base_scores)
                hard_spatial = self._hard_topk(ranking, per_frame_count)
                gate_spatial = self._straight_through(
                    hard_spatial, ranking, per_frame_count
                )
            else:
                ranking = base_scores
                hard_spatial = self._hard_topk(ranking, per_frame_count)
                gate_spatial = hard_spatial
            hard = hard_spatial[:, None, :].expand(-1, frames, -1)
            gate = gate_spatial[:, None, :].expand(-1, frames, -1)
            hard = hard.reshape(batch, frames, 1, *scores.shape[-2:])
            gate = gate.reshape(batch, frames, 1, *scores.shape[-2:])
            hard = hard.reshape(batch, frames, 1, *scores.shape[-2:])
            gate = gate.reshape(batch, frames, 1, *scores.shape[-2:])
            keep_count = per_frame_count * frames
        else:
            base_scores = scores.flatten(1)
            locations = base_scores.shape[1]
            per_frame_equivalent = max(
                1,
                min(
                    spatial_locations,
                    round(self.keep_fraction * spatial_locations),
                ),
            )
            keep_count = per_frame_equivalent * frames
            if self.mode == "random_spatiotemporal":
                hard = self._random_hard(
                    batch, locations, keep_count, scores.device, scores.dtype
                )
                gate = hard
            else:
                ranking = base_scores
                if self.training:
                    ranking = ranking + self._sample_gumbel_like(ranking)
                hard = self._hard_topk(ranking, keep_count)
                gate = (
                    self._straight_through(hard, ranking, keep_count)
                    if self.training else hard
                )
            hard = hard.reshape(batch, frames, 1, *scores.shape[-2:])
            gate = gate.reshape(batch, frames, 1, *scores.shape[-2:])

        return gate.reshape_as(flat_scores), flat_scores, hard, keep_count


class TAVideoJSCCGumbelGlobalTopKR2Plus1D(nn.Module):
    def __init__(
        self,
        c,
        channel_type="AWGN",
        snr=None,
        P=1.0,
        n_frames=5,
        hidden_dim=16,
        r2plus1d_ckpt=None,
        lambda_task=0.001,
        lambda_recon=1.0,
        keep_fraction=0.9,
        tau=1.0,
        random_seed=1729,
    ):
        super().__init__()
        if channel_type != "AWGN":
            raise NotImplementedError("The audited pilot currently supports AWGN only")
        self.n_frames = n_frames
        self.lambda_task = float(lambda_task)
        self.lambda_recon = float(lambda_recon)
        self.jscc = DeepJSCC(c=c, channel_type=channel_type, snr=snr, P=P)
        self.temporal = TemporalFusionModule(channels=3, hidden_dim=hidden_dim)
        self.selector = GumbelGlobalTopK(
            channels=2 * c,
            keep_fraction=keep_fraction,
            tau=tau,
            random_seed=random_seed,
        )
        if r2plus1d_ckpt is None:
            raise ValueError("A locked R(2+1)D checkpoint is required")
        self.task_model = FrozenR2Plus1D(r2plus1d_ckpt)
        self.last_selection_audit = {}

    def set_selector_mode(self, mode):
        self.selector.set_mode(mode)

    def reset_random_sequence(self):
        self.selector.reset_random_sequence()

    def set_keep_fraction(self, value):
        self.selector.set_keep_fraction(value)

    def set_temperature(self, value):
        self.selector.set_temperature(value)

    def set_codec_trainable(self, trainable):
        for module in (self.jscc, self.temporal):
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)

    def forward(self, clips):
        if clips.ndim != 5:
            raise ValueError(f"Expected [B,T,3,H,W], got {tuple(clips.shape)}")
        batch, frames, channels, height, width = clips.shape
        if frames != self.n_frames:
            raise ValueError(f"Expected {self.n_frames} frames, got {frames}")
        latent = self.jscc.encoder(
            clips.reshape(batch * frames, channels, height, width)
        )
        gate, logits, hard_gop, keep_count = self.selector(latent, batch, frames)
        total_locations = frames * latent.shape[-2] * latent.shape[-1]

        if self.selector.mode == "bypass" or keep_count == total_locations:
            received = self.jscc.channel(latent)
        else:
            masked = latent * gate
            # Exactly K spatial locations are sent. Each retained location has
            # all 2c real-valued latent channels and is normalised accordingly.
            dimensions = latent.shape[1] * keep_count
            masked_gop = masked.reshape(batch, frames, *masked.shape[1:])
            energy = masked_gop.square().sum(
                dim=(1, 2, 3, 4), keepdim=True
            ).clamp_min(1e-12)
            transmitted = math.sqrt(dimensions) * masked_gop / energy.sqrt()
            transmitted = transmitted.reshape_as(latent)
            snr_linear = 10.0 ** (self.jscc.channel.snr / 10.0)
            noise_std = math.sqrt(1.0 / (2.0 * snr_linear))
            noise = torch.randn_like(transmitted) * noise_std
            hard_flat = hard_gop.reshape(batch * frames, 1, *latent.shape[-2:])
            received = transmitted + noise * hard_flat

        hard_counts = hard_gop.detach().sum(dim=(2, 3, 4))
        self.last_selection_audit = {
            "mode": self.selector.mode,
            "keep_fraction_target": self.selector.keep_fraction,
            "keep_count": int(keep_count),
            "locations": int(total_locations),
            "hard_fraction": float(keep_count / total_locations),
            "hard_count_min": int(hard_gop.detach().flatten(1).sum(1).min().item()),
            "hard_count_max": int(hard_gop.detach().flatten(1).sum(1).max().item()),
            "frame_keep_counts": hard_counts.float().mean(dim=0).cpu().tolist(),
            "tau": self.selector.tau,
            "soft_score_mean": torch.sigmoid(logits.detach()).mean().item(),
        }
        decoded = self.jscc.decoder(received).reshape(
            batch, frames, channels, height, width
        )
        return self.temporal(decoded), gate, logits

    def joint_loss(self, clips, labels):
        reconstructed, gate, logits = self.forward(clips)
        reconstruction_loss = F.mse_loss(reconstructed, clips)
        task_logits = self.task_model(reconstructed)
        task_loss = F.cross_entropy(task_logits, labels)
        loss = self.lambda_recon * reconstruction_loss + self.lambda_task * task_loss
        info = {
            "loss": loss.item(),
            "l_task": task_loss.item(),
            "l_recon": reconstruction_loss.item(),
            "psnr": -10.0 * torch.log10(reconstruction_loss.clamp_min(1e-12)).item(),
            "hard_fraction": self.last_selection_audit["hard_fraction"],
            "score_mean": torch.sigmoid(logits.detach()).mean().item(),
            "tau": self.selector.tau,
        }
        return loss, info, reconstructed, task_logits

    def change_channel(self, channel_type="AWGN", snr=None):
        if channel_type != "AWGN":
            raise NotImplementedError("The audited pilot currently supports AWGN only")
        self.jscc.change_channel(channel_type, snr)

    def get_channel(self):
        return self.jscc.get_channel()
