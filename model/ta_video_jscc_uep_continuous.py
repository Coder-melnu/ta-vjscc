# -*- coding: utf-8 -*-
"""Task-aware VideoJSCC with continuous bounded spatial power allocation.

Every latent symbol is transmitted. Within each encoded frame, learned
importance scores produce continuous relative powers through a bounded tanh
mapping. Per-frame mean normalisation and exact sample-wise energy
renormalisation preserve the communication budget.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.jscc import DeepJSCC
from model.temporal import TemporalFusionModule
from model.videojscc_no_selector_task import FrozenTSN


class ImportancePowerScorer(nn.Module):
    """Lightweight spatial scorer operating on the encoder latent tensor."""

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


class ContinuousSpatialUEP(nn.Module):
    """Assign learned continuous spatial power under a fixed total budget."""

    def __init__(
        self,
        channels,
        alpha=0.2,
        random_seed=1729,
    ):
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        self.scorer = ImportancePowerScorer(channels)
        self.alpha = float(alpha)
        self.random_seed = int(random_seed)
        self.mode = "learned"
        self._random_row_offset = 0

    def set_mode(self, mode):
        valid = {"learned", "random", "all_ones", "bypass"}
        if mode not in valid:
            raise ValueError(f"mode must be one of {sorted(valid)}")
        self.mode = mode

    def reset_random_sequence(self):
        self._random_row_offset = 0

    def _scores_to_power(self, scores):
        # Centre independently within every encoded frame.
        spatial_dims = tuple(range(1, scores.ndim))
        centred = scores - scores.mean(dim=spatial_dims, keepdim=True)

        # Continuous bounded perturbation before mean normalisation.
        raw_power = 1.0 + self.alpha * torch.tanh(centred)

        # Exact unit mean for every encoded frame.
        power = raw_power / raw_power.mean(
            dim=spatial_dims, keepdim=True
        ).clamp_min(1e-12)
        return power

    def _randomise(self, power):
        flat = power.flatten(1)
        rows = []
        for row_index, row in enumerate(flat):
            generator = torch.Generator(device=row.device)
            generator.manual_seed(
                self.random_seed + self._random_row_offset + row_index
            )
            permutation = torch.randperm(
                row.numel(), generator=generator, device=row.device
            )
            rows.append(row[permutation])
        self._random_row_offset += flat.shape[0]
        return torch.stack(rows).view_as(power)

    def forward(self, latent):
        scores = self.scorer(latent)
        power = self._scores_to_power(scores)

        if self.mode == "random":
            power = self._randomise(power)
        elif self.mode in {"all_ones", "bypass"}:
            power = torch.ones_like(power)

        return power, scores


class TAVideoJSCCContinuousUEP(nn.Module):
    """Jointly fine-tuned VideoJSCC with continuous spatial UEP."""

    def __init__(
        self,
        c,
        channel_type="AWGN",
        snr=None,
        P=1.0,
        n_frames=5,
        hidden_dim=16,
        tsn_head_ckpt=None,
        lambda_task=0.001,
        lambda_recon=1.0,
        alpha=0.2,
        random_seed=1729,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.lambda_task = lambda_task
        self.lambda_recon = lambda_recon
        self.jscc = DeepJSCC(c=c, channel_type=channel_type, snr=snr, P=P)
        self.temporal = TemporalFusionModule(channels=3, hidden_dim=hidden_dim)
        self.allocator = ContinuousSpatialUEP(
            channels=2 * c,
            alpha=alpha,
            random_seed=random_seed,
        )
        if tsn_head_ckpt is None:
            raise ValueError("A corrected locked TSN checkpoint is required")
        self.tsn = FrozenTSN(tsn_head_ckpt)
        self.last_power_audit = {}
        self.last_relative_power = None
        self.last_spatial_power = None

    @staticmethod
    def _renormalize_to_reference(weighted, reference, eps=1e-12):
        dimensions = tuple(range(1, weighted.ndim))
        reference_energy = reference.square().sum(dim=dimensions, keepdim=True)
        weighted_energy = weighted.square().sum(dim=dimensions, keepdim=True)
        scale = torch.sqrt(reference_energy / weighted_energy.clamp_min(eps))
        return weighted * scale

    def set_allocation_mode(self, mode):
        self.allocator.set_mode(mode)

    def reset_random_sequence(self):
        self.allocator.reset_random_sequence()

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected (B,N,3,H,W), got {tuple(x.shape)}")
        batch, frames, channels, height, width = x.shape
        if frames != self.n_frames:
            raise ValueError(f"Expected {self.n_frames} frames, got {frames}")
        flat = x.reshape(batch * frames, channels, height, width)
        latent = self.jscc.encoder(flat)
        relative_power, scores = self.allocator(latent)

        # Power p corresponds to amplitude sqrt(p).
        weighted = latent * relative_power.clamp_min(1e-12).sqrt()
        if self.allocator.mode == "bypass":
            channel_input = latent
        else:
            channel_input = self._renormalize_to_reference(weighted, latent)

        dimensions = tuple(range(1, latent.ndim))
        reference_energy = latent.detach().square().sum(dim=dimensions)
        channel_energy = channel_input.detach().square().sum(dim=dimensions)
        actual_spatial_power = channel_input.detach().square().sum(dim=1, keepdim=True)
        actual_spatial_power = actual_spatial_power / actual_spatial_power.flatten(1).sum(
            dim=1, keepdim=True
        ).view(-1, 1, 1, 1).clamp_min(1e-12)
        self.last_relative_power = relative_power.detach()
        self.last_spatial_power = actual_spatial_power
        self.last_power_audit = {
            "mode": self.allocator.mode,
            "relative_power_mean": relative_power.detach().mean().item(),
            "relative_power_std": relative_power.detach().std(unbiased=False).item(),
            "relative_power_min": relative_power.detach().min().item(),
            "relative_power_max": relative_power.detach().max().item(),
            "score_mean": scores.detach().mean().item(),
            "score_std": scores.detach().std(unbiased=False).item(),
            "all_symbols_transmitted": bool(relative_power.detach().min().item() > 0),
            "energy_ratio_mean": (channel_energy / reference_energy.clamp_min(1e-12)).mean().item(),
            "energy_ratio_min": (channel_energy / reference_energy.clamp_min(1e-12)).min().item(),
            "energy_ratio_max": (channel_energy / reference_energy.clamp_min(1e-12)).max().item(),
        }

        received = self.jscc.channel(channel_input) if self.jscc.channel is not None else channel_input
        decoded = self.jscc.decoder(received).reshape(
            batch, frames, channels, height, width
        )
        reconstructed = self.temporal(decoded)
        return reconstructed, relative_power, scores

    def joint_loss(self, x, labels):
        reconstructed, relative_power, _ = self.forward(x)
        reconstruction_loss = F.mse_loss(reconstructed, x)
        logits = self.tsn(reconstructed)
        task_loss = F.cross_entropy(logits, labels)
        loss = self.lambda_recon * reconstruction_loss + self.lambda_task * task_loss
        power_mean = relative_power.mean()
        power_std = relative_power.std(unbiased=False)
        info = {
            "loss": loss.item(),
            "l_task": task_loss.item(),
            "l_recon": reconstruction_loss.item(),
            "psnr": -10.0 * torch.log10(reconstruction_loss.clamp_min(1e-12)).item(),
            "relative_power_mean": power_mean.item(),
            "relative_power_std": power_std.item(),
            "relative_power_cv": (power_std / power_mean.clamp_min(1e-12)).item(),
            "relative_power_min": self.last_power_audit["relative_power_min"],
            "relative_power_max": self.last_power_audit["relative_power_max"],
        }
        return loss, info, reconstructed, logits

    def change_channel(self, channel_type="AWGN", snr=None):
        self.jscc.change_channel(channel_type, snr)

    def get_channel(self):
        return self.jscc.get_channel()
