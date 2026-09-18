# -*- coding: utf-8 -*-
"""Saliency-supervised spatiotemporal power allocation for VideoJSCC.

All latent symbols are transmitted. Encoder width ``c`` therefore remains the
only bandwidth/CBR setting. A learned positive allocation map redistributes a
fixed total GoP energy budget. The uniform mode is the matched no-selector
control. A frozen R(2+1)D supplies both task loss and a detached latent-saliency
teacher during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.jscc import DeepJSCC
from model.temporal import TemporalFusionModule
from model.gop_awgn_channel import GoPAWGNChannel
from downstream.action_recognition.models.r2plus1d_recognizer import (
    KINETICS_MEAN,
    KINETICS_STD,
    R2Plus1DRecognizer,
    load_r2plus1d_checkpoint,
)


class FrozenR2Plus1D(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.model = R2Plus1DRecognizer(pretrained=False, num_classes=101)
        self.checkpoint_metadata = load_r2plus1d_checkpoint(
            self.model, checkpoint_path
        )
        if self.checkpoint_metadata.get("gop_size") != 5:
            raise RuntimeError("R(2+1)D checkpoint was not trained with five frames")
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


class SpatiotemporalImportanceScorer(nn.Module):
    """3D scorer that observes space and all five frames jointly."""

    def __init__(self, channels, hidden_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(
                channels, hidden_channels, kernel_size=(3, 3, 3), padding=1
            ),
            nn.PReLU(),
            nn.Conv3d(
                hidden_channels, hidden_channels, kernel_size=(3, 3, 3), padding=1
            ),
            nn.PReLU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Begin from the exact uniform-allocation baseline.
        final_conv = self.net[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

    def forward(self, latent_gop):
        if latent_gop.ndim != 5:
            raise ValueError("Expected latent GoP [B,T,C,H,W]")
        # Conv3d convention is [B,C,T,H,W]. Return [B,T,1,H,W].
        scores = self.net(latent_gop.permute(0, 2, 1, 3, 4))
        return scores.permute(0, 2, 1, 3, 4).contiguous()


class NormalizedImportanceAllocator(nn.Module):
    """Map scores to positive mean-one relative power allocations."""

    def __init__(self, min_relative_power=0.1, temperature=1.0):
        super().__init__()
        if not 0.0 <= min_relative_power < 1.0:
            raise ValueError("min_relative_power must be in [0,1)")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.min_relative_power = float(min_relative_power)
        self.temperature = float(temperature)

    def forward(self, scores):
        flat = scores.flatten(1)
        locations = flat.shape[1]
        distribution = torch.softmax(flat / self.temperature, dim=1)
        allocation = (
            self.min_relative_power
            + (1.0 - self.min_relative_power) * locations * distribution
        )
        return allocation.view_as(scores)


class TAVideoJSCCSaliencySpatiotemporalR2Plus1D(nn.Module):
    """Complete uniform/learned GoP-allocation experiment model."""

    def __init__(
        self,
        c,
        snr=13.0,
        power=1.0,
        n_frames=5,
        hidden_dim=16,
        scorer_hidden=16,
        r2plus1d_ckpt=None,
        lambda_task=0.001,
        lambda_recon=1.0,
        lambda_importance=0.01,
        allocation_temperature=1.0,
        min_relative_power=0.1,
    ):
        super().__init__()
        if n_frames != 5:
            raise ValueError("The locked evaluator requires five-frame GoPs")
        if r2plus1d_ckpt is None:
            raise ValueError("A locked R(2+1)D checkpoint is required")
        self.n_frames = n_frames
        self.power = float(power)
        self.lambda_task = float(lambda_task)
        self.lambda_recon = float(lambda_recon)
        self.lambda_importance = float(lambda_importance)
        # Use only the existing encoder/decoder. Its original per-frame channel
        # is not called anywhere in this experimental model.
        self.jscc = DeepJSCC(c=c, channel_type="AWGN", snr=None, P=power)
        self.temporal = TemporalFusionModule(channels=3, hidden_dim=hidden_dim)
        self.gop_channel = GoPAWGNChannel(snr=snr, power=power)
        self.scorer = SpatiotemporalImportanceScorer(
            channels=2 * c, hidden_channels=scorer_hidden
        )
        self.allocator = NormalizedImportanceAllocator(
            min_relative_power=min_relative_power,
            temperature=allocation_temperature,
        )
        self.task_model = FrozenR2Plus1D(r2plus1d_ckpt)
        self.allocation_mode = "spatiotemporal"
        self.last_allocation_audit = {}

    def set_allocation_mode(self, mode):
        if mode not in {"uniform", "spatiotemporal"}:
            raise ValueError("allocation mode must be uniform or spatiotemporal")
        self.allocation_mode = mode

    @staticmethod
    def _normalise_gop_energy(weighted, reference, eps=1e-12):
        dims = tuple(range(1, weighted.ndim))
        reference_energy = reference.square().sum(dim=dims, keepdim=True)
        weighted_energy = weighted.square().sum(dim=dims, keepdim=True)
        scale = torch.sqrt(reference_energy / weighted_energy.clamp_min(eps))
        return weighted * scale, scale

    def encode(self, clips):
        batch, frames, channels, height, width = clips.shape
        latent = self.jscc.encoder(
            clips.reshape(batch * frames, channels, height, width)
        )
        return latent.reshape(batch, frames, *latent.shape[1:])

    def decode(self, received_gop, output_shape):
        batch, frames, channels, height, width = output_shape
        flat = received_gop.reshape(batch * frames, *received_gop.shape[2:])
        decoded = self.jscc.decoder(flat).reshape(
            batch, frames, channels, height, width
        )
        return self.temporal(decoded)

    def allocate(self, latent_gop):
        scores = self.scorer(latent_gop)
        if self.allocation_mode == "uniform":
            relative_power = torch.ones_like(scores)
        else:
            relative_power = self.allocator(scores)
        allocation_amplitude = relative_power.clamp_min(1e-12).sqrt()
        weighted = latent_gop * allocation_amplitude
        transmitted, global_scale = self._normalise_gop_energy(
            weighted, latent_gop
        )
        # The complete transmitter gain must be known at the receiver, just as
        # SoftCast communicates/derives its coefficient scaling factors.
        receiver_gain = allocation_amplitude * global_scale
        return transmitted, relative_power, scores, receiver_gain

    def forward_from_latent(self, latent_gop, output_shape):
        transmitted, relative_power, scores, receiver_gain = self.allocate(latent_gop)
        received = self.gop_channel(transmitted)
        # Restore the latent signal scale before the unchanged neural decoder.
        # Allocation remains effective because the fixed AWGN is divided by
        # the same gain: high-power locations have lower effective noise and
        # low-power locations have higher effective noise.
        equalized = received / receiver_gain.clamp_min(1e-6)
        reconstructed = self.decode(equalized, output_shape)
        dims = tuple(range(1, latent_gop.ndim))
        reference_energy = latent_gop.detach().square().sum(dim=dims)
        transmit_energy = transmitted.detach().square().sum(dim=dims)
        frame_energy = transmitted.detach().square().sum(dim=(2, 3, 4))
        total_energy = frame_energy.sum(dim=1, keepdim=True)
        self.last_allocation_audit = {
            "mode": self.allocation_mode,
            "all_symbols_transmitted": True,
            "relative_power_mean": relative_power.detach().mean().item(),
            "relative_power_min": relative_power.detach().min().item(),
            "relative_power_max": relative_power.detach().max().item(),
            "relative_power_cv": (
                relative_power.detach().std(unbiased=False)
                / relative_power.detach().mean().clamp_min(1e-12)
            ).item(),
            "energy_ratio_min": (
                transmit_energy / reference_energy.clamp_min(1e-12)
            ).min().item(),
            "energy_ratio_max": (
                transmit_energy / reference_energy.clamp_min(1e-12)
            ).max().item(),
            "frame_energy_share": (
                frame_energy / total_energy.clamp_min(1e-12)
            ).mean(dim=0).cpu().tolist(),
            "noise_variance": self.gop_channel.last_audit["noise_variance"],
            "noise_variance_is_frame_independent": True,
            "receiver_gain_min": receiver_gain.detach().min().item(),
            "receiver_gain_max": receiver_gain.detach().max().item(),
            "allocation_map_available_at_receiver": True,
        }
        return reconstructed, relative_power, scores

    def forward(self, clips):
        if clips.ndim != 5 or clips.shape[1:3] != (5, 3):
            raise ValueError(f"Expected clips [B,5,3,H,W], got {tuple(clips.shape)}")
        latent_gop = self.encode(clips)
        return self.forward_from_latent(latent_gop, clips.shape)

    def dense_saliency_target(self, latent_gop, clips_shape, labels):
        """Detached |z*dL_task/dz| teacher from a dense uniform transmission."""
        teacher_latent = latent_gop.detach().requires_grad_(True)
        with torch.enable_grad():
            # Use a noiseless dense teacher so saliency is not dominated by a
            # particular random channel draw.
            teacher_reconstruction = self.decode(teacher_latent, clips_shape)
            teacher_logits = self.task_model(teacher_reconstruction)
            teacher_loss = F.cross_entropy(teacher_logits, labels)
            gradient = torch.autograd.grad(
                teacher_loss, teacher_latent, create_graph=False,
                retain_graph=False, only_inputs=True,
            )[0]
        saliency = (teacher_latent.detach() * gradient.detach()).abs().mean(
            dim=2, keepdim=True
        )
        flat = saliency.flatten(1).clamp_min(1e-12)
        return (flat / flat.sum(dim=1, keepdim=True)).view_as(saliency)

    @staticmethod
    def importance_loss(scores, target_distribution):
        log_prediction = F.log_softmax(scores.flatten(1), dim=1)
        target = target_distribution.flatten(1)
        return F.kl_div(log_prediction, target, reduction="batchmean")

    def joint_loss(self, clips, labels, use_importance_teacher=True):
        latent_gop = self.encode(clips)
        target = None
        if self.allocation_mode == "spatiotemporal" and use_importance_teacher:
            target = self.dense_saliency_target(latent_gop, clips.shape, labels)
        reconstructed, relative_power, scores = self.forward_from_latent(
            latent_gop, clips.shape
        )
        reconstruction_loss = F.mse_loss(reconstructed, clips)
        logits = self.task_model(reconstructed)
        task_loss = F.cross_entropy(logits, labels)
        if target is None:
            importance_loss = scores.sum() * 0.0
        else:
            importance_loss = self.importance_loss(scores, target)
        loss = (
            self.lambda_recon * reconstruction_loss
            + self.lambda_task * task_loss
            + self.lambda_importance * importance_loss
        )
        audit = self.last_allocation_audit
        info = {
            "loss": loss.item(),
            "l_recon": reconstruction_loss.item(),
            "l_task": task_loss.item(),
            "l_importance": importance_loss.item(),
            "psnr": -10.0 * torch.log10(
                reconstruction_loss.clamp_min(1e-12)
            ).item(),
            "relative_power_mean": audit["relative_power_mean"],
            "relative_power_cv": audit["relative_power_cv"],
        }
        return loss, info, reconstructed, logits

    def change_snr(self, snr):
        self.gop_channel.snr = float(snr)

    def get_channel(self):
        return self.gop_channel.get_channel()
