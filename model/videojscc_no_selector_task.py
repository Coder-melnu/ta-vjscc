# -*- coding: utf-8 -*-
"""Plain VideoJSCC+TFM with frozen task supervision and no selector.

This control intentionally does not import, construct, call, mask, weight, or
renormalise through any selector implementation.  Its communication path is
the original DeepJSCC path followed by the corrected TemporalFusionModule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.jscc import DeepJSCC
from model.temporal import TemporalFusionModule


class FrozenTSN(nn.Module):
    """Corrected locked TSN used only as a differentiable frozen loss network."""

    def __init__(self, head_ckpt: str):
        super().__init__()
        from downstream.action_recognition.models.tsn_recognizer import (
            TSNModel,
            load_tsn_checkpoint,
        )

        self.model = TSNModel(pretrained=True, num_classes=101)
        self.checkpoint_metadata = load_tsn_checkpoint(self.model, head_ckpt)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        )
        self.eval()
        print(
            f"[FrozenTSN] Loaded {head_ckpt} | "
            f"finetune={self.checkpoint_metadata.get('finetune', 'unknown')}"
        )

    def train(self, mode: bool = True):
        """Never allow frozen ResNet BatchNorm statistics to update."""
        super().train(False)
        self.model.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(
                f"Expected frames shaped (B,N,3,H,W), got {tuple(frames.shape)}"
            )
        normalised = (frames.clamp(0, 1) - self.mean) / self.std
        return self.model(normalised)


class NoSelectorVideoJSCC(nn.Module):
    """Task-fine-tuned VideoJSCC control with no selector parameters or path."""

    def __init__(
        self,
        c: int,
        channel_type: str = "AWGN",
        snr: float = None,
        P: float = 1.0,
        n_frames: int = 5,
        hidden_dim: int = 16,
        tsn_head_ckpt: str = None,
        lambda_task: float = 0.001,
        lambda_recon: float = 1.0,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.n_frames = n_frames
        self.lambda_task = lambda_task
        self.lambda_recon = lambda_recon
        self.jscc = DeepJSCC(c=c, channel_type=channel_type, snr=snr, P=P)
        self.temporal = TemporalFusionModule(channels=3, hidden_dim=hidden_dim)
        if tsn_head_ckpt is None:
            raise ValueError("No-selector task fine-tuning requires a TSN checkpoint")
        self.tsn = FrozenTSN(head_ckpt=tsn_head_ckpt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B,N,3,H,W), got {tuple(x.shape)}")
        B, N, C, H, W = x.shape
        if N != self.n_frames:
            raise ValueError(f"Expected {self.n_frames} frames, got {N}")
        x_hat = self.jscc(x.reshape(B * N, C, H, W))
        return self.temporal(x_hat.reshape(B, N, C, H, W))

    def joint_loss(self, x: torch.Tensor, labels: torch.Tensor):
        reconstructed = self.forward(x)
        l_recon = F.mse_loss(reconstructed, x)
        logits = self.tsn(reconstructed)
        l_task = F.cross_entropy(logits, labels)
        loss = self.lambda_recon * l_recon + self.lambda_task * l_task
        info = {
            "loss": loss.item(),
            "l_task": l_task.item(),
            "l_recon": l_recon.item(),
            "weighted_task": (self.lambda_task * l_task).item(),
            "weighted_recon": (self.lambda_recon * l_recon).item(),
            "psnr": -10.0 * torch.log10(l_recon.clamp_min(1e-12)).item(),
        }
        return loss, info, reconstructed, logits

    def change_channel(self, channel_type: str = "AWGN", snr: float = None):
        self.jscc.change_channel(channel_type, snr)

    def get_channel(self):
        return self.jscc.get_channel()
