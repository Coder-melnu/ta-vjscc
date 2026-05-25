# -*- coding: utf-8 -*-
"""
Video DeepJSCC model (Approach A) — assembles DeepJSCC + TemporalFusionModule.

Each frame in the GoP is independently encoded, passed through the channel,
and decoded. The N decoded frames are then refined by the TemporalFusionModule.

Input  : (B, N, 3, H, W)  — GoP of N frames
Output : (B, N, 3, H, W)  — reconstructed frames
"""

import torch
import torch.nn as nn

from .jscc import DeepJSCC
from .temporal import TemporalFusionModule


class VideoJSCC(nn.Module):
    """
    Video DeepJSCC with decoder-side temporal fusion (Approach A).

    Args:
        c            : bottleneck depth (number of complex symbols)
        channel_type : 'AWGN' or 'Rayleigh'
        snr          : channel SNR in dB
        P            : transmit power constraint
        n_frames     : GoP size (N=5)
        hidden_dim   : hidden channels in TemporalFusionModule
    """

    def __init__(self, c: int, channel_type: str = 'AWGN', snr: float = None,
                 P: float = 1.0, n_frames: int = 5, hidden_dim: int = 16):
        super(VideoJSCC, self).__init__()

        self.n_frames = n_frames

        # Shared DeepJSCC — same encoder/channel/decoder for all frames
        self.jscc = DeepJSCC(c=c, channel_type=channel_type, snr=snr, P=P)

        # Decoder-side temporal fusion
        self.temporal = TemporalFusionModule(channels=3, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, 3, H, W) — GoP of N frames
        Returns:
            (B, N, 3, H, W) — reconstructed + temporally refined frames
        """
        B, N, C, H, W = x.shape
        assert N == self.n_frames, f"Expected {self.n_frames} frames, got {N}"

        # Process each frame independently through shared DeepJSCC
        # Reshape to (B*N, C, H, W) — treat all frames as a flat batch
        x_flat = x.view(B * N, C, H, W)            # (B*N, 3, H, W)
        x_hat_flat = self.jscc(x_flat)              # (B*N, 3, H, W)

        # Reshape back to (B, N, 3, H, W)
        x_hat = x_hat_flat.view(B, N, C, H, W)     # (B, N, 3, H, W)

        # Temporal fusion at decoder side
        x_refined = self.temporal(x_hat)            # (B, N, 3, H, W)

        return x_refined

    def forward_no_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-frame baseline forward pass — no temporal fusion.
        Used for ablation: compare VideoJSCC vs per-frame DeepJSCC.

        Args:
            x : (B, N, 3, H, W)
        Returns:
            (B, N, 3, H, W)
        """
        B, N, C, H, W = x.shape
        x_flat = x.view(B * N, C, H, W)
        x_hat_flat = self.jscc(x_flat)
        return x_hat_flat.view(B, N, C, H, W)

    def loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        MSE loss over all N frames.

        Args:
            pred : (B, N, 3, H, W)
            gt   : (B, N, 3, H, W)
        Returns:
            scalar MSE loss
        """
        return nn.MSELoss(reduction='mean')(pred, gt)

    def change_channel(self, channel_type: str = 'AWGN', snr: float = None):
        """Hot-swap channel at test time."""
        self.jscc.change_channel(channel_type, snr)

    def get_channel(self) -> dict:
        return self.jscc.get_channel()


if __name__ == '__main__':
    model = VideoJSCC(c=16, channel_type='AWGN', snr=10, n_frames=5)
    print(model)

    x = torch.rand(2, 5, 3, 256, 256)      # batch=2, N=5, RGB, 256x256
    out = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")
    assert x.shape == out.shape, "Shape mismatch!"

    # Test ablation forward
    out_no_temp = model.forward_no_temporal(x)
    assert out_no_temp.shape == x.shape
    print(f"No-temporal output: {out_no_temp.shape}")

    # Test loss
    loss = model.loss(out, x)
    print(f"Loss: {loss.item():.6f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    print("VideoJSCC OK")