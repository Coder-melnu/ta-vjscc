# -*- coding: utf-8 -*-
"""
Temporal Fusion Module for Video DeepJSCC (Approach A).

Takes N independently decoded frames and applies 3D convolutions
to exploit temporal redundancy at the decoder side.

Input  : (B, N, 3, H, W)  — N decoded frames
Output : (B, N, 3, H, W)  — temporally refined frames
"""

import torch
import torch.nn as nn


class TemporalFusionModule(nn.Module):
    """
    Lightweight 3D Conv temporal fusion module.

    Args:
        channels   : number of image channels (3 for RGB)
        hidden_dim : number of hidden feature channels
    """

    def __init__(self, channels: int = 3, hidden_dim: int = 16):
        super(TemporalFusionModule, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels=channels,
            out_channels=hidden_dim,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)       # same padding — preserves (N, H, W)
        )
        self.conv2 = nn.Conv3d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )
        self.conv3 = nn.Conv3d(
            in_channels=hidden_dim,
            out_channels=channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)       # no temporal mixing at output
        )

        self.prelu1 = nn.PReLU(num_parameters=hidden_dim)
        self.prelu2 = nn.PReLU(num_parameters=hidden_dim)
        self.alpha   = nn.Parameter(torch.tensor(0.1))  # learnable residual weight

        # Kaiming init for PReLU layers
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        nn.init.xavier_normal_(self.conv3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, 3, H, W) — N decoded frames
        Returns:
            (B, N, 3, H, W) — temporally refined frames
        """
        # Rearrange to (B, C, N, H, W) for Conv3d
        x_in = x.permute(0, 2, 1, 3, 4)            # (B, 3, N, H, W)

        out = self.prelu1(self.conv1(x_in))  # (B, hidden, N, H, W)
        out = self.prelu2(self.conv2(out))  # (B, hidden, N, H, W)       
        out = self.conv3(out)                         # (B, 3, N, H, W)

        # Learnable weighted residual — module learns a small correction delta
        out = (self.alpha * out + x_in).clamp(0, 1)  # (B, 3, N, H, W)

        # Rearrange back to (B, N, 3, H, W)
        return out.permute(0, 2, 1, 3, 4)


if __name__ == '__main__':
    module = TemporalFusionModule(channels=3, hidden_dim=16)
    x = torch.rand(4, 5, 3, 256, 256)
    out = module(x)
    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")
    assert x.shape == out.shape, "Shape mismatch!"
    n_params = sum(p.numel() for p in module.parameters())
    print(f"Parameters: {n_params:,}")
    print("TemporalFusionModule OK")