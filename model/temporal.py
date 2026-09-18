# -*- coding: utf-8 -*-
"""Difference-aware decoder-side Temporal Fusion Module (TFM v3)."""

import torch
import torch.nn as nn


class TemporalFusionModule(nn.Module):
    """Predict a residual using only backward/forward frame differences."""

    def __init__(self, channels: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv3d(
            channels * 2, hidden_dim, (1, 3, 3), padding=(0, 1, 1), bias=False
        )
        self.conv2 = nn.Conv3d(
            hidden_dim, hidden_dim, (3, 3, 3), padding=(1, 1, 1), bias=False
        )
        self.conv3 = nn.Conv3d(
            hidden_dim, channels, (1, 3, 3), padding=(0, 1, 1), bias=False
        )
        self.prelu1 = nn.PReLU(num_parameters=hidden_dim)
        self.prelu2 = nn.PReLU(num_parameters=hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="leaky_relu")
        nn.init.normal_(self.conv3.weight, mean=0.0, std=1e-3)

    @staticmethod
    def _temporal_differences(x: torch.Tensor):
        backward = torch.zeros_like(x)
        forward = torch.zeros_like(x)
        backward[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        forward[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
        return backward, forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B,N,C,H,W), got {tuple(x.shape)}")
        if x.size(2) != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.size(2)}")
        x_in = x.permute(0, 2, 1, 3, 4).contiguous()
        backward, forward = self._temporal_differences(x_in)
        residual = self.prelu1(self.conv1(torch.cat((backward, forward), dim=1)))
        residual = self.prelu2(self.conv2(residual))
        residual = self.conv3(residual)
        refined = x_in + self.alpha * residual
        return refined.permute(0, 2, 1, 3, 4).contiguous()


if __name__ == "__main__":
    module = TemporalFusionModule()
    sample = torch.rand(2, 5, 3, 128, 128)
    output = module(sample)
    assert output.shape == sample.shape and torch.isfinite(output).all()
    print("TFM v3 OK")
