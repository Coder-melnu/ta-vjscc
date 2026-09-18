# -*- coding: utf-8 -*-
"""
Difference-aware RGB Temporal Fusion Module.

Input:
    Decoded RGB GoP: (B, N, 3, H, W)

The module explicitly uses:
    1. Current decoded RGB frames
    2. Backward differences: x_t - x_{t-1}
    3. Forward differences:  x_{t+1} - x_t

Output:
    Residually refined RGB GoP: (B, N, 3, H, W)
"""

import torch
import torch.nn as nn


class TemporalFusionModule(nn.Module):
    """
    Difference-aware decoder-side temporal fusion.

    The residual branch is initialized close to zero, so the initial
    module behaves approximately like the identity function.
    """

    def __init__(self, channels: int = 3, hidden_dim: int = 16):
        super().__init__()

        self.channels = channels

        # Only backward and forward temporal differences.
        # Current RGB is available only through the identity residual path.
        input_channels = channels * 2

        # First extract spatial features from the explicit temporal signals.
        self.conv1 = nn.Conv3d(
            in_channels=input_channels,
            out_channels=hidden_dim,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )

        # Explicit temporal-spatial interaction.
        self.conv2 = nn.Conv3d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )

        # Predict an RGB residual correction.
        self.conv3 = nn.Conv3d(
            in_channels=hidden_dim,
            out_channels=channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )

        self.prelu1 = nn.PReLU(num_parameters=hidden_dim)
        self.prelu2 = nn.PReLU(num_parameters=hidden_dim)

        # Learnable strength of the residual correction.
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(
            self.conv1.weight,
            nonlinearity="leaky_relu",
        )
        nn.init.kaiming_normal_(
            self.conv2.weight,
            nonlinearity="leaky_relu",
        )

        # Small nonzero initialization:
        # near identity, while gradients can still reach all layers.
        nn.init.normal_(self.conv3.weight, mean=0.0, std=1e-3)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)

    @staticmethod
    def _temporal_differences(x: torch.Tensor):
        """
        Args:
            x: (B, C, N, H, W)

        Returns:
            backward_diff, forward_diff with the same shape as x.
        """
        backward_diff = torch.zeros_like(x)
        forward_diff = torch.zeros_like(x)

        # x_t - x_{t-1}; first frame has no previous frame.
        backward_diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]

        # x_{t+1} - x_t; last frame has no next frame.
        forward_diff[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]

        return backward_diff, forward_diff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3, H, W)

        Returns:
            Refined frames: (B, N, 3, H, W)

        Note:
            No clamp is applied here. Clamp only for metrics,
            visualization, or saving reconstructed frames.
        """
        if x.dim() != 5:
            raise ValueError(
                f"Expected input shape (B, N, C, H, W), got {tuple(x.shape)}"
            )

        if x.size(2) != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {x.size(2)}"
            )

        # Conv3d format: (B, C, N, H, W)
        x_in = x.permute(0, 2, 1, 3, 4).contiguous()

        backward_diff, forward_diff = self._temporal_differences(x_in)

        features = torch.cat(
            [backward_diff, forward_diff],
            dim=1,
        )

        residual = self.prelu1(self.conv1(features))
        residual = self.prelu2(self.conv2(residual))
        residual = self.conv3(residual)

        # No internal clamp: preserve gradients during training.
        refined = x_in + self.alpha * residual

        return refined.permute(0, 2, 1, 3, 4).contiguous()


if __name__ == "__main__":
    module = TemporalFusionModule(channels=3, hidden_dim=16)

    x = torch.rand(2, 5, 3, 128, 128)
    output = module(x)

    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)
    print("Initial alpha:", module.alpha.item())
    print(
        "Initial mean absolute correction:",
        torch.mean(torch.abs(output - x)).item(),
    )

    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    print("Difference-aware TemporalFusionModule: OK")