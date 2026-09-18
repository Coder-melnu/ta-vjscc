# -*- coding: utf-8 -*-
"""GoP-level fixed-variance AWGN channel for allocation experiments.

This module is intentionally separate from ``model/channel.py``. It accepts a
complete video GoP and uses one transmitter power convention for every frame,
so reallocating signal power between frames does not also rescale their noise.
"""

import math

import torch
import torch.nn as nn


class GoPAWGNChannel(nn.Module):
    """Fixed-variance complex AWGN for tensors shaped [B,T,C,H,W]."""

    def __init__(self, snr=20.0, power=1.0):
        super().__init__()
        if power <= 0:
            raise ValueError("power must be positive")
        self.snr = float(snr)
        self.power = float(power)
        self.last_audit = {}

    def forward(self, transmitted):
        if transmitted.ndim != 5:
            raise ValueError(
                f"Expected GoP tensor [B,T,C,H,W], got {tuple(transmitted.shape)}"
            )
        snr_linear = 10.0 ** (self.snr / 10.0)
        # Match the repository's complex-AWGN convention. Unlike Channel,
        # this value is fixed by P and SNR, not recalculated per frame.
        noise_variance = self.power / (2.0 * snr_linear)
        noise_std = math.sqrt(noise_variance)
        noise = torch.randn_like(transmitted) * noise_std
        received = transmitted + noise
        frame_signal_power = transmitted.detach().square().mean(dim=(2, 3, 4))
        self.last_audit = {
            "snr_db": self.snr,
            "power_constraint": self.power,
            "noise_variance": noise_variance,
            "noise_std": noise_std,
            "noise_variance_is_frame_independent": True,
            "frame_signal_power_mean": frame_signal_power.mean(dim=0).cpu().tolist(),
        }
        return received

    def get_channel(self):
        return "GoP-AWGN", self.snr
