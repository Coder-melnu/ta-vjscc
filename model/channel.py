# -*- coding: utf-8 -*-
"""
Channel module for Deep-JSCC.
Supports: AWGN, Rayleigh fading

Note: original channel.py from Deep-JSCC-PyTorch is kept intact here.
Rayleigh fading will be added in Week 2.
"""

import torch
import torch.nn as nn
import math


class Channel(nn.Module):
    """
    Wireless channel simulation layer.

    Args:
        channel_type : 'AWGN' or 'Rayleigh'
        snr          : signal-to-noise ratio in dB
    """

    SUPPORTED = ('AWGN', 'Rayleigh')

    def __init__(self, channel_type: str = 'AWGN', snr: float = 10.0):
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr

        if channel_type not in self.SUPPORTED:
            raise ValueError(f"channel_type must be one of {self.SUPPORTED}, got '{channel_type}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_type == 'AWGN':
            return self._awgn(x)
        elif self.channel_type == 'Rayleigh':
            return self._rayleigh(x)

    def _awgn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Additive White Gaussian Noise channel.
        Noise power is computed from SNR (dB) assuming unit signal power
        after power normalization in the encoder.
        """
        snr_linear = 10 ** (self.snr / 10.0)
        # Signal power: E[||x||^2] / k = 1 after normalization
        # Noise variance per dimension: 1 / (2 * snr_linear)
        noise_std = math.sqrt(1.0 / (2 * snr_linear))
        noise = torch.randn_like(x) * noise_std
        return x + noise

    def _rayleigh(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rayleigh flat-fading channel (real-valued approximation).
        h ~ Rayleigh(1/sqrt(2)) per sample, with AWGN noise added.

        Each sample in the batch gets an independent fading coefficient.
        Channel uses perfect CSI at receiver (equalization via h).

        TODO (Week 2): implement full complex Rayleigh with equalization.
        """
        snr_linear = 10 ** (self.snr / 10.0)
        noise_std  = math.sqrt(1.0 / (2 * snr_linear))

        # Rayleigh fading: h = |h_c| where h_c ~ CN(0, 1)
        # Real-valued approximation: sample two Gaussian components per batch sample
        batch_size = x.size(0)
        h_real = torch.randn(batch_size, 1, 1, 1, device=x.device)
        h_imag = torch.randn(batch_size, 1, 1, 1, device=x.device)
        h_mag  = torch.sqrt(h_real**2 + h_imag**2) / math.sqrt(2)  # E[h^2] = 1

        # Apply fading + noise
        noise  = torch.randn_like(x) * noise_std
        y      = h_mag * x + noise

        # Equalization (divide by fading magnitude — perfect CSI assumption)
        y_eq   = y / (h_mag + 1e-8)
        return y_eq

    def get_channel(self) -> dict:
        return {'type': self.channel_type, 'snr_db': self.snr}

    def __repr__(self):
        return f"Channel(type={self.channel_type}, snr={self.snr}dB)"