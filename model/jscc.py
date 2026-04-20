# -*- coding: utf-8 -*-
"""
DeepJSCC model — assembles Encoder, Channel, and Decoder.
"""

import torch
import torch.nn as nn

from .encoder import _Encoder, ratio2filtersize
from .decoder import _Decoder
from .channel import Channel


class DeepJSCC(nn.Module):
    """
    Deep Joint Source-Channel Coding model.

    Args:
        c            : bottleneck depth (number of complex symbols); controls bandwidth ratio
        channel_type : 'AWGN' or 'Rayleigh'
        snr          : channel SNR in dB (None = no channel, clean latent passed through)
        P            : transmit power constraint (default 1.0)

    Example:
        model = DeepJSCC(c=16, channel_type='AWGN', snr=10)
        x_hat = model(x)                     # (B, 3, H, W) → (B, 3, H, W)
    """

    def __init__(self, c: int, channel_type: str = 'AWGN',
                 snr: float = None, P: float = 1.0):
        super(DeepJSCC, self).__init__()
        self.encoder = _Encoder(c=c, P=P)
        self.decoder = _Decoder(c=c)
        self.channel = Channel(channel_type, snr) if snr is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z     = self.encoder(x)
        z_rx  = self.channel(z) if self.channel is not None else z
        x_hat = self.decoder(z_rx)
        return x_hat

    def change_channel(self, channel_type: str = 'AWGN', snr: float = None):
        """Hot-swap the channel at test time (e.g. sweep SNR without reloading model)."""
        self.channel = Channel(channel_type, snr) if snr is not None else None

    def get_channel(self) -> dict:
        return self.channel.get_channel() if self.channel is not None else None

    def loss(self, prd: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss."""
        return nn.MSELoss(reduction='mean')(prd, gt)


if __name__ == '__main__':
    model = DeepJSCC(c=20, channel_type='AWGN', snr=10)
    print(model)
    x = torch.rand(1, 3, 128, 128)
    y = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")