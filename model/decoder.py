# -*- coding: utf-8 -*-
"""
Decoder module for Deep-JSCC.
Contains: _TransConvWithPReLU, _Decoder
"""

import torch
import torch.nn as nn


class _TransConvWithPReLU(nn.Module):
    """ConvTranspose2d + activation with appropriate initialization."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 activate: nn.Module = None):
        super(_TransConvWithPReLU, self).__init__()
        self.activate = activate if activate is not None else nn.PReLU()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride, padding, output_padding
        )
        # Kaiming for PReLU, Xavier for others (e.g. Sigmoid on final layer)
        if isinstance(self.activate, nn.PReLU):
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out',
                                    nonlinearity='leaky_relu')
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.transconv(x))


class _Decoder(nn.Module):
    """
    Deep-JSCC CNN Decoder.
    Maps received latent (B, 2c, H/4, W/4) → reconstructed image (B, 3, H, W).

    Args:
        c : number of complex channel symbols (must match encoder)
    """

    def __init__(self, c: int = 1):
        super(_Decoder, self).__init__()
        self.tconv1 = _TransConvWithPReLU(in_channels=2*c, out_channels=32,
                                           kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(in_channels=32,  out_channels=32,
                                           kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(in_channels=32,  out_channels=32,
                                           kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(in_channels=32,  out_channels=16,
                                           kernel_size=5, stride=2, padding=2,
                                           output_padding=1)
        self.tconv5 = _TransConvWithPReLU(in_channels=16,  out_channels=3,
                                           kernel_size=5, stride=2, padding=2,
                                           output_padding=1,
                                           activate=nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x