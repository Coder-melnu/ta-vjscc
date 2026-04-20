# -*- coding: utf-8 -*-
"""
Encoder module for Deep-JSCC.
Contains: ratio2filtersize, _ConvWithPReLU, _Encoder
"""

import torch
import torch.nn as nn


def ratio2filtersize(x: torch.Tensor, ratio: float) -> int:
    """
    Compute the number of filters (c) needed to achieve a given bandwidth ratio.

    Args:
        x     : input image tensor (B, C, H, W)
        ratio : target bandwidth compression ratio

    Returns:
        int : number of filters c for the encoder bottleneck
    """
    if x.dim() == 4:
        before_size = torch.prod(torch.tensor(x.size()[1:]))
    elif x.dim() == 3:
        before_size = torch.prod(torch.tensor(x.size()))
    else:
        raise Exception('Unknown size of input')

    encoder_temp = _Encoder(is_temp=True)
    z_temp = encoder_temp(x)
    c = before_size * ratio / torch.prod(torch.tensor(z_temp.size()[-2:]))
    return int(c)


class _ConvWithPReLU(nn.Module):
    """Conv2d + PReLU activation with Kaiming initialization."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        super(_ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.conv(x))


class _Encoder(nn.Module):
    """
    Deep-JSCC CNN Encoder.
    Maps input image (B, 3, H, W) → normalized latent (B, 2c, H/4, W/4).

    Args:
        c       : number of complex channel symbols (bottleneck depth = 2c)
        is_temp : if True, skip final conv + normalization (used by ratio2filtersize)
        P       : transmit power constraint
    """

    def __init__(self, c: int = 1, is_temp: bool = False, P: float = 1.0):
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        self.conv1 = _ConvWithPReLU(in_channels=3,   out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(in_channels=16,  out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = _ConvWithPReLU(in_channels=32,  out_channels=32, kernel_size=5, padding=2)
        self.conv4 = _ConvWithPReLU(in_channels=32,  out_channels=32, kernel_size=5, padding=2)
        self.conv5 = _ConvWithPReLU(in_channels=32,  out_channels=2*c, kernel_size=5, padding=2)
        self.norm  = self._normalization_layer(P=P)

    @staticmethod
    def _normalization_layer(P: float = 1.0):
        """
        Power normalization: scales z so that E[||z||^2] = P * k,
        where k is the total number of channel symbols.
        """
        def _inner(z_hat: torch.Tensor) -> torch.Tensor:
            if z_hat.dim() == 4:
                batch_size = z_hat.size(0)
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
            elif z_hat.dim() == 3:
                batch_size = 1
                k = torch.prod(torch.tensor(z_hat.size()))
            else:
                raise Exception('Unknown size of input')

            z_temp  = z_hat.reshape(batch_size, 1, 1, -1)
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)
            tensor  = torch.sqrt(P * k) * z_hat / torch.sqrt(z_temp @ z_trans)

            return tensor.squeeze(0) if batch_size == 1 else tensor
        return _inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if not self.is_temp:
            x = self.conv5(x)
            x = self.norm(x)
        return x