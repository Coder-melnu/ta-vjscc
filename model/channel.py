import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, channel_type='AWGN', snr=20):
        if channel_type not in ['AWGN', 'Rayleigh']:
            raise Exception('Unknown type of channel')
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr

    def forward(self, z_hat):
        if z_hat.dim() not in {3, 4}:
            raise ValueError('Input tensor must be 3D or 4D')

        if z_hat.dim() == 3:
            z_hat = z_hat.unsqueeze(0)

        # Per-sample signal power and noise
        k = z_hat[0].numel()
        sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
        noi_pwr = sig_pwr / (10 ** (self.snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr / 2)

        if self.channel_type == 'Rayleigh':
            # Multiply by a complex Rayleigh coefficient h ~ CN(0, I)
            # before adding noise. First half of channels = real part,
            # second half = imaginary part of the transmitted signal.
            h = torch.randn(2, device=z_hat.device)
            z_hat = z_hat.clone()
            half = z_hat.size(1) // 2
            z_hat[:, :half] = h[0] * z_hat[:, :half]
            z_hat[:, half:] = h[1] * z_hat[:, half:]

        return z_hat + noise

    def get_channel(self):
        return self.channel_type, self.snr


if __name__ == '__main__':
    # test AWGN
    channel = Channel(channel_type='AWGN', snr=10)
    z_hat = torch.randn(64, 10, 5, 5)
    y = channel(z_hat)
    print(f"AWGN:     in {z_hat.shape} -> out {y.shape}")

    # test Rayleigh
    channel = Channel(channel_type='Rayleigh', snr=10)
    z_hat = torch.randn(64, 10, 5, 5)
    y = channel(z_hat)
    print(f"Rayleigh: in {z_hat.shape} -> out {y.shape}")