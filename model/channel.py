import math

import torch
import torch.nn as nn


class Channel(nn.Module):
    """AWGN or independent per-sample slow complex Rayleigh fading channel."""

    def __init__(self, channel_type="AWGN", snr=20):
        super().__init__()
        if channel_type not in {"AWGN", "Rayleigh"}:
            raise ValueError(f"Unknown channel type: {channel_type}")

        self.channel_type = channel_type
        self.snr = float(snr)

    def forward(self, z_hat):
        if z_hat.dim() not in {3, 4}:
            raise ValueError("Input tensor must be 3D or 4D")

        unbatched = z_hat.dim() == 3
        if unbatched:
            z_hat = z_hat.unsqueeze(0)

        # Transmitter-referenced, per-sample signal and noise power.
        # This preserves the project's existing AWGN SNR convention.
        k = z_hat[0].numel()
        signal_power = (
            z_hat.abs().square().sum(dim=(1, 2, 3), keepdim=True) / k
        )
        snr_linear = 10.0 ** (self.snr / 10.0)
        noise_power = signal_power / snr_linear

        channel_output = z_hat

        if self.channel_type == "Rayleigh":
            channels = z_hat.size(1)
            if channels % 2 != 0:
                raise ValueError(
                    "Complex Rayleigh fading requires an even number "
                    "of latent channels"
                )

            half = channels // 2
            x_real = z_hat[:, :half]
            x_imag = z_hat[:, half:]

            # Independent slow/block fading per encoded sample:
            # h ~ CN(0,1), hence E[|h|^2] = 1.
            fading_shape = (z_hat.size(0), 1, 1, 1)
            component_std = 1.0 / math.sqrt(2.0)

            h_real = torch.randn(
                fading_shape,
                device=z_hat.device,
                dtype=z_hat.dtype,
            ) * component_std
            h_imag = torch.randn(
                fading_shape,
                device=z_hat.device,
                dtype=z_hat.dtype,
            ) * component_std

            # (h_real + j*h_imag) * (x_real + j*x_imag)
            faded_real = h_real * x_real - h_imag * x_imag
            faded_imag = h_imag * x_real + h_real * x_imag
            channel_output = torch.cat(
                [faded_real, faded_imag], dim=1
            )

        # Complex-AWGN convention retained from the existing implementation.
        noise = torch.randn_like(z_hat) * torch.sqrt(noise_power / 2.0)
        received = channel_output + noise

        return received.squeeze(0) if unbatched else received

    def get_channel(self):
        return self.channel_type, self.snr


if __name__ == "__main__":
    torch.manual_seed(42)

    z = torch.randn(64, 10, 5, 5)

    awgn = Channel(channel_type="AWGN", snr=10)
    print("AWGN:", z.shape, "->", awgn(z).shape)

    rayleigh = Channel(channel_type="Rayleigh", snr=10)
    print("Rayleigh:", z.shape, "->", rayleigh(z).shape)
