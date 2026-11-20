import torch
from torch import nn


def fda_transfer(source: torch.Tensor, target: torch.Tensor, low_frequency_weight: torch.Tensor, beta: float = 0.1):
    """Keep source phase; blend target amplitude inside the central FFT window."""
    if source.shape != target.shape:
        raise ValueError("FDA source and target must have the same shape")
    if not 0 < beta <= 0.5:
        raise ValueError("FDA beta must be in (0, 0.5]")
    height, width = source.shape[-2:]
    # FFTs run in float32 so FDA also supports non-power-of-two sizes under AMP.
    src_fft = torch.fft.fftshift(torch.fft.fft2(source.float(), dim=(-2, -1)), dim=(-2, -1))
    tgt_fft = torch.fft.fftshift(torch.fft.fft2(target.float(), dim=(-2, -1)), dim=(-2, -1))
    radius_y = max(1, int(height * beta / 2))
    radius_x = max(1, int(width * beta / 2))
    mask = torch.zeros_like(source)
    mask[..., height // 2 - radius_y:height // 2 + radius_y,
         width // 2 - radius_x:width // 2 + radius_x] = low_frequency_weight
    amplitude = src_fft.abs() * (1 - mask) + tgt_fft.abs() * mask
    spectrum = amplitude * torch.exp(1j * torch.angle(src_fft))
    return torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1)).real.clamp(0, 1)


class SpectralTransfer(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        if not 0 < beta <= 0.5:
            raise ValueError("FDA beta must be in (0, 0.5]")
        self.beta = beta
        self.low_frequency_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        return fda_transfer(source, target, self.low_frequency_logit.sigmoid(), self.beta)


class PreFusionFilter(nn.Module):
    """Timestep-dependent low-frequency suppression before level-wise addition."""

    def __init__(self, channels: int):
        super().__init__()
        self.strength_logit = nn.Parameter(torch.tensor(0.0))
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, features: torch.Tensor, t_fraction: torch.Tensor):
        height, width = features.shape[-2:]
        fy = torch.fft.fftfreq(height, device=features.device).view(height, 1)
        fx = torch.fft.fftfreq(width, device=features.device).view(1, width)
        radius = torch.sqrt(fx.square() + fy.square())
        low_pass = torch.exp(-radius.square() / (2 * 0.1**2))
        strength = self.strength_logit.sigmoid() * (1 - t_fraction.view(-1, 1, 1, 1))
        spectrum = torch.fft.fft2(features.float(), dim=(-2, -1))
        filtered = torch.fft.ifft2(spectrum * (1 - strength * low_pass), dim=(-2, -1)).real
        return self.projection(filtered.to(features.dtype))
