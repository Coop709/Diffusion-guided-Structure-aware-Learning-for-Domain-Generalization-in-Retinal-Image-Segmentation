"""Structural cues and the five reverse-diffusion initialization strategies."""

import torch
from torch.nn import functional as F


STRATEGIES = ("gaussian", "sam_only", "channel_only", "additional_condition", "joint")


def channel_guidance(target_image, coarse_mask, generator=None):
    """Random RGB mixture, broadcast to the task's mask channels."""
    if target_image.ndim != 4 or target_image.shape[1] != 3:
        raise ValueError("Target images must have shape [B, 3, H, W]")
    image = F.interpolate(target_image, size=coarse_mask.shape[-2:],
                          mode="bilinear", align_corners=False)
    color = torch.rand(image.shape[0], 3, 1, 1, device=image.device,
                       dtype=image.dtype, generator=generator)
    prior = (image * color).sum(1, keepdim=True) / color.sum(1, keepdim=True).clamp_min(1e-6)
    return prior.expand_as(coarse_mask)


def convolve_structure(noise, cue):
    """Channel-wise linear convolution, center-cropped to the mask size.

    Zero padding avoids circular wraparound. The cue is held fixed for each
    draw, so this is a linear transformation of the Gaussian noise field.
    """
    if noise.ndim != 4 or noise.shape != cue.shape:
        raise ValueError("Noise and structural cue must have equal BCHW shapes")
    height, width = noise.shape[-2:]
    full_size = (2 * height - 1, 2 * width - 1)
    spectrum = (torch.fft.rfft2(noise.float(), s=full_size)
                * torch.fft.rfft2(cue.float(), s=full_size))
    full = torch.fft.irfft2(spectrum, s=full_size)
    top, left = (height - 1) // 2, (width - 1) // 2
    return full[..., top:top + height, left:left + width].to(noise.dtype)


def initialize_mask(coarse_mask, channel_prior, strategy="joint", generator=None):
    """Eq. (7): convolve the same iid N(0,1) field with the selected cues."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown initialization: {strategy}")
    if coarse_mask.shape != channel_prior.shape:
        raise ValueError("Coarse mask and channel prior must have equal shapes")
    noise = torch.randn(coarse_mask.shape, device=coarse_mask.device,
                        dtype=coarse_mask.dtype, generator=generator)
    if strategy in ("gaussian", "additional_condition"):
        return noise
    if strategy == "sam_only":
        return convolve_structure(noise, coarse_mask)
    if strategy == "channel_only":
        return convolve_structure(noise, channel_prior)
    return convolve_structure(noise, coarse_mask + channel_prior)
