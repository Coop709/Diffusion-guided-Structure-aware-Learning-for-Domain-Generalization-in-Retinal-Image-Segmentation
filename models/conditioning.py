"""Trainable alternative: supply both structural cues as extra conditions."""

import torch
from torch import nn
from torch.nn import functional as F


class StructuralCueConditioner(nn.Module):
    """Project coarse-mask/channel cues into the three image-condition levels."""

    def __init__(self, mask_channels, width):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * mask_channels, width, 3, padding=1),
                          nn.GELU(), nn.Conv2d(width, width, 1))
            for _ in range(3)
        ])

    def forward(self, coarse_mask, channel_prior, condition_levels):
        cues = torch.cat((coarse_mask, channel_prior), dim=1)
        return [level + projection(F.interpolate(cues, size=level.shape[-2:],
                                                 mode="bilinear", align_corners=False))
                for projection, level in zip(self.projections, condition_levels)]
