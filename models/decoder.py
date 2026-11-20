"""Multi-scale feature projections and U-shaped transformer decoder."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.uvit import UViTBackbone


class FeaturePyramid(nn.Module):
    """Project the 256-channel encoder embedding to 64, 32 and 16 pixel levels."""

    def __init__(self, width: int):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(256, width, 1) for _ in range(3)])

    def forward(self, embedding: torch.Tensor):
        return [projection(F.interpolate(embedding, size=(size, size), mode="bilinear", align_corners=False))
                for projection, size in zip(self.projections, (64, 32, 16))]


class UViTDecoder(nn.Module):
    """U-ViT token backbone followed by the multi-scale prediction head."""

    def __init__(self, width: int, out_channels: int, depth: int = 9, heads: int = 4):
        super().__init__()
        self.backbone = UViTBackbone(width, depth=depth, heads=heads)
        self.up32 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU())
        self.up64 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU())
        self.output = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(width, out_channels, 1))

    def forward(self, levels, time_embedding, return_features: bool = False):
        level64, level32, level16 = levels
        x = self.backbone(levels, time_embedding)
        x = self.up32(F.interpolate(x, size=level32.shape[-2:], mode="bilinear", align_corners=False) + level32)
        x = self.up64(F.interpolate(x, size=level64.shape[-2:], mode="bilinear", align_corners=False) + level64)
        features = self.output[1](self.output[0](x))
        prediction = F.interpolate(self.output[2](features), scale_factor=4,
                                   mode="bilinear", align_corners=False)
        return (prediction, features) if return_features else prediction


def sinusoidal_time_embedding(t: torch.Tensor, width: int):
    half = width // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    angles = t.float()[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=1)
