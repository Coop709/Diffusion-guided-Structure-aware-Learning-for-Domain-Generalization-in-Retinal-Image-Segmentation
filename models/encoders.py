"""Trainable image encoders for the ablations without SAM initialization."""

from torch import nn


class ImageEncoder(nn.Module):
    """RGB-to-embedding encoder with the feature interface used by both branches."""

    def __init__(self, image_size=256):
        super().__init__()
        self.img_size = image_size
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.GELU(),
        )

    def forward(self, image):
        return self.layers(image)
