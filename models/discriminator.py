"""Patch discriminator for adversarial spectral-transfer pretraining."""

from torch import nn


class ImageDiscriminator(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in (width, width * 2, width * 4):
            layers.extend([nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                           nn.LeakyReLU(0.2, inplace=True)])
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, 1, 3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, image):
        return self.layers(image)
