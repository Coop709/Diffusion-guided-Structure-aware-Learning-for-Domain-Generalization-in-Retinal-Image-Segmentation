"""Patch-token diffusion backbone with a time token and long skip connections."""

import torch
from torch import nn


class UViTBlock(nn.Module):
    def __init__(self, width, heads, skip=False):
        super().__init__()
        self.skip_projection = nn.Linear(2 * width, width) if skip else None
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))

    def forward(self, tokens, skip=None):
        if self.skip_projection is not None:
            if skip is None:
                raise ValueError("A decoder block requires its paired encoder tokens")
            tokens = self.skip_projection(torch.cat((tokens, skip), dim=-1))
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class UViTBackbone(nn.Module):
    """Encode fused feature patches and decode through symmetric token skips."""

    def __init__(self, width, depth=9, heads=4, token_grid=16):
        super().__init__()
        if depth < 3 or depth % 2 != 1 or width % heads:
            raise ValueError("Use an odd depth >= 3 and a width divisible by the head count")
        self.token_grid = token_grid
        # The three fused feature levels are 64, 32 and 16 pixels per side.
        self.patch_embeddings = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=patch, stride=patch)
            for patch in (4, 2, 1)
        ])
        self.time_token = nn.Linear(width, width)
        self.position_embedding = nn.Parameter(torch.zeros(1, 1 + token_grid ** 2, width))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        half = depth // 2
        self.encoder_blocks = nn.ModuleList([UViTBlock(width, heads) for _ in range(half)])
        self.middle_block = UViTBlock(width, heads)
        self.decoder_blocks = nn.ModuleList([UViTBlock(width, heads, skip=True) for _ in range(half)])
        self.norm = nn.LayerNorm(width)

    def forward(self, levels, time_embedding):
        if len(levels) != len(self.patch_embeddings):
            raise ValueError("U-ViT expects three fused feature levels")
        patches = [projection(level) for projection, level in zip(self.patch_embeddings, levels)]
        if any(patch.shape[-2:] != (self.token_grid, self.token_grid) for patch in patches):
            raise ValueError("Feature patches do not match the configured token grid")
        tokens = sum(patch.flatten(2).transpose(1, 2) for patch in patches)
        time = self.time_token(time_embedding).unsqueeze(1)
        tokens = torch.cat((time, tokens), dim=1) + self.position_embedding
        skips = []
        for block in self.encoder_blocks:
            tokens = block(tokens)
            skips.append(tokens)
        tokens = self.middle_block(tokens)
        for block in self.decoder_blocks:
            tokens = block(tokens, skips.pop())
        tokens = self.norm(tokens)[:, 1:]
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[-1], self.token_grid, self.token_grid)
