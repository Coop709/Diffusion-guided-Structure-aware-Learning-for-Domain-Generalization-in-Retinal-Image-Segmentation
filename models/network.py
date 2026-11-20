import copy

import torch
from torch import nn
from torch.nn import functional as F

from configs.model import ModelConfig
from models.conditioning import StructuralCueConditioner
from models.decoder import FeaturePyramid, UViTDecoder, sinusoidal_time_embedding
from models.frequency import PreFusionFilter
from models.encoders import ImageEncoder
from models.sam_adapter import build_image_encoder, prepare_sam_input


class SourceSegmenter(nn.Module):
    """Step 1: source-supervised image encoder and task-specific decoder."""

    def __init__(self, config: ModelConfig, sam_checkpoint: str = None, use_sam: bool = True):
        super().__init__()
        self.config = config
        self.use_sam = use_sam
        self.encoder = (build_image_encoder(config.sam_type, sam_checkpoint) if use_sam
                        else ImageEncoder(config.image_size))
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, config.mask_channels, 1),
        )

    def forward(self, image: torch.Tensor):
        encoder_input = (prepare_sam_input(image, self.encoder.img_size) if self.use_sam else
                         F.interpolate(image, size=(self.encoder.img_size,) * 2,
                                       mode="bilinear", align_corners=False))
        encoded = self.encoder(encoder_input)
        logits = self.decoder(encoded)
        return F.interpolate(logits, size=(self.config.image_size,) * 2, mode="bilinear", align_corners=False)


class ConditionalDiffusionSegmenter(nn.Module):
    """Step 2/3: frozen mask encoder, separate image encoder, level fusion, decoder.

    During Stage 2 training ``segmentation_encoder`` is the encoder transferred
    from Stage 1.  Fig. 3 uses the original pretrained encoder for this branch
    at inference time; :meth:`from_original` constructs that inference variant.
    """

    def __init__(self, config: ModelConfig, segmentation_encoder: nn.Module, condition_encoder: nn.Module,
                 cue_conditioning: bool = False, use_sam: bool = True):
        super().__init__()
        self.config = config
        self.use_sam = use_sam
        self.segmentation_encoder = segmentation_encoder
        self.condition_encoder = condition_encoder
        for parameter in self.segmentation_encoder.parameters():
            parameter.requires_grad_(False)
        self.segmentation_pyramid = FeaturePyramid(config.feature_width)
        self.condition_pyramid = FeaturePyramid(config.feature_width)
        self.time_projection = nn.Sequential(nn.Linear(config.feature_width, config.feature_width), nn.SiLU(),
                                             nn.Linear(config.feature_width, config.feature_width))
        self.filters = nn.ModuleList([PreFusionFilter(config.feature_width) for _ in range(3)])
        self.decoder = UViTDecoder(config.feature_width, config.mask_channels,
                                   depth=config.uvit_depth, heads=config.uvit_heads)
        self.cue_conditioner = (StructuralCueConditioner(config.mask_channels, config.feature_width)
                                if cue_conditioning else None)

    @classmethod
    def from_stage1(cls, config: ModelConfig, stage1: SourceSegmenter, sam_checkpoint: str,
                    cue_conditioning: bool = False, use_sam: bool = True):
        """Build Stage 2 from the transferred Stage-1 encoder."""
        condition_encoder = (build_image_encoder(config.sam_type, sam_checkpoint) if use_sam
                             else ImageEncoder(config.image_size))
        return cls(config, copy.deepcopy(stage1.encoder), condition_encoder,
                   cue_conditioning=cue_conditioning, use_sam=use_sam)

    @classmethod
    def without_sam(cls, config: ModelConfig, cue_conditioning: bool = False):
        """No source SAM fine-tuning or encoder transfer; train both branches."""
        return cls(config, ImageEncoder(config.image_size), ImageEncoder(config.image_size),
                   cue_conditioning=cue_conditioning, use_sam=False)

    @classmethod
    def from_original(cls, config: ModelConfig, sam_checkpoint: str, cue_conditioning: bool = False):
        """Build the Fig. 3 inference model with the original SAM encoder.

        The condition encoder is initialized from the same SAM checkpoint and
        receives its trained Stage 2 weights.  The segmentation encoder remains
        at the original pretrained parameters and is therefore intentionally
        excluded when the Stage 2 checkpoint is restored by ``infer.py``.
        """
        return cls(
            config,
            build_image_encoder(config.sam_type, sam_checkpoint),
            build_image_encoder(config.sam_type, sam_checkpoint),
            cue_conditioning=cue_conditioning,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.segmentation_encoder.eval()
        return self

    def encode_condition(self, image: torch.Tensor, structural_cues=None):
        condition_input = (prepare_sam_input(image, self.condition_encoder.img_size) if self.use_sam else
                           F.interpolate(image, size=(self.condition_encoder.img_size,) * 2,
                                         mode="bilinear", align_corners=False))
        embedding = self.condition_encoder(condition_input)
        levels = self.condition_pyramid(embedding)
        if self.cue_conditioner is not None:
            if structural_cues is None:
                raise ValueError("This checkpoint requires coarse-mask and channel-guidance conditions")
            levels = self.cue_conditioner(*structural_cues, levels)
        elif structural_cues is not None:
            raise ValueError("Extra cues require a separately trained cue-conditioning checkpoint")
        return levels

    def forward(self, noisy_mask: torch.Tensor, t: torch.Tensor, condition_features, use_pff: bool = True,
                return_features: bool = False):
        if noisy_mask.shape[1] == 1:
            mask_rgb = noisy_mask.repeat(1, 3, 1, 1)
        else:
            mask_rgb = F.pad(noisy_mask, (0, 0, 0, 0, 0, 3 - noisy_mask.shape[1]))
        mask_rgb = (mask_rgb.tanh() + 1) / 2
        mask_input = (prepare_sam_input(mask_rgb, self.segmentation_encoder.img_size) if self.use_sam else
                      F.interpolate(mask_rgb, size=(self.segmentation_encoder.img_size,) * 2,
                                    mode="bilinear", align_corners=False))
        with torch.no_grad():
            embedding = self.segmentation_encoder(mask_input)
        mask_levels = self.segmentation_pyramid(embedding)
        time_embedding = self.time_projection(sinusoidal_time_embedding(t, self.config.feature_width))
        time_fraction = t.float() / (self.config.diffusion_steps - 1)
        fused = []
        for mask_level, condition_level, filter_layer in zip(mask_levels, condition_features, self.filters):
            condition = filter_layer(condition_level, time_fraction) if use_pff else condition_level
            fused.append(mask_level + condition)
        return self.decoder(fused, time_embedding, return_features=return_features)
