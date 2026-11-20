"""Small adapter around the official Segment Anything image encoder."""

import torch
from torch.nn import functional as F
from segment_anything import sam_model_registry


SAM_MEAN = (123.675, 116.28, 103.53)
SAM_STD = (58.395, 57.12, 57.375)


def prepare_sam_input(image: torch.Tensor, image_size: int) -> torch.Tensor:
    """Convert a BCHW RGB image in [0, 1] to SAM's normalized input size."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("SAM input must have shape [B, 3, H, W]")
    image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = image.new_tensor(SAM_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(SAM_STD).view(1, 3, 1, 1)
    return (image * 255 - mean) / std


def build_image_encoder(sam_type: str, checkpoint: str):
    """Load pretrained weights; ownership of the returned encoder is independent."""
    if sam_type not in sam_model_registry:
        raise ValueError(f"Unknown SAM model type: {sam_type}")
    sam = sam_model_registry[sam_type](checkpoint=checkpoint)
    return sam.image_encoder
