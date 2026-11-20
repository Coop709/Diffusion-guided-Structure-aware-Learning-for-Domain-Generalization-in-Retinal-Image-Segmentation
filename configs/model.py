from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    task: str = "vessel"
    image_size: int = 256
    sam_type: str = "vit_b"
    feature_width: int = 128
    uvit_depth: int = 9
    uvit_heads: int = 4
    diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    @property
    def mask_channels(self) -> int:
        if self.task == "vessel":
            return 1
        if self.task == "odoc":
            return 2
        raise ValueError(f"Unknown segmentation task: {self.task}")
