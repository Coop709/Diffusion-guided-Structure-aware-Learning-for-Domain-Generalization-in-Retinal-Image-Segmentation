import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF

from datasets.manifest import read_manifest


class RetinaDataset(Dataset):
    def __init__(self, manifest: str, task: str, training: bool, size: int = 256,
                 with_labels: Optional[bool] = None):
        self.manifest = Path(manifest).resolve()
        self.with_labels = training if with_labels is None else with_labels
        self.records = read_manifest(str(self.manifest), task, require_labels=self.with_labels)
        self.task = task
        self.training = training
        self.size = size

    def __len__(self):
        return len(self.records)

    def _path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.manifest.parent / path

    @staticmethod
    def _disc_crop(image, masks, center):
        width, height = image.size
        cx, cy = center if center is not None else (width / 2, height / 2)
        side = min(width, height)
        left = int(max(0, min(width - side, cx - side / 2)))
        top = int(max(0, min(height - side, cy - side / 2)))
        box = (left, top, left + side, top + side)
        return image.crop(box), [mask.crop(box) for mask in masks]

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(self._path(record["image"])).convert("RGB")
        if self.task == "vessel":
            keys = ["mask"]
        elif self.task == "odoc":
            keys = ["disc_mask", "cup_mask"]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        masks = [Image.open(self._path(record[key])).convert("L") for key in keys] if self.with_labels else []
        coarse_mask = (Image.open(self._path(record["coarse_mask"])).convert("L")
                       if "coarse_mask" in record else None)

        if self.task == "odoc":
            original_image = image
            image, masks = self._disc_crop(image, masks, record.get("disc_center"))
            if coarse_mask is not None:
                _, cropped_masks = self._disc_crop(original_image, [coarse_mask], record.get("disc_center"))
                coarse_mask = cropped_masks[0]

        if self.training:
            i, j, h, w = RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(0.9, 1.1))
            image = TF.resized_crop(image, i, j, h, w, (self.size, self.size), InterpolationMode.BILINEAR)
            masks = [TF.resized_crop(mask, i, j, h, w, (self.size, self.size), InterpolationMode.NEAREST) for mask in masks]
            if coarse_mask is not None:
                coarse_mask = TF.resized_crop(coarse_mask, i, j, h, w, (self.size, self.size),
                                              InterpolationMode.NEAREST)
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            masks = [TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST) for mask in masks]
            if coarse_mask is not None:
                coarse_mask = TF.rotate(coarse_mask, angle, interpolation=InterpolationMode.NEAREST)
            if random.random() < 0.5:
                image, masks = TF.hflip(image), [TF.hflip(mask) for mask in masks]
                if coarse_mask is not None:
                    coarse_mask = TF.hflip(coarse_mask)
            if random.random() < 0.5:
                image, masks = TF.vflip(image), [TF.vflip(mask) for mask in masks]
                if coarse_mask is not None:
                    coarse_mask = TF.vflip(coarse_mask)
        else:
            image = TF.resize(image, (self.size, self.size), InterpolationMode.BILINEAR)
            masks = [TF.resize(mask, (self.size, self.size), InterpolationMode.NEAREST) for mask in masks]
            if coarse_mask is not None:
                coarse_mask = TF.resize(coarse_mask, (self.size, self.size), InterpolationMode.NEAREST)

        result = {"image": TF.to_tensor(image), "name": f"{index:06d}_{Path(record['image']).stem}"}
        if masks:
            result["mask"] = torch.stack([torch.from_numpy((np.asarray(mask) > 127).astype(np.float32)) for mask in masks])
        if coarse_mask is not None:
            result["coarse_mask"] = torch.from_numpy((np.asarray(coarse_mask) > 127).astype(np.float32))[None]
        if "source_image" in record:
            source = Image.open(self._path(record["source_image"])).convert("RGB")
            if self.task == "odoc":
                source, _ = self._disc_crop(source, [], record.get("source_disc_center"))
            source = TF.resize(source, (self.size, self.size), InterpolationMode.BILINEAR)
            result["source_image"] = TF.to_tensor(source)
        return result
