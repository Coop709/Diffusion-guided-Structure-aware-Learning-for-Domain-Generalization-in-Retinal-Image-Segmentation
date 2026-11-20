"""Bounded, class-balanced sampling of exported decoder feature maps."""

from pathlib import Path

import numpy as np
from PIL import Image

from utils.evaluation_io import labeled_samples


def collect_feature_bank(manifest, prediction_dir, samples_per_class=1000, seed=42, size=256):
    if samples_per_class < 1:
        raise ValueError("Feature sample limit must be positive")
    rng = np.random.default_rng(seed)
    banks, priorities = {}, {}
    counts = {"background": 0, "vessel": 0}
    for sample in labeled_samples(manifest, "vessel", size):
        path = Path(prediction_dir) / "features" / f"{sample['name']}.npz"
        with np.load(path, allow_pickle=False) as data:
            features = data["decoder_features"]
            if int(data["timestep"]) != 0:
                raise ValueError(f"Expected features from the final denoising invocation: {path}")
        if features.ndim != 3 or not np.isfinite(features).all():
            raise ValueError(f"Expected finite [C,H,W] features: {path}")
        height, width = features.shape[-2:]

        def resize(mask):
            return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
                (width, height), Image.Resampling.NEAREST), dtype=bool)

        truth, roi = resize(sample["truth"][0]), resize(sample["roi"])
        vectors = features.transpose(1, 2, 0)
        for name, mask in (("background", ~truth & roi), ("vessel", truth & roi)):
            candidates = vectors[mask]
            counts[name] += len(candidates)
            keys = rng.random(len(candidates))
            if name in banks:
                candidates = np.concatenate((banks[name], candidates))
                keys = np.concatenate((priorities[name], keys))
            if len(keys) > samples_per_class:
                keep = np.argpartition(keys, -samples_per_class)[-samples_per_class:]
                candidates, keys = candidates[keep], keys[keep]
            banks[name], priorities[name] = candidates, keys
    for name in counts:
        if not counts[name]:
            raise ValueError(f"No {name} features available in {manifest}")
    return banks, {"available": counts, "sampled": {name: len(value) for name, value in banks.items()}}
