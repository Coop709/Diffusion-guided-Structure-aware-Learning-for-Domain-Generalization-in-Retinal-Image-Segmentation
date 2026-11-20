"""Prediction/label pairing and machine-readable evaluation records (CPU only)."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.manifest import read_manifest


IDENTIFIERS = ("method", "task", "setting", "seed", "domain", "sample_id", "channel")
PERCENT_METRICS = {"dice", "auc", "accuracy", "sensitivity", "specificity",
                   "cldice", "thin_recall", "bifurcation_f1"}


def sample_name(index, record):
    return f"{index:06d}_{Path(record['image']).stem}"


def crop_box(size, center):
    width, height = size
    cx, cy = center if center is not None else (width / 2, height / 2)
    side = min(width, height)
    left = int(max(0, min(width - side, cx - side / 2)))
    top = int(max(0, min(height - side, cy - side / 2)))
    return left, top, left + side, top + side


def labeled_samples(manifest, task, size=256):
    """Apply the same deterministic evaluation crop/resize as RetinaDataset."""
    manifest = Path(manifest).resolve()
    records = read_manifest(str(manifest), task, require_labels=True)

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else manifest.parent / path

    for index, record in enumerate(records):
        with Image.open(resolve(record["image"])) as image:
            original_size = image.size
        box = crop_box(original_size, record.get("disc_center")) if task == "odoc" else None

        def load_mask(key):
            with Image.open(resolve(record[key])) as image:
                if image.size != original_size:
                    raise ValueError(f"Image/{key} dimensions differ: {record['image']}")
                image = image.convert("L")
                if box is not None:
                    image = image.crop(box)
                image = image.resize((size, size), Image.Resampling.NEAREST)
                return np.asarray(image) > 127

        keys = ("mask",) if task == "vessel" else ("disc_mask", "cup_mask")
        truth = np.stack([load_mask(key) for key in keys])
        roi = load_mask("fov_mask") if "fov_mask" in record else np.ones((size, size), dtype=bool)
        if not roi.any():
            raise ValueError(f"Empty field of view: {record['image']}")
        yield {"name": sample_name(index, record), "sample_id": record.get("id", record["image"]),
               "domain": record.get("domain", "unspecified"), "truth": truth, "roi": roi}


def prediction_samples(manifest, task, predictions, size=256):
    for sample in labeled_samples(manifest, task, size):
        path = Path(predictions) / f"{sample['name']}_probabilities.npy"
        probabilities = np.load(path, allow_pickle=False)
        if probabilities.shape != sample["truth"].shape:
            raise ValueError(f"Prediction/label shape mismatch: {path}")
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"Predictions must contain probabilities in [0,1]: {path}")
        yield dict(sample, probabilities=probabilities)


def prediction_metadata(directory):
    path = Path(directory) / "run.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("No evaluation records to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    """Represent undefined numerical quantities as JSON null."""
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def domain_summary(rows, metric_names):
    """Mean over cases in each domain/channel, then equal weight over groups."""
    result = {}
    groups = sorted({(row["domain"], row["channel"]) for row in rows})
    for metric in metric_names:
        scores = []
        for domain, channel in groups:
            values = np.asarray([row[metric] for row in rows
                                 if row["domain"] == domain and row["channel"] == channel], dtype=float)
            finite = values[np.isfinite(values)]
            mean = float(finite.mean()) if finite.size else float("nan")
            result.setdefault(domain, {}).setdefault(channel, {})[metric] = mean
            scores.append(mean)
        result.setdefault("Average", {})[metric] = float(np.mean(scores))
    return result
