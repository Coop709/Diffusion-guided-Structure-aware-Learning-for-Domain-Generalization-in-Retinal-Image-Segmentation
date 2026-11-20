"""Read and validate source/target JSONL manifests without importing PyTorch."""

import json
from pathlib import Path


def read_manifest(filename: str, task: str, require_labels: bool) -> list[dict]:
    manifest = Path(filename).resolve()
    if task not in ("vessel", "odoc"):
        raise ValueError(f"Unknown task: {task}")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    required = ["image"]
    if require_labels:
        required.extend(["mask"] if task == "vessel" else ["disc_mask", "cup_mask"])
    records = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{manifest}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"{manifest}:{line_number}: expected a JSON object")
        for key in required:
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"{manifest}:{line_number}: missing {key!r} path")
        for key in ("image", "mask", "disc_mask", "cup_mask", "source_image", "fov_mask", "coarse_mask"):
            if key in record:
                if not isinstance(record[key], str) or not record[key]:
                    raise ValueError(f"{manifest}:{line_number}: invalid {key!r} path")
                path = Path(record[key])
                resolved = path if path.is_absolute() else manifest.parent / path
                if not resolved.is_file():
                    raise FileNotFoundError(f"{manifest}:{line_number}: {key} not found: {resolved}")
        for key in ("disc_center", "source_disc_center"):
            if key in record and (not isinstance(record[key], list) or len(record[key]) != 2 or
                                  not all(isinstance(value, (int, float)) for value in record[key])):
                raise ValueError(f"{manifest}:{line_number}: {key} must be [x, y]")
        records.append(record)
    if not records:
        raise ValueError(f"{manifest}: no samples found")
    return records
