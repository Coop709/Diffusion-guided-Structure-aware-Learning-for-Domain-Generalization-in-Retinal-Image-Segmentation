"""Read per-case records and enforce comparable runs before aggregation."""

import csv
import glob
from pathlib import Path

import numpy as np

from utils.evaluation_io import IDENTIFIERS


def load_records(patterns):
    paths = sorted({str(Path(path).resolve()) for pattern in patterns
                    for path in glob.glob(str(pattern))})
    if not paths:
        raise ValueError("No CSV files matched the input paths")
    rows, metrics, seen = [], None, set()
    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            if not set(IDENTIFIERS) <= set(columns):
                raise ValueError(f"Missing identifier columns in {path}")
            current = tuple(key for key in columns if key not in IDENTIFIERS)
            if metrics is None:
                metrics = current
            if set(current) != set(metrics):
                raise ValueError("Input CSVs must contain the same metric columns")
            for row in reader:
                if not row["seed"]:
                    raise ValueError(f"Missing run seed: {path}")
                row["seed"] = int(row["seed"])
                key = tuple(row[column] for column in IDENTIFIERS)
                if key in seen:
                    raise ValueError(f"Duplicate evaluation record: {key}")
                seen.add(key)
                for metric in metrics:
                    row[metric] = float(row[metric]) if row[metric] else float("nan")
                rows.append(row)
    if not rows or not metrics:
        raise ValueError("No numerical evaluation records")
    return rows, metrics


def validate_repeated_cases(rows, seeds):
    groups = {}
    for row in rows:
        key = (row["method"], row["task"], row["setting"])
        groups.setdefault(key, {}).setdefault(row["seed"], set()).add(
            (row["domain"], row["sample_id"], row["channel"]))
    for key, runs in groups.items():
        if set(runs) != set(seeds):
            raise ValueError(f"{key}: expected seeds {seeds}; received {sorted(runs)}")
        reference = runs[seeds[0]]
        if any(cases != reference for cases in runs.values()):
            raise ValueError(f"{key}: case/domain/channel membership differs across seeds")


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")
