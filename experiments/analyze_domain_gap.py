"""Class-conditional source/held-out-target MMD squared (Table 10)."""

import argparse
import json
from pathlib import Path

import numpy as np

from utils.evaluation_io import prediction_metadata, write_csv, write_json
from evaluation.feature_bank import collect_feature_bank
from evaluation.feature_metrics import rbf_mmd_squared
from datasets.manifest import read_manifest


def analyze(config_path, sigma=1.0, samples=1000, sampling_seed=42, size=256):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = config["target_domain"]
    domains = config["domains"]
    if len(domains) != 4 or target not in domains:
        raise ValueError("A vessel DG fold requires exactly three source domains and one target")
    banks, bank_info, signature = {}, {}, None
    for domain, entry in domains.items():
        manifest = (config_path.parent / entry["manifest"]).resolve()
        predictions = (config_path.parent / entry["predictions"]).resolve()
        records = read_manifest(str(manifest), "vessel", require_labels=True)
        if {record.get("domain") for record in records} != {domain}:
            raise ValueError(f"Manifest does not contain only {domain}: {manifest}")
        metadata = prediction_metadata(predictions)
        if metadata.get("task", "vessel") != "vessel" or metadata.get("image_size", size) != size:
            raise ValueError("MMD masks must use the task and evaluation resolution of the feature export")
        keys = ("stage2_checkpoint", "seed", "initialization", "steps", "eta", "no_pff")
        if not all(key in metadata for key in keys):
            raise ValueError(f"Missing inference metadata: {predictions}/run.json")
        current = tuple(metadata[key] for key in keys) + (metadata.get("fda"),)
        if signature is None:
            signature = current
        if current != signature:
            raise ValueError("All four domains must use the same fold checkpoint and inference settings")
        if metadata["seed"] != config["seed"]:
            raise ValueError("Config seed differs from the exported feature run")
        banks[domain], bank_info[domain] = collect_feature_bank(
            manifest, predictions, samples, sampling_seed, size)
    pairs = []
    for source in domains:
        if source == target:
            continue
        for class_name in ("background", "vessel"):
            pairs.append({"source": source, "target": target, "class": class_name,
                          "mmd_squared": rbf_mmd_squared(
                              banks[source][class_name], banks[target][class_name], sigma)})
    value = float(np.mean([pair["mmd_squared"] for pair in pairs]))
    row = {"method": config["method"], "task": "vessel", "setting": "vessel_dg_mmd",
           "seed": config["seed"], "domain": target, "sample_id": f"fold:{target}",
           "channel": "class_average", "mmd_squared": value}
    return row, {"config": config, "sigma": sigma, "samples_per_class": samples,
                 "sampling_seed": sampling_seed, "feature_banks": bank_info, "pairs": pairs,
                 "estimator": "biased RBF MMD-squared on L2-normalized vectors; no square root"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    rows, details = [], []
    for path in args.config:
        row, detail = analyze(path, args.sigma, args.samples_per_class, args.sampling_seed, args.size)
        rows.append(row)
        details.append(detail)
        print(f"{row['method']} target={row['domain']} MMD^2={row['mmd_squared']:.6f}")
    write_csv(args.output, rows)
    write_json(str(Path(args.output).with_suffix(".metadata.json")), details)


if __name__ == "__main__":
    main()
