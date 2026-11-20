"""Per-image structure evaluation and equal-domain summaries for Table 11."""

import argparse
from pathlib import Path

from utils.evaluation_io import (PERCENT_METRICS, domain_summary, prediction_metadata,
                           prediction_samples, write_csv, write_json)
from evaluation.structure_metrics import vessel_structure_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="Ours")
    parser.add_argument("--setting", default="vessel_dg")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thin-diameter", type=float, default=3.0)
    parser.add_argument("--bifurcation-tolerance", type=float, default=2.0)
    args = parser.parse_args()
    metadata = prediction_metadata(args.predictions)
    if metadata.get("task", "vessel") != "vessel" or metadata.get("image_size", args.size) != args.size:
        raise ValueError("Structure evaluation requires vessel predictions at the same resolution")
    if args.seed is not None and metadata.get("seed", args.seed) != args.seed:
        raise ValueError("Evaluation seed differs from prediction metadata")
    seed = args.seed if args.seed is not None else metadata.get("seed", "")
    rows = []
    for sample in prediction_samples(args.manifest, "vessel", args.predictions, args.size):
        values = vessel_structure_metrics(sample["probabilities"][0], sample["truth"][0],
                                          args.threshold, args.thin_diameter,
                                          args.bifurcation_tolerance, sample["roi"])
        rows.append(dict(method=args.method, task="vessel", setting=args.setting, seed=seed,
                         domain=sample["domain"], sample_id=sample["sample_id"], channel="vessel",
                         **{key: value * (100 if key in PERCENT_METRICS else 1)
                            for key, value in values.items()}))
    summary = domain_summary(rows, ("cldice", "thin_recall", "bifurcation_f1", "hd95"))
    write_csv(args.output, rows)
    write_json(str(Path(args.output).with_suffix(".metadata.json")),
               {"arguments": vars(args), "prediction_run": metadata, "summary": summary,
                "units": {"cldice": "percent", "thin_recall": "percent",
                          "bifurcation_f1": "percent", "hd95": "pixels"},
                "empty_hd95": "image diagonal if exactly one mask is empty; zero if both empty",
                "thin_reference": "GT skeleton pixels with EDT diameter <= thin-diameter"})
    for metric, value in summary["Average"].items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
