import argparse
from pathlib import Path

from utils.evaluation_io import (PERCENT_METRICS, domain_summary, prediction_metadata,
                           prediction_samples, write_csv, write_json)
from evaluation.metrics import binary_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate the metrics defined in the manuscript")
    parser.add_argument("--task", choices=("vessel", "odoc"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", help="Per-image CSV (scores in percent)")
    parser.add_argument("--method", default="Ours")
    parser.add_argument("--setting", default="unspecified")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    metadata = prediction_metadata(args.predictions)
    if metadata.get("task", args.task) != args.task or metadata.get("image_size", args.size) != args.size:
        raise ValueError("Evaluation task/resolution differs from the prediction run")
    if args.seed is not None and metadata.get("seed", args.seed) != args.seed:
        raise ValueError("Evaluation seed differs from prediction metadata")
    seed = args.seed if args.seed is not None else metadata.get("seed", "")
    labels = ("vessel",) if args.task == "vessel" else ("disc", "cup")
    rows = []
    for sample in prediction_samples(args.manifest, args.task, args.predictions, args.size):
        for label, pred, gt in zip(labels, sample["probabilities"], sample["truth"]):
            roi = sample["roi"]
            values = binary_metrics(pred[roi], gt[roi])
            rows.append(dict(method=args.method, task=args.task, setting=args.setting, seed=seed,
                             domain=sample["domain"], sample_id=sample["sample_id"], channel=label,
                             **{key: float(value) * (100 if key in PERCENT_METRICS else 1)
                                for key, value in values.items()}))
    metrics = ("dice", "accuracy", "sensitivity", "specificity", "auc")
    summary = domain_summary(rows, metrics)
    for key, value in summary["Average"].items():
        print(f"{key}: {value:.4f}%")
    if args.output:
        write_csv(args.output, rows)
        write_json(str(Path(args.output).with_suffix(".metadata.json")),
                   {"arguments": vars(args), "prediction_run": metadata,
                    "units": "percent", "summary": summary})


if __name__ == "__main__":
    main()
