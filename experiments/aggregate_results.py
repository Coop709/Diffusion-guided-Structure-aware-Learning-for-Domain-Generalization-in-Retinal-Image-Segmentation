"""Compute within-run domain averages first, then mean/sample SD over seeds."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.evaluation_io import write_csv, write_json
from utils.result_records import finite_mean, load_records, validate_repeated_cases


def aggregate(rows, metrics, seeds=(0, 42, 3407)):
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be a nonempty list of distinct integers")
    validate_repeated_cases(rows, seeds)
    cases = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in ("method", "task", "setting", "seed", "domain", "channel"))
        cases[key].append(row)
    domain_scores = {key: {metric: finite_mean([case[metric] for case in values]) for metric in metrics}
                     for key, values in cases.items()}
    results = defaultdict(dict)
    for (method, task, setting, seed, domain, channel), scores in domain_scores.items():
        for metric, score in scores.items():
            results[(method, task, setting, domain, channel, metric)][seed] = score
    for method, task, setting in sorted({key[:3] for key in domain_scores}):
        for seed in seeds:
            selected = {key: value for key, value in domain_scores.items()
                        if key[:4] == (method, task, setting, seed)}
            channels = sorted({key[-1] for key in selected})
            domains = sorted({key[-2] for key in selected})
            if set((key[-2], key[-1]) for key in selected) != {(d, c) for d in domains for c in channels}:
                raise ValueError("Every domain must contain the same segmentation channels")
            for metric in metrics:
                for channel in channels:
                    scores = [value[metric] for key, value in selected.items() if key[-1] == channel]
                    results[(method, task, setting, "Average", channel, metric)][seed] = float(np.mean(scores))
                for domain in domains:
                    scores = [value[metric] for key, value in selected.items() if key[-2] == domain]
                    results[(method, task, setting, domain, "overall", metric)][seed] = float(np.mean(scores))
                scores = [value[metric] for value in selected.values()]
                results[(method, task, setting, "Average", "overall", metric)][seed] = float(np.mean(scores))
    summaries = []
    for key, values in sorted(results.items()):
        scores = np.asarray([values[seed] for seed in seeds], dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError(f"Undefined metric in a run for {key}; inspect per-case records")
        summaries.append(dict(zip(("method", "task", "setting", "domain", "channel", "metric"), key),
                              n_seeds=len(seeds), mean=float(scores.mean()),
                              std=float(scores.std(ddof=1)) if len(seeds) > 1 else "",
                              **{f"seed_{seed}": float(values[seed]) for seed in seeds}))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="CSV paths or quoted glob patterns")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 3407])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows, metrics = load_records(args.inputs)
    summaries = aggregate(rows, metrics, args.seeds)
    write_csv(args.output, summaries)
    write_json(str(Path(args.output).with_suffix(".metadata.json")),
               {"arguments": vars(args), "sd_ddof": 1,
                "averaging": "cases within domain/channel, then equal channels/domains within each seed"})
    for row in summaries:
        if row["domain"] == "Average" and row["channel"] == "overall":
            suffix = f" +/- {row['std']:.4f}" if row["std"] != "" else ""
            print(f"{row['method']} {row['setting']} {row['metric']}: {row['mean']:.4f}{suffix}")


if __name__ == "__main__":
    main()
