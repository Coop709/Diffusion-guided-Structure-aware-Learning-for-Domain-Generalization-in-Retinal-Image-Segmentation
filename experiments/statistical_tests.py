"""Paired analysis when both methods have predictions for the same test cases."""

import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from utils.evaluation_io import write_json
from utils.result_records import load_records, validate_repeated_cases


def paired_test(baseline, ours, metric, seeds, unit="case", bootstrap_samples=5000, seed=42):
    """Average seeds per case; optionally average cases within domains."""
    if unit not in ("case", "domain") or bootstrap_samples < 1:
        raise ValueError("Invalid pairing unit or bootstrap count")

    def paired_values(records):
        validate_repeated_cases(records, tuple(seeds))
        if len({(row["method"], row["task"], row["setting"]) for row in records}) != 1:
            raise ValueError("Analyze one method/task/setting per input")
        values = defaultdict(list)
        for row in records:
            if not np.isfinite(row[metric]):
                raise ValueError(f"Undefined {metric}; specify a comparable valid cohort")
            values[(row["task"], row["setting"], row["domain"], row["sample_id"], row["channel"])].append(row[metric])
        result = {key: float(np.mean(value)) for key, value in values.items()}
        if unit == "domain":
            domains = defaultdict(list)
            for key, value in result.items():
                domains[(key[0], key[1], key[2], key[4])].append(value)
            result = {key: float(np.mean(value)) for key, value in domains.items()}
        return result

    left, right = paired_values(baseline), paired_values(ours)
    if set(left) != set(right):
        raise ValueError("Methods must have identical task/setting/domain/case/channel identifiers")
    if len({key[-1] for key in left}) != 1:
        raise ValueError("Select one channel before pairing; disc/cup are not independent cases")
    differences = np.asarray([right[key] - left[key] for key in sorted(left)])
    if len(differences) < 2:
        raise ValueError("At least two paired evaluation units are required")
    statistic, pvalue = (0.0, 1.0) if np.all(differences == 0) else wilcoxon(
        differences, alternative="two-sided", zero_method="wilcox")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples)
    for index in range(bootstrap_samples):
        bootstrap[index] = rng.choice(differences, len(differences), replace=True).mean()
    return {"metric": metric, "unit": unit, "n_pairs": len(differences),
            "mean_difference_ours_minus_baseline": float(differences.mean()),
            "bootstrap_95_ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "wilcoxon_statistic": float(statistic), "p_value_two_sided": float(pvalue)}


def holm_adjust(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, pvalues[index] * (len(pvalues) - rank)))
        adjusted[index] = previous
    return adjusted.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--ours", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+", default=["dice"])
    parser.add_argument("--channel", default="vessel")
    parser.add_argument("--domain", help="Restrict analysis to one target domain")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 3407])
    parser.add_argument("--unit", choices=("case", "domain"), default="case")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline, left_metrics = load_records(args.baseline)
    ours, right_metrics = load_records(args.ours)
    if not set(args.metrics) <= set(left_metrics) & set(right_metrics):
        raise ValueError("Requested metrics are missing from the per-case records")

    def select(rows):
        return [row for row in rows if row["channel"] == args.channel
                and (args.domain is None or row["domain"] == args.domain)]

    results = [paired_test(select(baseline), select(ours), metric, args.seeds, args.unit,
                           args.bootstrap_samples, args.bootstrap_seed) for metric in args.metrics]
    for result, adjusted in zip(results, holm_adjust([row["p_value_two_sided"] for row in results])):
        result["p_value_holm"] = adjusted
    write_json(args.output, {"arguments": vars(args), "results": results,
                            "seed_handling": "average repeated-seed scores within each case before pairing"})


if __name__ == "__main__":
    main()
