import unittest

from experiments.aggregate_results import aggregate
from experiments.statistical_tests import holm_adjust, paired_test


def record(seed, domain, score, sample="case", channel="vessel", method="Ours"):
    return dict(method=method, task="vessel", setting="dg", seed=seed, domain=domain,
                sample_id=f"{domain}/{sample}", channel=channel, dice=score)


class AggregationTests(unittest.TestCase):
    def test_average_sd_is_computed_from_run_averages(self):
        rows = []
        for seed, delta in ((0, -1), (42, 0), (3407, 1)):
            rows += [record(seed, "A", 80 + delta), record(seed, "B", 70 - delta)]
        output = aggregate(rows, ["dice"])
        average = next(row for row in output if row["domain"] == "Average" and row["channel"] == "overall")
        domain_a = next(row for row in output if row["domain"] == "A" and row["channel"] == "vessel")
        self.assertEqual(average["mean"], 75.0)
        self.assertEqual(average["std"], 0.0)
        self.assertEqual(domain_a["std"], 1.0)

    def test_domains_are_equally_weighted_despite_different_case_counts(self):
        rows = [record(42, "A", 100), record(42, "B", 0, "one"), record(42, "B", 0, "two")]
        output = aggregate(rows, ["dice"], seeds=[42])
        average = next(row for row in output if row["domain"] == "Average" and row["channel"] == "overall")
        self.assertEqual(average["mean"], 50.0)
        self.assertEqual(average["std"], "")

    def test_odoc_combines_both_structures_within_each_run(self):
        rows = [record(42, "A", 90, channel="disc"), record(42, "A", 70, channel="cup"),
                record(42, "B", 80, channel="disc"), record(42, "B", 60, channel="cup")]
        output = aggregate(rows, ["dice"], seeds=[42])
        average = next(row for row in output if row["domain"] == "Average" and row["channel"] == "overall")
        self.assertEqual(average["mean"], 75.0)

    def test_missing_seed_or_changed_cohort_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate([record(42, "A", 80)], ["dice"])
        with self.assertRaises(ValueError):
            aggregate([record(0, "A", 80), record(42, "B", 80)], ["dice"], seeds=[0, 42])

    def test_paired_analysis_does_not_treat_seeds_as_independent_cases(self):
        rows = [record(seed, "A", 80 + index, str(index)) for seed in (0, 42, 3407) for index in range(5)]
        result = paired_test(rows, rows, "dice", [0, 42, 3407], bootstrap_samples=20)
        self.assertEqual(result["n_pairs"], 5)
        self.assertEqual(result["p_value_two_sided"], 1.0)
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
