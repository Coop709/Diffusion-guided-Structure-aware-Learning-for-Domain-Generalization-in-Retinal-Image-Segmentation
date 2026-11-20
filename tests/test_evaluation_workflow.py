"""Exercise saved predictions -> per-case CSV -> summaries and feature MMD."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments import aggregate_results
from experiments import analyze_domain_gap
import evaluate
from experiments import evaluate_structure
from experiments import statistical_tests


class EvaluationWorkflowTests(unittest.TestCase):
    def invoke(self, module, *arguments):
        with patch.object(sys, "argv", [module.__name__, *map(str, arguments)]), contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def test_file_pairing_evaluation_aggregation_statistics_and_mmd(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            domains = ("STARE", "HRF", "DRIVE", "CHASEDB1")
            entries = {}
            for domain in domains:
                data_dir = root / domain
                data_dir.mkdir()
                records = []
                for index in range(2):
                    truth = np.zeros((32, 32), dtype=np.uint8)
                    truth[6:26, 6:22] = 255
                    truth[2:30, 27] = 255
                    Image.fromarray(np.repeat(truth[..., None], 3, axis=2)).save(data_dir / f"image{index}.png")
                    Image.fromarray(truth).save(data_dir / f"mask{index}.png")
                    records.append({"image": f"image{index}.png", "mask": f"mask{index}.png",
                                    "domain": domain, "id": f"{domain}/{index}"})
                manifest = data_dir / "labels.jsonl"
                manifest.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
                feature_dir = data_dir / "features_run"
                (feature_dir / "features").mkdir(parents=True)
                metadata = dict(stage2_checkpoint="same_fold.pt", seed=42, initialization="joint",
                                steps=50, eta=0, no_pff=False, image_size=32, task="vessel")
                (feature_dir / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
                for index in range(2):
                    vectors = np.random.default_rng(index).normal(size=(4, 8, 8)).astype(np.float32)
                    np.savez(feature_dir / "features" / f"{index:06d}_image{index}.npz",
                             decoder_features=vectors, timestep=np.int64(0))
                entries[domain] = dict(manifest=str(manifest), predictions=str(feature_dir))

            manifest = root / "STARE" / "labels.jsonl"
            baseline_csvs, ours_csvs = [], []
            for seed in (0, 42, 3407):
                for method in ("Baseline", "Ours"):
                    predictions = root / f"{method}_{seed}"
                    predictions.mkdir()
                    (predictions / "run.json").write_text(json.dumps(dict(seed=seed, task="vessel", image_size=32)),
                                                          encoding="utf-8")
                    for index in range(2):
                        truth = np.asarray(Image.open(root / "STARE" / f"mask{index}.png")) > 127
                        probabilities = truth.astype(np.float32) * 0.8 + 0.1
                        if method == "Baseline":
                            probabilities = np.roll(probabilities, 2, axis=1)
                        np.save(predictions / f"{index:06d}_image{index}_probabilities.npy", probabilities[None])
                    output = root / f"{method}_{seed}.csv"
                    self.invoke(evaluate, "--task", "vessel", "--manifest", manifest,
                                "--predictions", predictions, "--size", 32, "--method", method,
                                "--setting", "vessel_dg", "--output", output)
                    (baseline_csvs if method == "Baseline" else ours_csvs).append(output)
            summary = root / "summary.csv"
            self.invoke(aggregate_results, "--inputs", *ours_csvs, "--output", summary)
            with summary.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            dice = next(row for row in rows if row["metric"] == "dice" and row["domain"] == "Average"
                        and row["channel"] == "overall")
            self.assertAlmostEqual(float(dice["mean"]), 100, places=5)
            self.assertEqual(float(dice["std"]), 0)

            structure = root / "structure.csv"
            self.invoke(evaluate_structure, "--manifest", manifest, "--predictions", root / "Ours_42",
                        "--size", 32, "--output", structure)
            with structure.open(newline="", encoding="utf-8") as stream:
                structure_rows = list(csv.DictReader(stream))
            self.assertEqual(float(structure_rows[0]["hd95"]), 0)
            self.assertEqual(float(structure_rows[0]["cldice"]), 100)

            paired = root / "paired.json"
            self.invoke(statistical_tests, "--baseline", *baseline_csvs, "--ours", *ours_csvs,
                        "--bootstrap-samples", 20, "--output", paired)
            self.assertEqual(json.loads(paired.read_text())["results"][0]["n_pairs"], 2)

            config_path = root / "mmd_config.json"
            config_path.write_text(json.dumps(dict(method="Ours", seed=42, target_domain="STARE", domains=entries)),
                                   encoding="utf-8")
            mmd_output = root / "mmd.csv"
            self.invoke(analyze_domain_gap, "--config", config_path, "--size", 32,
                        "--samples-per-class", 20, "--output", mmd_output)
            with mmd_output.open(newline="", encoding="utf-8") as stream:
                mmd_rows = list(csv.DictReader(stream))
            self.assertAlmostEqual(float(mmd_rows[0]["mmd_squared"]), 0.0, places=7)
            details = json.loads(mmd_output.with_suffix(".metadata.json").read_text())
            self.assertEqual(len(details[0]["pairs"]), 6)


if __name__ == "__main__":
    unittest.main()
