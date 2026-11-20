import unittest
from pathlib import Path

from experiments.run_experiments import build_plan


class ExperimentPlanTests(unittest.TestCase):
    def test_dg_fda_pretraining_uses_only_source_manifest_and_precedes_segmentation(self):
        config = {"protocol": "dg", "task": "vessel", "seeds": [42], "variants": ["joint"],
                  "structure_metrics": False, "fda": {"epochs": 3},
                  "splits": [{"name": "STARE", "source_train": "source.jsonl", "target": "target.jsonl"}]}
        plan = build_plan(config, Path("configs"), Path("runs"), Path("sam.pth"))
        command = plan[0]["command"]
        self.assertEqual(command[1:3], ["-m", "experiments.pretrain_fda"])
        self.assertEqual(command[command.index("--protocol") + 1], "dg")
        self.assertEqual(command[command.index("--source-manifest") + 1],
                         command[command.index("--style-manifest") + 1])
        training = [item["command"] for item in plan if item["command"][1:3] == ["-m", "train"]]
        stage1 = next(item for item in training if item[item.index("--stage") + 1] == "1")
        stage2 = next(item for item in training if item[item.index("--stage") + 1] == "2")
        self.assertNotIn("--fda-checkpoint", stage1)
        self.assertIn("--fda-checkpoint", stage2)

    def test_non_sam_baseline_does_not_require_a_sam_checkpoint(self):
        config = {"protocol": "dg", "task": "vessel", "use_sam": False,
                  "seeds": [42], "variants": ["gaussian"], "structure_metrics": False,
                  "splits": [{"name": "STARE", "source_train": "source.jsonl", "target": "target.jsonl"}]}
        plan = build_plan(config, Path("configs"), Path("runs"), None)
        stage1 = next(item["command"] for item in plan
                      if item["command"][1:3] == ["-m", "train"] and
                      item["command"][item["command"].index("--stage") + 1] == "1")
        self.assertIn("--no-sam", stage1)
        self.assertNotIn("--sam-checkpoint", stage1)

    def test_additional_condition_has_its_own_training_and_checkpoint(self):
        config = {"protocol": "dg", "task": "vessel", "seeds": [42],
                  "variants": ["gaussian", "additional_condition", "joint"],
                  "structure_metrics": False,
                  "splits": [{"name": "STARE", "source_train": "source.jsonl", "target": "target.jsonl"}]}
        plan = build_plan(config, Path("configs"), Path("runs"), Path("sam.pth"))
        training = [item["command"] for item in plan if item["command"][1:3] == ["-m", "train"]]
        self.assertEqual(len(training), 3)
        self.assertEqual(sum("--cue-conditioning" in command for command in training), 1)
        inference = [item["command"] for item in plan if item["command"][1:3] == ["-m", "infer"]]
        extra = next(command for command in inference if command[command.index("--initialization") + 1] == "additional_condition")
        self.assertIn("extra_condition", extra[extra.index("--stage2-checkpoint") + 1])


if __name__ == "__main__":
    unittest.main()
