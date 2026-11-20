"""Launch seeded DG training, initialization ablations and per-case evaluation."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.variants import VARIANTS
from datasets.manifest import read_manifest


ROOT = Path(__file__).resolve().parent.parent


def build_plan(config, config_dir, output_dir, sam_checkpoint, only=None, skip_training=False):
    if config.get("protocol") != "dg":
        raise ValueError("This launcher implements leave-one-domain-out DG")
    if config["task"] not in ("vessel", "odoc"):
        raise ValueError("Unknown task")
    seeds = config.get("seeds", [0, 42, 3407])
    variants = only or config.get("variants", ["joint"])
    if not seeds or len(set(seeds)) != len(seeds) or any(name not in VARIANTS for name in variants):
        raise ValueError("Seeds must be distinct and variants must be registered")
    if not config.get("splits"):
        raise ValueError("At least one DG split is required")
    common = ["--task", config["task"], "--sam-type", config.get("sam_type", "vit_b")]
    if config.get("use_sam", True):
        if sam_checkpoint is None:
            raise ValueError("A SAM checkpoint is required unless use_sam is false")
        common += ["--sam-checkpoint", str(sam_checkpoint)]
    else:
        common += ["--no-sam"]
    epochs = config.get("epochs", {"stage1": 100, "stage2": 100})
    steps, eta = config.get("sampling", {}).get("steps", 50), config.get("sampling", {}).get("eta", 0)
    plan = []

    def command(module, arguments, log):
        plan.append({"command": [sys.executable, "-m", module, *map(str, arguments)],
                     "log": str(log)})

    def resolve(path):
        return str((config_dir / path).resolve())

    for split in config["splits"]:
        name = split["name"]
        if not name or any(char in name for char in "/\\:") or name in (".", ".."):
            raise ValueError("Split names must be simple directory names")
        for seed in seeds:
            run_dir = output_dir / name / f"seed_{seed}"
            stage1 = run_dir / "stage1.pt"
            fda_options = config.get("fda")
            fda_checkpoint = run_dir / "fda.pt" if fda_options is not None else None
            if fda_checkpoint is not None and not skip_training:
                command("experiments.pretrain_fda", ["--protocol", "dg", "--task", config["task"],
                        "--source-manifest", resolve(split["source_train"]),
                        "--style-manifest", resolve(split["source_train"]),
                        "--epochs", fda_options["epochs"], "--beta", fda_options.get("beta", 0.1),
                        "--batch-size", config.get("batch_size", 2), "--seed", seed,
                        "--output", fda_checkpoint], run_dir / "logs" / "fda.log")
            train_common = [*common, "--manifest", resolve(split["source_train"]), "--seed", seed,
                            "--batch-size", config.get("batch_size", 2)]
            stage2_common = [*train_common]
            if fda_checkpoint is not None:
                stage2_common += ["--fda-checkpoint", fda_checkpoint]
            if split.get("source_val"):
                train_common += ["--val-manifest", resolve(split["source_val"])]
            if config.get("amp", False):
                train_common += ["--amp"]
            if not skip_training:
                command("train", [*train_common, "--stage", 1, "--epochs", epochs["stage1"],
                                      "--output", stage1], run_dir / "logs" / "stage1.log")
            for cue_conditioning in sorted({VARIANTS[variant]["cue_conditioning"] for variant in variants}):
                suffix = "extra_condition" if cue_conditioning else "standard"
                if not skip_training:
                    command("train", [*stage2_common, "--stage", 2, "--epochs", epochs["stage2"],
                                          "--stage1-checkpoint", stage1, "--output", run_dir / f"stage2_{suffix}.pt",
                                          *(["--cue-conditioning"] if cue_conditioning else [])],
                            run_dir / "logs" / f"stage2_{suffix}.log")
            for variant in variants:
                suffix = "extra_condition" if VARIANTS[variant]["cue_conditioning"] else "standard"
                predictions = run_dir / variant / "predictions"
                arguments = [*common, "--manifest", resolve(split["target"]),
                             "--stage1-checkpoint", stage1,
                             "--stage2-checkpoint", run_dir / f"stage2_{suffix}.pt",
                             "--seed", seed, "--steps", steps, "--eta", eta,
                             "--initialization", variant, "--output-dir", predictions]
                if config.get("save_features", False):
                    arguments += ["--save-features"]
                command("infer", arguments, run_dir / "logs" / f"infer_{variant}.log")
                evaluation = ["--manifest", resolve(split["target"]), "--predictions", predictions,
                              "--method", variant, "--setting", f"{config['task']}_dg", "--seed", seed]
                command("evaluate", ["--task", config["task"], *evaluation,
                                         "--output", run_dir / variant / "metrics.csv"],
                        run_dir / "logs" / f"evaluate_{variant}.log")
                if config["task"] == "vessel" and config.get("structure_metrics", True):
                    structure = config.get("structure", {})
                    command("experiments.evaluate_structure", [*evaluation, "--output", run_dir / variant / "structure.csv",
                                                       "--thin-diameter", structure.get("thin_diameter", 3),
                                                       "--bifurcation-tolerance", structure.get("bifurcation_tolerance", 2)],
                            run_dir / "logs" / f"structure_{variant}.log")
    return plan


def validate_splits(config, config_dir):
    for split in config["splits"]:
        source = read_manifest(str(config_dir / split["source_train"]), config["task"], True)
        target = read_manifest(str(config_dir / split["target"]), config["task"], True)
        source_domains = {record.get("domain") for record in source}
        target_domains = {record.get("domain") for record in target}
        if len(source_domains) != 3 or len(target_domains) != 1 or None in source_domains | target_domains:
            raise ValueError("Each fold requires three labeled source domains and one named target domain")
        if source_domains & target_domains:
            raise ValueError("DG source and held-out target domains overlap")
        if split.get("source_val"):
            validation = read_manifest(str(config_dir / split["source_val"]), config["task"], True)
            if not {record.get("domain") for record in validation} <= source_domains:
                raise ValueError("Validation must contain only source-domain samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--only", nargs="+", choices=tuple(VARIANTS))
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running or creating artifacts")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(args.output_dir).absolute().resolve()
    sam_checkpoint = Path(args.sam_checkpoint).absolute().resolve() if args.sam_checkpoint else None
    plan = build_plan(config, config_path.parent, output, sam_checkpoint,
                      args.only, args.skip_training)
    if args.dry_run:
        for item in plan:
            print(shlex.join(item["command"]))
        print(f"{len(plan)} commands; dry run only")
        return
    if config.get("use_sam", True) and (not args.sam_checkpoint or not Path(args.sam_checkpoint).is_file()):
        raise FileNotFoundError(args.sam_checkpoint)
    validate_splits(config, config_path.parent)
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps({"configuration": config, "commands": plan}, indent=2) + "\n",
                                       encoding="utf-8")
    for index, item in enumerate(plan, 1):
        log = Path(item["log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(plan)}] {shlex.join(item['command'])}", flush=True)
        environment = dict(os.environ, PYTHONHASHSEED=str(item["command"][item["command"].index("--seed") + 1]))
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(item["command"], cwd=ROOT, env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
