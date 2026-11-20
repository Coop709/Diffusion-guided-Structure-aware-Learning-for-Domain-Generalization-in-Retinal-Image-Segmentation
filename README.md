# Diffusion-guided Structure-aware Learning for Domain Generalization in Retinal Image Segmentation

Model components, training and inference scripts, and evaluation utilities for retinal vessel and optic disc/cup (OD/OC) segmentation.

## Project Structure

```text
models/       SAM-based networks, U-ViT diffusion, and guidance modules
datasets/     Dataset loading, preprocessing, and manifest tools
evaluation/   Segmentation, structure, and feature-discrepancy metrics
experiments/  Experiment launchers and result analysis
configs/      Model settings and experiment configurations
utils/        Checkpoints, result files, and common utilities
tests/        Code checks
train.py      Training entry point
infer.py      Inference entry point
evaluate.py   Segmentation evaluation
```

## Installation

```bash
python -m pip install -r requirements.txt
```

Prepare the datasets and SAM checkpoint separately. The examples below use SAM ViT-B. Install a PyTorch build compatible with your environment. The baseline configuration uses trainable image encoders without SAM initialization.

## Data

Use JSONL manifests with one sample per line. Paths may be absolute or relative to the manifest file.

Vessel segmentation:

```json
{"image":"../data/DRIVE/images/01.png","mask":"../data/DRIVE/masks/01.png","source_image":"../data/STARE/images/im001.png","domain":"DRIVE"}
```

For OD/OC, replace `mask` with `disc_mask` and `cup_mask`, and provide `disc_center` for cropping. FDA inference uses `source_image` as the content reference and the manifest's `image` as the target-style image. Inference manifests can omit labels; evaluation manifests must contain labels and use the same sample order. For structural initialization with `--no-sam`, provide a precomputed `coarse_mask` in each inference record.

Manifest tools are available through `python -m datasets.make_manifest --help` and `python -m datasets.preflight --help`.

## Training and Inference

Run commands from the repository root. Example training commands:

```bash
python train.py --stage 1 --task vessel \
  --manifest manifests/source_train.jsonl \
  --sam-checkpoint checkpoints/sam_vit_b.pth \
  --epochs 100 --output checkpoints/stage1.pt

python -m experiments.pretrain_fda --protocol dg --task vessel \
  --source-manifest manifests/source_train.jsonl \
  --style-manifest manifests/source_train.jsonl \
  --epochs 10 --output checkpoints/fda.pt

python train.py --stage 2 --task vessel \
  --manifest manifests/source_train.jsonl \
  --sam-checkpoint checkpoints/sam_vit_b.pth \
  --fda-checkpoint checkpoints/fda.pt \
  --stage1-checkpoint checkpoints/stage1.pt \
  --epochs 100 --output checkpoints/stage2.pt
```

Inference and evaluation:

```bash
python infer.py --task vessel --manifest manifests/target.jsonl \
  --sam-checkpoint checkpoints/sam_vit_b.pth \
  --stage1-checkpoint checkpoints/stage1.pt \
  --stage2-checkpoint checkpoints/stage2.pt \
  --output-dir predictions/target

python evaluate.py --task vessel --manifest manifests/target_labeled.jsonl \
  --predictions predictions/target --output results/metrics.csv
```

Use `--task odoc` for OD/OC segmentation. Inference saves probability maps and masks; evaluation reports Dice, AUC, accuracy, sensitivity, and specificity.
During inference, the noisy-mask branch follows Fig. 3 and uses the original pretrained SAM encoder, while the condition branch restores the trained Stage 2 weights.
The full-model DG configurations also pretrain FDA using source-domain images and attach its checkpoint to Stage 2. Each target inference record therefore needs a paired source-domain `source_image`.

## FDA and Domain Adaptation

Pretrain the spectral-transfer module with source images and unlabeled target style images:

```bash
python -m experiments.pretrain_fda --protocol da --task vessel \
  --source-manifest manifests/source_train.jsonl \
  --style-manifest manifests/target_unlabeled.jsonl \
  --epochs 10 --output checkpoints/fda.pt
```

For DA training, add `--protocol da --target-manifest manifests/target_unlabeled.jsonl --fda-checkpoint checkpoints/fda.pt` to the training commands. Target images provide frequency information; source masks supply segmentation supervision. The pretrained FDA parameters are frozen and saved with the segmentation checkpoint. FDA inference uses the saved parameters and paired `source_image` entries in the target manifest.

For DG, FDA pretraining uses source domains only (`--protocol dg`). The experiment launcher can enable this with an `fda` entry such as `{"epochs": 10, "beta": 0.1}` in its configuration.

## Experiments

| Module | Purpose |
| --- | --- |
| `experiments.pretrain_fda` | Adversarial spectral-transfer pretraining |
| `experiments.run_experiments` | Multi-seed runs and initialization ablations |
| `experiments.aggregate_results` | Mean and standard-deviation summaries |
| `experiments.analyze_domain_gap` | Feature-distribution analysis |
| `experiments.evaluate_structure` | clDice, thin-vessel recall, bifurcation F1, and HD95 |
| `experiments.statistical_tests` | Paired comparisons of per-case results |

Feature-distribution analysis reports biased RBF MMD squared (MMD$^2$), without taking its square root. Structure evaluation uses a 3-pixel thin-vessel diameter threshold and a 2-pixel bifurcation matching radius at the 256 x 256 evaluation resolution.

Edit the dataset paths in `configs/` before launching experiments:

```bash
python -m experiments.run_experiments --config configs/vessel_dg.json \
  --sam-checkpoint checkpoints/sam_vit_b.pth --output-dir runs/vessel_dg
```

Use `configs/odoc_dg.json` for OD/OC or `configs/initialization_ablation.json` for initialization comparisons. Add `--dry-run` to preview commands. For module options, use `python -m <module> --help`.
Use `configs/vessel_baseline_dg.json` for the vessel baseline used by the feature and structure analyses.
