import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from utils.checkpoints import load_checkpoint
from configs.model import ModelConfig
from datasets.retina import RetinaDataset
from models.diffusion import GaussianDiffusion
from utils.evaluation_io import write_json
from models.frequency import fda_transfer
from models.initialization import STRATEGIES
from models.network import ConditionalDiffusionSegmenter, SourceSegmenter
from utils.common import device_name, seed_everything
from utils.fda_checkpoint import load_spectral_transfer, restore_spectral_transfer


def parse_args():
    parser = argparse.ArgumentParser(description="Structure-aware DDIM inference (Fig. Infer_Arch)")
    parser.add_argument("--task", choices=("vessel", "odoc"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sam-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    spectral_options = parser.add_mutually_exclusive_group()
    spectral_options.add_argument("--fda-checkpoint", help="Pretrained FDA weights, if not embedded in Stage 2")
    spectral_options.add_argument("--fda-weight", type=float, help="Explicit fixed-weight FDA comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sal", action="store_true")
    parser.add_argument("--no-pff", action="store_true")
    parser.add_argument("--initialization", choices=STRATEGIES, default="joint")
    parser.add_argument("--save-features", action="store_true",
                        help="Export decoder features before the last 1x1 head at denoising step 0")
    parser.add_argument("--no-sam", action="store_true",
                        help="Restore the source-trained encoder instead of the original SAM encoder")
    return parser.parse_args()


def load_stage2_for_original_inference(model, checkpoint_state):
    """Restore Stage 2 weights while retaining the original SAM encoder.

    Stage 2 training freezes the transferred Stage 1 encoder, so its checkpoint
    contains ``segmentation_encoder.*`` keys even though Fig. 3 specifies the
    original pretrained encoder for the inference branch.  All other learned
    components, including the condition encoder, are restored here.
    """
    state = {
        key: value for key, value in checkpoint_state.items()
        if not key.startswith("segmentation_encoder.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected_missing = {
        key for key in model.state_dict()
        if key.startswith("segmentation_encoder.")
    }
    if set(missing) != expected_missing or unexpected:
        raise ValueError(
            "Stage-2 checkpoint does not match original-encoder inference: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def load_models(config, args, device):
    first = load_checkpoint(args.stage1_checkpoint, 1, config)
    second = load_checkpoint(args.stage2_checkpoint, 2, config)
    for saved in (first, second):
        if saved.get("seed") is not None and saved["seed"] != args.seed:
            raise ValueError("Inference --seed must match the trained run's seed")
    use_sam = second.get("model_options", {}).get("use_sam", not args.no_sam)
    if args.no_sam == use_sam:
        raise ValueError("--no-sam must match the Stage-2 checkpoint architecture")
    coarse_model = SourceSegmenter(config, args.sam_checkpoint, use_sam=use_sam)
    coarse_model.load_state_dict(first["model"])
    cue_conditioning = second.get("model_options", {}).get("cue_conditioning", False)
    if use_sam:
        diffusion_model = ConditionalDiffusionSegmenter.from_original(
            config, args.sam_checkpoint, cue_conditioning=cue_conditioning
        )
        load_stage2_for_original_inference(diffusion_model, second["model"])
    else:
        diffusion_model = ConditionalDiffusionSegmenter.from_stage1(
            config, coarse_model, None, cue_conditioning=cue_conditioning, use_sam=False)
        diffusion_model.load_state_dict(second["model"])
    fda = restore_spectral_transfer(second["fda"], device) if second.get("fda") else None
    if args.fda_checkpoint:
        supplied = load_spectral_transfer(args.fda_checkpoint, device)
        if fda is not None and (fda.beta != supplied.beta or any(
                not torch.equal(value, supplied.state_dict()[name]) for name, value in fda.state_dict().items())):
            raise ValueError("Supplied FDA weights differ from the Stage-2 checkpoint")
        fda = supplied
    if fda is not None and args.fda_weight is not None:
        raise ValueError("A fixed FDA weight cannot override pretrained FDA weights")
    return coarse_model.to(device).eval(), diffusion_model.to(device).eval(), fda


def main():
    args = parse_args()
    if not args.no_sam and not args.sam_checkpoint:
        raise ValueError("--sam-checkpoint is required unless --no-sam is selected")
    if args.no_sal:
        if args.initialization != "joint":
            raise ValueError("Use --no-sal or --initialization, not both")
        args.initialization = "gaussian"
    if args.fda_weight is not None and not 0 <= args.fda_weight <= 1:
        raise ValueError("--fda-weight must be in [0, 1]")
    seed_everything(args.seed)
    config = ModelConfig(task=args.task, sam_type=args.sam_type)
    device = torch.device(device_name())
    dataset = RetinaDataset(args.manifest, args.task, training=False, size=config.image_size)
    if args.no_sam and args.initialization != "gaussian" and any(
            "coarse_mask" not in record for record in dataset.records):
        raise ValueError("Structural initialization without SAM requires coarse_mask entries in the manifest")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    coarse_model, diffusion_model, fda = load_models(config, args, device)
    use_fda = fda is not None or args.fda_weight is not None
    if use_fda and any("source_image" not in record for record in dataset.records):
        raise ValueError("FDA inference requires a paired source_image in each target record")
    if not use_fda and any("source_image" in record for record in dataset.records):
        raise ValueError("Paired FDA inference requires pretrained weights or an explicit --fda-weight")
    diffusion = GaussianDiffusion(config).to(device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        (output / "features").mkdir(exist_ok=True)
    spectral_settings = dict(mode="pretrained" if fda is not None else "fixed" if use_fda else "disabled",
                             weight=float(fda.low_frequency_logit.sigmoid().item()) if fda is not None else args.fda_weight,
                             beta=fda.beta if fda is not None else 0.1 if use_fda else None,
                             pretraining=fda.pretraining if fda is not None else {})
    write_json(output / "run.json", dict(vars(args), image_size=config.image_size, fda=spectral_settings,
                                         stage2_checkpoint=str(Path(args.stage2_checkpoint).resolve()),
                                         segmentation_encoder_source=("stage1_non_sam" if args.no_sam else
                                                                      "original_pretrained_sam"),
                                         feature_layer="decoder.output before final 1x1 projection",
                                         feature_timestep=0))

    with torch.inference_mode():
        for batch in loader:
            target = batch["image"].to(device)
            condition = target
            if "source_image" in batch:
                source = batch["source_image"].to(device)
                condition = (fda(source, target) if fda is not None else
                             fda_transfer(source, target, target.new_tensor(args.fda_weight)))
            if args.no_sam and args.initialization != "gaussian":
                coarse = batch["coarse_mask"].to(device)
                if coarse.shape[1] == 1 and config.mask_channels > 1:
                    coarse = coarse.expand(-1, config.mask_channels, -1, -1)
                if coarse.shape[1] != config.mask_channels:
                    raise ValueError("coarse_mask channels must match the selected task")
            elif args.no_sam:
                coarse = torch.zeros((target.shape[0], config.mask_channels, *target.shape[-2:]),
                                     device=target.device, dtype=target.dtype)
            else:
                coarse = coarse_model(target).sigmoid()
            result = diffusion.ddim_sample(diffusion_model, target, coarse, condition,
                                           steps=args.steps, eta=args.eta, use_pff=not args.no_pff,
                                           initialization=args.initialization,
                                           return_features=args.save_features)
            probabilities, features = result if args.save_features else (result, None)
            name = batch["name"][0]
            values = probabilities[0].cpu().numpy()
            np.save(output / f"{name}_probabilities.npy", values)
            if features is not None:
                np.savez_compressed(output / "features" / f"{name}.npz",
                                    decoder_features=features[0].cpu().numpy().astype(np.float32),
                                    timestep=np.int64(0))
            labels = (values >= 0.5).astype(np.uint8) * 255
            for channel, label in enumerate(labels):
                suffix = "vessel" if args.task == "vessel" else ("disc", "cup")[channel]
                Image.fromarray(label).save(output / f"{name}_{suffix}.png")
            print(f"saved {name}")


if __name__ == "__main__":
    main()
