import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.checkpoints import load_checkpoint, save_checkpoint
from configs.model import ModelConfig
from datasets.pairing import paired_batches
from datasets.retina import RetinaDataset
from models.diffusion import GaussianDiffusion
from models.network import ConditionalDiffusionSegmenter, SourceSegmenter
from models.initialization import channel_guidance
from utils.common import device_name, seed_everything
from utils.fda_checkpoint import load_spectral_transfer, restore_spectral_transfer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the two stages in Fig. Total_Arch")
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--task", choices=("vessel", "odoc"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest")
    parser.add_argument("--protocol", choices=("dg", "da"), default="dg")
    parser.add_argument("--target-manifest", help="Unlabeled target images for DA training")
    parser.add_argument("--fda-checkpoint", help="Adversarially pretrained spectral-transfer weights")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--no-sam", action="store_true",
                        help="Use trainable image encoders instead of SAM initialization and transfer")
    parser.add_argument("--sam-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--resume", help="Continue training from a checkpoint of this stage")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    parser.add_argument("--cue-conditioning", action="store_true",
                        help="Train the additional-condition ablation branch in Stage 2")
    parser.add_argument("--no-pff", action="store_true", help="Train without the pre-fusion filter")
    return parser.parse_args()


def run_epoch(model, diffusion, loader, device, loss_function, optimizer=None, scheduler=None, scaler=None,
              amp=False, guidance_model=None, use_pff=True, fda=None, target_loader=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = (paired_batches(loader, target_loader) if target_loader is not None else
               ((batch, None) for batch in loader))
    for batch, target_batch in batches:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        condition = image
        if fda is not None:
            # Source geometry and labels stay paired; target images supply only style.
            style = (target_batch["image"].to(device, non_blocking=True) if target_batch is not None else
                     image[torch.randperm(len(image), device=device)])
            indices = torch.arange(len(image), device=device) % len(style)
            with torch.no_grad():
                condition = fda(image, style[indices])
        elif target_loader is not None:
            raise ValueError("DA training requires a pretrained FDA module")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=amp):
                cues = None
                if guidance_model is not None:
                    with torch.no_grad():
                        coarse = guidance_model(condition).sigmoid()
                        cues = (coarse, channel_guidance(condition, coarse))
                loss = (loss_function(model(condition), mask) if diffusion is None else
                        diffusion.training_loss(model, mask, condition, use_pff=use_pff, structural_cues=cues))
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
        total_loss += loss.detach().item()
    return total_loss / len(loader)


def main():
    args = parse_args()
    if args.stage == 2 and not args.stage1_checkpoint:
        raise ValueError("Stage 2 requires --stage1-checkpoint")
    if args.stage == 1 and args.cue_conditioning:
        raise ValueError("--cue-conditioning applies only to Stage 2")
    if not args.no_sam and not args.sam_checkpoint:
        raise ValueError("--sam-checkpoint is required unless --no-sam is selected")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Epochs and batch size must be positive; workers cannot be negative")
    if args.protocol == "da" and not args.target_manifest:
        raise ValueError("DA training requires --target-manifest with unlabeled images")
    if args.protocol == "dg" and args.target_manifest:
        raise ValueError("DG training must not access a target-domain manifest")
    seed_everything(args.seed)
    config = ModelConfig(task=args.task, sam_type=args.sam_type)
    device = torch.device(device_name())
    saved = load_checkpoint(args.resume, args.stage, config) if args.resume else None
    fda = load_spectral_transfer(args.fda_checkpoint, device) if args.fda_checkpoint else None
    if saved is not None:
        resumed_fda = restore_spectral_transfer(saved["fda"], device) if saved.get("fda") else None
        if fda is not None:
            if resumed_fda is None or fda.beta != resumed_fda.beta or any(
                    not torch.equal(value, resumed_fda.state_dict()[name]) for name, value in fda.state_dict().items()):
                raise ValueError("FDA weights must match the resumed training run")
        fda = resumed_fda
    if args.protocol == "da" and fda is None:
        raise ValueError("DA training requires --fda-checkpoint, or FDA weights stored in --resume")
    if args.protocol == "dg" and fda is not None and fda.pretraining.get("protocol") != "dg":
        raise ValueError("DG training requires FDA pretraining restricted to source domains")
    train_data = RetinaDataset(args.manifest, args.task, training=True, size=config.image_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if args.protocol == "dg" and fda is not None:
        source_domains = {record.get("domain") for record in train_data.records}
        pretrained_domains = set(fda.pretraining.get("source_domains", [])) | set(
            fda.pretraining.get("style_domains", []))
        if None in source_domains or not pretrained_domains or not pretrained_domains <= source_domains:
            raise ValueError("FDA pretraining domains must belong to this DG fold's source domains")
    target_loader = None
    if args.target_manifest:
        target_data = RetinaDataset(args.target_manifest, args.task, training=True,
                                    size=config.image_size, with_labels=False)
        target_loader = DataLoader(target_data, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = None
    if args.val_manifest:
        val_data = RetinaDataset(args.val_manifest, args.task, training=False,
                                 size=config.image_size, with_labels=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda")

    guidance_model = None
    if args.stage == 1:
        model = SourceSegmenter(config, args.sam_checkpoint, use_sam=not args.no_sam)
        diffusion = None
        loss_function = nn.BCEWithLogitsLoss()
    else:
        source_model = SourceSegmenter(config, args.sam_checkpoint, use_sam=not args.no_sam)
        source_model.load_state_dict(load_checkpoint(args.stage1_checkpoint, 1, config)["model"])
        model = ConditionalDiffusionSegmenter.from_stage1(
            config, source_model, args.sam_checkpoint, cue_conditioning=args.cue_conditioning,
            use_sam=not args.no_sam)
        if args.cue_conditioning:
            guidance_model = source_model.to(device).eval()
            for parameter in guidance_model.parameters():
                parameter.requires_grad_(False)
        del source_model
        diffusion = GaussianDiffusion(config).to(device)
        loss_function = None

    model = model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=1e-4, betas=(0.99, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    start_epoch = 0
    if args.resume:
        if saved.get("model_options", {}).get("cue_conditioning", False) != args.cue_conditioning:
            raise ValueError("Resume checkpoint uses a different cue-conditioning architecture")
        if saved.get("training_options", {}).get("use_pff", True) != (not args.no_pff):
            raise ValueError("Resume checkpoint uses a different PFF setting")
        if saved.get("training_options", {}).get("protocol", "dg") != args.protocol:
            raise ValueError("Resume checkpoint uses a different DG/DA protocol")
        recorded_target = saved.get("training_options", {}).get("target_manifest")
        current_target = str(target_data.manifest) if target_loader is not None else None
        if recorded_target != current_target:
            raise ValueError("Resume checkpoint uses a different target training manifest")
        if saved.get("seed") is not None and saved["seed"] != args.seed:
            raise ValueError("--seed must match the resumed run")
        if saved.get("planned_epochs") != args.epochs:
            raise ValueError("--epochs must match the original planned total when resuming")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if saved.get("scaler") is not None:
            scaler.load_state_dict(saved["scaler"])
        start_epoch = saved["epoch"]
        if start_epoch >= args.epochs:
            raise ValueError("The checkpoint has already reached --epochs")

    print(f"stage={args.stage} task={args.task} protocol={args.protocol} samples={len(train_data)} device={device}")
    for epoch in range(start_epoch, args.epochs):
        train_loss = run_epoch(model, diffusion, train_loader, device, loss_function,
                               optimizer, scheduler, scaler, amp=args.amp and device.type == "cuda",
                               guidance_model=guidance_model, use_pff=not args.no_pff,
                               fda=fda, target_loader=target_loader)
        val_loss = None
        if val_loader is not None:
            val_loss = run_epoch(model, diffusion, val_loader, device, loss_function,
                                 amp=args.amp and device.type == "cuda", guidance_model=guidance_model,
                                 use_pff=not args.no_pff)
        message = f"epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.6f}"
        if val_loss is not None:
            message += f" val_loss={val_loss:.6f}"
        print(message)
        save_checkpoint(args.output, model, args.stage, config, epoch + 1,
                        optimizer, scheduler, scaler, planned_epochs=args.epochs,
                        validation_loss=val_loss, seed=args.seed,
        model_options={"cue_conditioning": args.cue_conditioning, "use_sam": not args.no_sam},
                        training_options={"use_pff": not args.no_pff, "protocol": args.protocol,
                                          "source_manifest": str(train_data.manifest),
                                          "target_manifest": str(target_data.manifest) if target_loader is not None else None,
                                          "fda_conditioning": fda is not None}, fda=fda)


if __name__ == "__main__":
    main()
