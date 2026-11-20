"""Pretrain spectral transfer against an image-domain discriminator."""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.pairing import paired_batches
from models.discriminator import ImageDiscriminator
from models.frequency import SpectralTransfer
from utils.common import device_name, seed_everything
from utils.fda_checkpoint import save_spectral_checkpoint


def adversarial_step(transfer, discriminator, source, style, generator_optimizer, discriminator_optimizer):
    """Alternate discriminator and spectral-transfer updates, without mask labels."""
    loss_function = nn.BCEWithLogitsLoss()
    discriminator_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        generated = transfer(source, style)
    real_logits = discriminator(style)
    fake_logits = discriminator(generated)
    discriminator_loss = 0.5 * (loss_function(real_logits, torch.ones_like(real_logits)) +
                                loss_function(fake_logits, torch.zeros_like(fake_logits)))
    discriminator_loss.backward()
    discriminator_optimizer.step()
    discriminator_optimizer.zero_grad(set_to_none=True)

    generator_optimizer.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    try:
        generated_logits = discriminator(transfer(source, style))
        generator_loss = loss_function(generated_logits, torch.ones_like(generated_logits))
        generator_loss.backward()
        generator_optimizer.step()
    finally:
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)
    return generator_loss.detach().item(), discriminator_loss.detach().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--style-manifest", required=True)
    parser.add_argument("--protocol", choices=("dg", "da"), required=True)
    parser.add_argument("--task", choices=("vessel", "odoc"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.size < 16 or args.lr <= 0:
        raise ValueError("Use positive epochs, batch size and learning rate, and size >= 16")
    seed_everything(args.seed)
    device = torch.device(device_name())
    from datasets.retina import RetinaDataset

    source_data = RetinaDataset(args.source_manifest, args.task, training=True,
                                size=args.size, with_labels=False)
    style_data = RetinaDataset(args.style_manifest, args.task, training=True,
                               size=args.size, with_labels=False)
    source_domains = {record.get("domain") for record in source_data.records}
    style_domains = {record.get("domain") for record in style_data.records}
    if None in source_domains | style_domains:
        raise ValueError("FDA pretraining manifests must name each sample's domain")
    if args.protocol == "dg" and not style_domains <= source_domains:
        raise ValueError("DG pretraining may use styles from source domains only")
    source_loader = DataLoader(source_data, batch_size=args.batch_size, shuffle=True)
    style_loader = DataLoader(style_data, batch_size=args.batch_size, shuffle=True)
    transfer = SpectralTransfer(args.beta).to(device)
    transfer.pretraining = dict(protocol=args.protocol, source_domains=sorted(source_domains),
                               style_domains=sorted(style_domains), seed=args.seed,
                               source_manifest=str(source_data.manifest), style_manifest=str(style_data.manifest),
                               objective="binary_cross_entropy_adversarial", discriminator="ImageDiscriminator",
                               image_size=args.size)
    discriminator = ImageDiscriminator().to(device)
    generator_optimizer = torch.optim.Adam(transfer.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    history = []
    for epoch in range(args.epochs):
        generator_total, discriminator_total, count = 0.0, 0.0, 0
        for source_batch, style_batch in paired_batches(source_loader, style_loader):
            size = min(len(source_batch["image"]), len(style_batch["image"]))
            source = source_batch["image"][:size].to(device)
            style = style_batch["image"][:size].to(device)
            generator_loss, discriminator_loss = adversarial_step(
                transfer, discriminator, source, style, generator_optimizer, discriminator_optimizer)
            generator_total += generator_loss
            discriminator_total += discriminator_loss
            count += 1
        losses = dict(epoch=epoch + 1, generator=generator_total / count,
                      discriminator=discriminator_total / count)
        history.append(losses)
        save_spectral_checkpoint(args.output, transfer, epoch=epoch + 1, seed=args.seed,
                                 options=vars(args), discriminator=discriminator.state_dict(),
                                 generator_optimizer=generator_optimizer.state_dict(),
                                 discriminator_optimizer=discriminator_optimizer.state_dict(), history=history)
        print(f"epoch {epoch + 1}/{args.epochs}: generator={losses['generator']:.6f} "
              f"discriminator={losses['discriminator']:.6f}")


if __name__ == "__main__":
    main()
