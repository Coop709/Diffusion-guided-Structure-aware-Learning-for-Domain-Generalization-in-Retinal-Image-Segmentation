"""Save and restore learned spectral-transfer parameters."""

from pathlib import Path

import torch

from models.frequency import SpectralTransfer


def spectral_payload(module):
    return {"kind": "adversarial_fda", "format_version": 1, "beta": module.beta,
            "model": module.state_dict(), "pretraining": getattr(module, "pretraining", {})}


def restore_spectral_transfer(payload, device):
    if payload.get("kind") != "adversarial_fda":
        raise ValueError("Expected an adversarial FDA checkpoint")
    module = SpectralTransfer(beta=float(payload["beta"])).to(device)
    module.load_state_dict(payload["model"])
    module.pretraining = payload.get("pretraining", {})
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def load_spectral_transfer(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return restore_spectral_transfer(payload.get("fda") or payload, device)


def save_spectral_checkpoint(path, module, **training_state):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(spectral_payload(module), **training_state)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
