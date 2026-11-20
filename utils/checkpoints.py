"""Checkpoint format shared by training and inference."""

import os
from dataclasses import asdict
from pathlib import Path

import torch

from configs.model import ModelConfig
from utils.fda_checkpoint import spectral_payload


def save_checkpoint(path: str, model, stage: int, config: ModelConfig, epoch: int,
                    optimizer=None, scheduler=None, scaler=None, planned_epochs=None,
                    validation_loss=None, seed=None, model_options=None, training_options=None, fda=None):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "stage": stage,
        "task": config.task,
        "sam_type": config.sam_type,
        "config": asdict(config),
        "epoch": epoch,
        "planned_epochs": planned_epochs,
        "validation_loss": validation_loss,
        "seed": seed,
        "model_options": model_options or {},
        "training_options": training_options or {},
        "fda": spectral_payload(fda) if fda is not None else None,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(path: str, stage: int, config: ModelConfig):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("stage") != stage:
        raise ValueError(f"Expected a Stage-{stage} checkpoint: {path}")
    if payload.get("task") != config.task or payload.get("sam_type") != config.sam_type:
        raise ValueError(f"Checkpoint task/SAM variant differs from current configuration: {path}")
    saved_config = payload.get("config")
    if saved_config is not None and saved_config != asdict(config):
        raise ValueError(f"Checkpoint model configuration differs from current configuration: {path}")
    return payload
