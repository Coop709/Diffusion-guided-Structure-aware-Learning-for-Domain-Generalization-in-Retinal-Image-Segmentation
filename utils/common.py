"""Reproducibility and small runtime helpers."""

import random

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"
