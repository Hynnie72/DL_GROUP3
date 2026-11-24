"""
utils.py – Shared utilities for the Medical Image Regression project.

Provides:
  - Metric computation (MAE, MSE, RMSE, R²)
  - Checkpoint save / load
  - Training history management
  - Plotting helpers
  - Seed setup
  - Logger factory
  - L1 regularisation helper
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# ===========================================================================
# Logging
# ===========================================================================

def get_logger(name: str = "train", log_file: Optional[str] = None) -> logging.Logger:
    """
    Create and return a named logger.
    Outputs to stdout and optionally to a file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    return logger


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic=True ensures exact reproducibility but can slow down training.
    # benchmark=False required when deterministic=True.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: train.py overrides benchmark=True when CUDA is detected,
    # trading exact reproducibility for speed (acceptable for regression tasks).


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Parameters
    ----------
    preds   : 1-D numpy array of predictions
    targets : 1-D numpy array of ground-truth values

    Returns
    -------
    dict with keys: mae, mse, rmse, r2
    """
    preds   = np.asarray(preds,   dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()

    mae  = float(np.mean(np.abs(preds - targets)))
    mse  = float(np.mean((preds - targets) ** 2))
    rmse = math.sqrt(mse)

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# ===========================================================================
# L1 Regularisation
# ===========================================================================

def compute_l1_loss(
    model: nn.Module,
    lambda_l1: float,
    exclude_bn: bool = True,
    exclude_bias: bool = True,
) -> torch.Tensor:
    """
    Compute L1 penalty on model weights.

    Parameters
    ----------
    model       : the nn.Module
    lambda_l1   : regularisation strength
    exclude_bn  : skip BatchNorm parameters
    exclude_bias: skip bias parameters

    Returns
    -------
    scalar tensor (0-dim) on the same device as model parameters
    """
    if lambda_l1 <= 0.0:
        # Return a zero tensor on model's device
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        return torch.tensor(0.0, device=dev, requires_grad=False)

    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if exclude_bn and "bn" in name.lower():
            continue
        if exclude_bias and name.endswith(".bias"):
            continue
        l1 = l1 + param.abs().sum()

    return lambda_l1 * l1


# ===========================================================================
# Checkpoint
# ===========================================================================

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str,
) -> str:
    """
    Save a training checkpoint using an atomic write pattern.

    Writes to a temporary file first, then renames to the final path.
    This prevents a corrupt checkpoint if the process is killed mid-write
    (common on Kaggle/Colab with time limits).

    Parameters
    ----------
    state          : dict containing model_state_dict, optimizer_state_dict,
                     scheduler_state_dict, epoch, best_mae, config, ...
    checkpoint_dir : directory to save to
    filename       : e.g. 'best_model.pth' or 'latest_checkpoint.pth'

    Returns
    -------
    Full path of the saved file.
    """
    import tempfile
    import shutil

    os.makedirs(checkpoint_dir, exist_ok=True)
    final_path = os.path.join(checkpoint_dir, filename)
    tmp_path   = final_path + ".tmp"

    # Write to temp file first
    torch.save(state, tmp_path)

    # Atomic rename: if this succeeds, the file is complete and valid
    shutil.move(tmp_path, final_path)

    return final_path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler=None,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model (+ optionally optimizer, scheduler) state.

    Cross-platform safe: a checkpoint saved on GPU (Kaggle) loads correctly
    on CPU (Colab) and vice-versa via map_location.

    Returns
    -------
    The full checkpoint dict so the caller can read epoch, best_mae, config, etc.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device("cpu")

    # map_location remaps all tensors to the current device
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    if logger:
        logger.info(f"Loaded model weights from '{path}'")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # Move optimizer state tensors to current device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            if logger:
                logger.info("Restored optimizer state.")
        except Exception as e:
            if logger:
                logger.warning(f"Could not restore optimizer state: {e}. Starting fresh optimizer.")

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if logger:
                logger.info("Restored scheduler state.")
        except Exception as e:
            if logger:
                logger.warning(f"Could not restore scheduler state: {e}. Starting fresh scheduler.")

    epoch    = ckpt.get("epoch", 0)
    best_mae = ckpt.get("best_mae", float("inf"))
    if logger:
        logger.info(f"Resuming from epoch {epoch}, best MAE = {best_mae:.4f}")

    return ckpt


# ===========================================================================
# Training History
# ===========================================================================

class TrainingHistory:
    """Accumulates per-epoch metrics and can save/load from JSON."""

    KEYS = [
        "epoch",
        "train_loss", "train_mae", "train_mse",
        "train_l1_reg",
        "val_loss", "val_mae", "val_mse", "val_rmse", "val_r2",
        "lr",
        "epoch_duration_s",
    ]

    def __init__(self):
        self.records: List[Dict[str, float]] = []

    def append(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def to_csv(self, path: str) -> None:
        import csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not self.records:
            return
        fieldnames = list(self.records[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TrainingHistory":
        h = cls()
        with open(path, "r", encoding="utf-8") as f:
            h.records = json.load(f)
        return h

    def __len__(self) -> int:
        return len(self.records)

    def get_column(self, key: str) -> List[float]:
        return [r[key] for r in self.records if key in r]


# ===========================================================================
# Plotting
# ===========================================================================

def plot_training_curves(
    history: TrainingHistory,
    artifact_dir: str,
    experiment_name: str = "experiment",
) -> None:
    """
    Save four plots:
      1. Train / Val Loss
      2. Train / Val MAE
      3. Train / Val MSE
      4. Learning Rate schedule
    """
    os.makedirs(artifact_dir, exist_ok=True)
    epochs = history.get_column("epoch")

    def _save(fig_path, train_key, val_key, ylabel, title):
        train_vals = history.get_column(train_key)
        val_vals   = history.get_column(val_key)
        if not train_vals:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs[:len(train_vals)], train_vals, label="Train", marker="o", markersize=3)
        ax.plot(epochs[:len(val_vals)],   val_vals,   label="Val",   marker="s", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{experiment_name} – {title}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=100)
        plt.close(fig)

    _save(
        os.path.join(artifact_dir, f"{experiment_name}_loss.png"),
        "train_loss", "val_loss", "Loss", "Train vs Val Loss",
    )
    _save(
        os.path.join(artifact_dir, f"{experiment_name}_mae.png"),
        "train_mae", "val_mae", "MAE (years)", "Train vs Val MAE",
    )
    _save(
        os.path.join(artifact_dir, f"{experiment_name}_mse.png"),
        "train_mse", "val_mse", "MSE", "Train vs Val MSE",
    )

    # LR curve
    lr_vals = history.get_column("lr")
    if lr_vals:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs[:len(lr_vals)], lr_vals, color="green", marker=".", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"{experiment_name} – Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        lr_path = os.path.join(artifact_dir, f"{experiment_name}_lr.png")
        fig.savefig(lr_path, dpi=100)
        plt.close(fig)


# ===========================================================================
# Misc
# ===========================================================================

def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device if available and preferred, else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Return True (and print warning) if tensor contains NaN or Inf."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"[WARNING] {name} contains NaN or Inf values!")
        return True
    return False


def format_time(seconds: float) -> str:
    """Format seconds as mm:ss or hh:mm:ss."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
