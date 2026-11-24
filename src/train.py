"""
train.py – Training pipeline for Medical Image Regression (Chest X-ray Age Prediction)

Usage examples (Windows):
  # Experiment 1: MAE Loss, no augmentation
  python train.py --loss mae --augment_strategy none --exp_name exp1_mae_noaug

  # Experiment 2: MSE Loss, no augmentation
  python train.py --loss mse --augment_strategy none --exp_name exp2_mse_noaug

  # Experiment 3: MAE Loss + standard augmentation
  python train.py --loss mae --augment_strategy standard --exp_name exp3_mae_aug

  # Experiment 4: MSE Loss + strong augmentation
  python train.py --loss mse --augment_strategy strong --exp_name exp4_mse_strongaug

  # Resume from latest checkpoint
  python train.py --resume --exp_name exp1_mae_noaug

  # Quick smoke-test (1 epoch, tiny batch, CPU)
  python train.py --epochs 1 --batch_size 4 --num_workers 0 --exp_name smoke_test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root on sys.path so relative imports work
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dataset import (ChestXrayAgeDataset, get_train_transforms, get_val_transforms,
                     METADATA_DIM, FINDING_LABELS, NUM_FINDING_LABELS)
from model import MetadataAwareEfficientNetV2
from utils import (
    TrainingHistory,
    check_nan_inf,
    compute_l1_loss,
    compute_metrics,
    count_parameters,
    format_time,
    get_device,
    get_logger,
    load_checkpoint,
    plot_training_curves,
    save_checkpoint,
    set_seed,
)


# ===========================================================================
# Paths — override with env vars for cross-platform compatibility
# (Kaggle/Colab: set env vars before running, or pass CLI args)
# ===========================================================================
_REPO_ROOT    = _SRC_DIR.parent

# Allow env-var overrides so the same script works on Kaggle, Colab, and local
_DATA_CSV     = Path(os.environ.get(
    "CXR_DATA_CSV",
    r"D:\datasets\CXR8\Data_Entry_2017_v2020.csv"
))
_IMAGES_ROOT  = Path(os.environ.get(
    "CXR_IMAGES_ROOT",
    r"D:\datasets\CXR8\images"
))
_TRAIN_LIST   = Path(os.environ.get(
    "CXR_TRAIN_LIST",
    r"D:\datasets\CXR8\train_val_list.txt"
))
_TEST_LIST    = Path(os.environ.get(
    "CXR_TEST_LIST",
    r"D:\datasets\CXR8\test_list.txt"
))
_TRAINED_DIR  = Path(os.environ.get(
    "CXR_TRAINED_DIR",
    str(_REPO_ROOT / "trained_model")
))
_ARTIFACT_DIR = Path(os.environ.get(
    "CXR_ARTIFACT_DIR",
    str(_REPO_ROOT / "artifact")
))

# ===========================================================================
# Argument Parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MetadataAwareEfficientNetV2 for Chest X-ray Age Regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment
    p.add_argument("--exp_name", type=str, default="experiment",
                   help="Experiment name (used for checkpoint/artifact sub-directory).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Debug / quick-run
    p.add_argument("--max_batches", type=int, default=0,
                   help="Limit number of batches per epoch (0 = no limit). "
                        "Use for quick smoke-testing on CPU.")

    # Data
    p.add_argument("--data_csv",    type=str, default=str(_DATA_CSV))
    p.add_argument("--images_root", type=str, default=str(_IMAGES_ROOT))
    p.add_argument("--train_list",  type=str, default=str(_TRAIN_LIST))
    p.add_argument("--test_list",   type=str, default=str(_TEST_LIST))
    p.add_argument("--val_split",   type=float, default=0.1,
                   help="Fraction of train_val set held out for validation.")
    p.add_argument("--target_col",  type=str, default="Patient Age",
                   help="CSV column name for the regression target.")
    p.add_argument("--min_age",     type=int, default=1)
    p.add_argument("--max_age",     type=int, default=100)
    p.add_argument("--image_size",  type=int, default=224)
    p.add_argument("--grayscale",   action="store_true", default=True,
                   help="Load images as single-channel grayscale.")

    # Data augmentation
    p.add_argument("--augment_strategy", type=str, default="standard",
                   choices=["none", "standard", "strong"],
                   help="Augmentation strategy for the training set.")

    # Model
    p.add_argument("--pretrained",        action="store_true", default=True)
    p.add_argument("--no_pretrained",     dest="pretrained", action="store_false")
    p.add_argument("--meta_embedding_dim", type=int, default=128)
    p.add_argument("--fusion_dim",         type=int, default=256)
    p.add_argument("--head_dropout",       type=float, default=0.3)

    # Training
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--device",        type=str,   default="auto",
                   help="'auto', 'cuda', or 'cpu'.")

    # Loss
    p.add_argument("--loss", type=str, default="mae", choices=["mae", "mse"],
                   help="Primary training loss function.")

    # Optimizer
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="L2 regularisation via AdamW weight_decay.")

    # L1 regularisation
    p.add_argument("--use_l1",    action="store_true", default=False,
                   help="Enable L1 regularisation.")
    p.add_argument("--lambda_l1", type=float, default=1e-5,
                   help="L1 regularisation coefficient.")

    # Scheduler
    p.add_argument("--scheduler", type=str, default="plateau",
                   choices=["plateau", "cosine", "none"],
                   help="Learning-rate scheduler.")
    p.add_argument("--patience",   type=int,   default=5,
                   help="ReduceLROnPlateau patience (epochs without improvement).")
    p.add_argument("--lr_factor",  type=float, default=0.5,
                   help="ReduceLROnPlateau reduction factor.")
    p.add_argument("--min_lr",     type=float, default=1e-7,
                   help="Minimum learning rate for scheduler.")

    # Gradient clipping
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Max gradient norm (0 to disable).")

    # Early stopping
    p.add_argument("--early_stop_patience", type=int, default=15,
                   help="Stop training if val MAE doesn't improve for this many epochs (0 to disable).")

    # Checkpointing
    p.add_argument("--resume",      action="store_true", default=False,
                   help="Resume training from latest checkpoint.")
    p.add_argument("--save_every",  type=int, default=1,
                   help="Save a 'latest' checkpoint every N epochs (always saves best).")

    # Paths
    p.add_argument("--trained_dir",  type=str, default=str(_TRAINED_DIR))
    p.add_argument("--artifact_dir", type=str, default=str(_ARTIFACT_DIR))

    return p.parse_args()


# ===========================================================================
# Loss function factory
# ===========================================================================

def build_criterion(loss_name: str) -> nn.Module:
    """Return an nn.Module loss function for the chosen loss."""
    if loss_name == "mae":
        return nn.L1Loss()
    elif loss_name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: '{loss_name}'. Choose 'mae' or 'mse'.")


# ===========================================================================
# Data loaders
# ===========================================================================

def build_dataloaders(
    args: argparse.Namespace,
    logger,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders.

    Train/Val split is derived from train_val_list.txt (stratified by random_split).
    Test DataLoader uses test_list.txt exclusively.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    logger.info("Building datasets …")

    train_tf = get_train_transforms(
        image_size=args.image_size,
        augment_strategy=args.augment_strategy,
        grayscale=args.grayscale,
    )
    val_tf = get_val_transforms(image_size=args.image_size, grayscale=args.grayscale)

    # Full train+val dataset (uses train_val_list.txt)
    full_dataset = ChestXrayAgeDataset(
        csv_path=args.data_csv,
        images_root=args.images_root,
        split_list_path=args.train_list,
        transform=None,       # transforms applied per-split below
        target_col=args.target_col,
        min_age=args.min_age,
        max_age=args.max_age,
    )

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Apply per-split transforms by wrapping in a thin adapter
    train_dataset = _TransformSubset(train_subset, train_tf)
    val_dataset   = _TransformSubset(val_subset,   val_tf)

    # Test dataset (separate list, val transforms)
    test_dataset = ChestXrayAgeDataset(
        csv_path=args.data_csv,
        images_root=args.images_root,
        split_list_path=args.test_list,
        transform=val_tf,
        target_col=args.target_col,
        min_age=args.min_age,
        max_age=args.max_age,
    )

    age_stats = full_dataset.get_age_stats()
    logger.info(
        f"Dataset split — Train: {n_train}, Val: {n_val}, Test: {len(test_dataset)}"
    )
    logger.info(
        f"Target '{args.target_col}' stats (train+val): "
        f"mean={age_stats['mean']:.1f}, std={age_stats['std']:.1f}, "
        f"min={age_stats['min']:.0f}, max={age_stats['max']:.0f}"
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),   # only pin when CUDA is actually available
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


class _TransformSubset(torch.utils.data.Dataset):
    """Applies a transform to a Subset of ChestXrayAgeDataset."""

    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, metadata, target = self.subset[idx]
        # image at this point is a PIL Image (transform=None in base dataset)
        if self.transform is not None:
            image = self.transform(image)
        return image, metadata, target


# ===========================================================================
# Model builder
# ===========================================================================

def build_model(args: argparse.Namespace, logger) -> MetadataAwareEfficientNetV2:
    logger.info("Building model (MetadataAwareEfficientNetV2) …")
    in_channels = 1 if args.grayscale else 3
    model = MetadataAwareEfficientNetV2(
        metadata_dim=METADATA_DIM,
        pretrained=args.pretrained,
        in_channels=in_channels,
        meta_embedding_dim=args.meta_embedding_dim,
        fusion_dim=args.fusion_dim,
        head_dropout=args.head_dropout,
    )
    logger.info(f"Trainable parameters: {count_parameters(model):,}")
    return model


# ===========================================================================
# Optimizer & Scheduler
# ===========================================================================

def build_optimizer(model: nn.Module, args: argparse.Namespace):
    """
    AdamW with per-parameter groups:
      - BatchNorm and bias → no weight_decay (L2)
      - Everything else   → weight_decay = args.weight_decay
    """
    no_decay = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            for pname, _ in module.named_parameters(recurse=False):
                no_decay.add(f"{name}.{pname}" if name else pname)

    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and n not in no_decay and not n.endswith(".bias")
            ],
            "weight_decay": args.weight_decay,
            "name": "decay",
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and (n in no_decay or n.endswith(".bias"))
            ],
            "weight_decay": 0.0,
            "name": "no_decay",
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    return optimizer


def build_scheduler(optimizer, args: argparse.Namespace):
    """Return the LR scheduler (or None if 'none')."""
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.patience,
            min_lr=args.min_lr,
        )
    elif args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    elif args.scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


# ===========================================================================
# Training step (one epoch)
# ===========================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    logger,
) -> Dict[str, float]:
    """
    Run one full training epoch.

    Returns
    -------
    dict: train_loss, train_mae, train_mse, train_l1_reg, num_samples
    """
    model.train()

    total_data_loss = 0.0
    total_l1_loss   = 0.0
    all_preds:   List[float] = []
    all_targets: List[float] = []
    num_samples = 0

    pbar = tqdm(loader, desc=f"  [Train] Epoch {epoch}", leave=False, ncols=100)

    for batch_idx, (images, metadata, targets) in enumerate(pbar):
        # Debug: limit batches per epoch
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break

        images   = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        targets  = targets.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        preds = model(images, metadata).squeeze(1)   # [B]

        # Sanity check
        if check_nan_inf(preds, f"predictions (batch {batch_idx})"):
            logger.warning(f"NaN/Inf in predictions at batch {batch_idx}. Skipping.")
            continue

        # Data loss (MAE or MSE)
        data_loss = criterion(preds, targets)

        # L1 regularisation
        l1_loss = compute_l1_loss(model, args.lambda_l1 if args.use_l1 else 0.0)

        # Total loss
        total_loss = data_loss + l1_loss

        if check_nan_inf(total_loss, f"total_loss (batch {batch_idx})"):
            logger.warning(f"NaN/Inf in loss at batch {batch_idx}. Skipping backward.")
            continue

        # Backward
        total_loss.backward()

        # Gradient clipping
        if args.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Accumulate (weighted by batch size for correct epoch-level average)
        bs = images.size(0)
        total_data_loss += data_loss.item() * bs
        total_l1_loss   += l1_loss.item()   * bs
        num_samples += bs

        preds_np   = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        all_preds.extend(preds_np.tolist())
        all_targets.extend(targets_np.tolist())

        # Live tqdm postfix
        batch_mae = float(np.mean(np.abs(preds_np - targets_np)))
        pbar.set_postfix(
            loss=f"{data_loss.item():.3f}",
            mae=f"{batch_mae:.2f}",
        )

    pbar.close()

    if num_samples == 0:
        logger.warning("No valid samples processed in this training epoch!")
        return {
            "train_loss": float("nan"), "train_mae": float("nan"),
            "train_mse": float("nan"), "train_l1_reg": float("nan"),
            "num_samples": 0,
        }

    epoch_loss = total_data_loss / num_samples
    epoch_l1   = total_l1_loss   / num_samples
    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))

    return {
        "train_loss":   epoch_loss,
        "train_mae":    metrics["mae"],
        "train_mse":    metrics["mse"],
        "train_l1_reg": epoch_l1,
        "num_samples":  num_samples,
    }


# ===========================================================================
# Validation / Evaluation step
# ===========================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Val",
    max_batches: int = 0,
) -> Dict[str, float]:
    """
    Evaluate model on a DataLoader.  No gradients, no augmentation.

    Returns
    -------
    dict: loss, mae, mse, rmse, r2
    """
    model.eval()

    total_data_loss = 0.0
    all_preds:   List[float] = []
    all_targets: List[float] = []
    num_samples = 0

    pbar = tqdm(loader, desc=f"  [{desc}]  ", leave=False, ncols=100)

    for batch_idx, (images, metadata, targets) in enumerate(pbar):
        # Debug: limit batches
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images   = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        targets  = targets.to(device, non_blocking=True)

        preds = model(images, metadata).squeeze(1)  # [B]

        bs = images.size(0)
        loss = criterion(preds, targets)
        total_data_loss += loss.item() * bs
        num_samples += bs

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

        batch_mae = float(np.mean(np.abs(
            preds.cpu().numpy() - targets.cpu().numpy()
        )))
        pbar.set_postfix(loss=f"{loss.item():.3f}", mae=f"{batch_mae:.2f}")

    pbar.close()

    if num_samples == 0:
        return {k: float("nan") for k in ["loss", "mae", "mse", "rmse", "r2"]}

    epoch_loss = total_data_loss / num_samples
    metrics    = compute_metrics(np.array(all_preds), np.array(all_targets))

    return {
        "loss": epoch_loss,
        "mae":  metrics["mae"],
        "mse":  metrics["mse"],
        "rmse": metrics["rmse"],
        "r2":   metrics["r2"],
    }


# ===========================================================================
# Epoch Logging (formatted output)
# ===========================================================================

def log_epoch(
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float],
    val_metrics:   Dict[str, float],
    best_val_mae:  float,
    best_epoch:    int,
    current_lr:    float,
    epoch_duration: float,
    args: argparse.Namespace,
    logger,
) -> None:
    """Print a detailed, well-formatted epoch summary."""
    sep  = "=" * 60
    dash = "-" * 60

    logger.info(sep)
    logger.info(f"EPOCH [{epoch}/{total_epochs}]")
    logger.info(sep)

    logger.info("[TRAIN]")
    logger.info(f"  - Loss ({args.loss.upper()}): {train_metrics['train_loss']:.5f}")
    logger.info(f"  - MAE              : {train_metrics['train_mae']:.4f} years")
    logger.info(f"  - MSE              : {train_metrics['train_mse']:.4f}")
    logger.info(f"  - L1 Regularisation: {train_metrics['train_l1_reg']:.6f}"
                + (" (enabled)" if args.use_l1 else " (disabled)"))
    logger.info(f"  - L2 (weight_decay): {args.weight_decay:.2e} via AdamW")
    logger.info(f"  - Learning Rate    : {current_lr:.2e}")
    logger.info(f"  - Samples          : {train_metrics['num_samples']}")
    if args.grad_clip > 0:
        logger.info(f"  - Grad Clip        : {args.grad_clip}")

    logger.info("[VALIDATION]")
    logger.info(f"  - Loss ({args.loss.upper()}): {val_metrics['loss']:.5f}")
    logger.info(f"  - MAE  : {val_metrics['mae']:.4f} years")
    logger.info(f"  - MSE  : {val_metrics['mse']:.4f}")
    logger.info(f"  - RMSE : {val_metrics['rmse']:.4f}")
    r2_str = f"{val_metrics['r2']:.4f}" if not math.isnan(val_metrics['r2']) else "N/A"
    logger.info(f"  - R²   : {r2_str}")

    logger.info("[MODEL STATUS]")
    logger.info(f"  - Best Val MAE : {best_val_mae:.4f} years")
    logger.info(f"  - Best Epoch   : {best_epoch}")
    logger.info(f"  - Learning Rate: {current_lr:.2e}")
    logger.info(f"  - Epoch Duration: {format_time(epoch_duration)}")
    logger.info(dash)


# ===========================================================================
# Main Training Loop
# ===========================================================================

def train(args: argparse.Namespace) -> None:
    """
    Full training pipeline:
      setup → data → model → optimizer → loop → evaluate → save artifacts
    """
    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    set_seed(args.seed)

    # Directories
    exp_ckpt_dir  = os.path.join(args.trained_dir,  args.exp_name)
    exp_art_dir   = os.path.join(args.artifact_dir, args.exp_name)
    os.makedirs(exp_ckpt_dir, exist_ok=True)
    os.makedirs(exp_art_dir,  exist_ok=True)

    log_path = os.path.join(exp_art_dir, "train.log")
    logger   = get_logger(name=args.exp_name, log_file=log_path)

    logger.info("=" * 60)
    logger.info("Medical Image Regression — Chest X-ray Age Prediction")
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Loss function   : {args.loss.upper()}")
    logger.info(f"Aug strategy    : {args.augment_strategy}")
    logger.info(f"L1 regularisation: {'ON (λ=' + str(args.lambda_l1) + ')' if args.use_l1 else 'OFF'}")
    logger.info(f"L2 (weight_decay): {args.weight_decay}")
    logger.info(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Metadata dim    : {METADATA_DIM}  (sex=1, view=1, findings={NUM_FINDING_LABELS})")
    logger.info(f"Finding labels  : {FINDING_LABELS}")
    logger.info("=" * 60)

    # Device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # GPU info (useful for Kaggle/Colab debugging)
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM total : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        # benchmark=True speeds up training when input size is fixed
        torch.backends.cudnn.benchmark = True
    logger.info(f"  PyTorch version: {torch.__version__}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader = build_dataloaders(args, logger)

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model = build_model(args, logger)
    model = model.to(device)

    # ------------------------------------------------------------------
    # 3. Loss, Optimizer, Scheduler
    # ------------------------------------------------------------------
    criterion = build_criterion(args.loss)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    logger.info(f"Criterion : {criterion.__class__.__name__} ({args.loss.upper()})")
    logger.info(f"Optimizer : AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    logger.info(f"Scheduler : {args.scheduler}")

    # ------------------------------------------------------------------
    # 4. Resume from checkpoint?
    # ------------------------------------------------------------------
    start_epoch = 1
    best_val_mae = float("inf")
    best_epoch   = 0
    history      = TrainingHistory()
    early_stop_counter = 0

    latest_ckpt = os.path.join(exp_ckpt_dir, "latest_checkpoint.pth")
    if args.resume and os.path.isfile(latest_ckpt):
        ckpt = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, device, logger
        )
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_val_mae = ckpt.get("best_mae", float("inf"))
        best_epoch   = ckpt.get("best_epoch", 0)
        early_stop_counter = ckpt.get("early_stop_counter", 0)
        # Reload history
        history_path = os.path.join(exp_art_dir, "training_history.json")
        if os.path.isfile(history_path):
            history = TrainingHistory.from_json(history_path)
        logger.info(f"Resuming from epoch {start_epoch}.")
    elif args.resume:
        logger.warning(f"No checkpoint found at '{latest_ckpt}'. Starting fresh.")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    logger.info(f"Starting training from epoch {start_epoch} to {args.epochs}.")

    # Paths used inside loop (defined once for clarity)
    history_json_path = os.path.join(exp_art_dir, "training_history.json")
    history_csv_path  = os.path.join(exp_art_dir, "training_history.csv")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()

            # --- Train ---
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, args, epoch, logger
            )

            # --- Validate ---
            val_metrics = evaluate(model, val_loader, criterion, device, desc="Val",
                                   max_batches=args.max_batches)

            # --- Scheduler step ---
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["mae"])
                else:
                    scheduler.step()
                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr != current_lr:
                    logger.info(f"  LR adjusted: {current_lr:.2e} → {new_lr:.2e}")
                current_lr = new_lr

            epoch_duration = time.time() - epoch_start

            # --- Log epoch ---
            log_epoch(
                epoch, args.epochs,
                train_metrics, val_metrics,
                best_val_mae, best_epoch,
                current_lr, epoch_duration,
                args, logger,
            )

            # --- History ---
            history.append({
                "epoch":            epoch,
                "train_loss":       train_metrics["train_loss"],
                "train_mae":        train_metrics["train_mae"],
                "train_mse":        train_metrics["train_mse"],
                "train_l1_reg":     train_metrics["train_l1_reg"],
                "val_loss":         val_metrics["loss"],
                "val_mae":          val_metrics["mae"],
                "val_mse":          val_metrics["mse"],
                "val_rmse":         val_metrics["rmse"],
                "val_r2":           val_metrics["r2"],
                "lr":               current_lr,
                "epoch_duration_s": epoch_duration,
            })

            # --- Checkpoint state ---
            ckpt_state = {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_mae":             best_val_mae,
                "best_epoch":           best_epoch,
                "early_stop_counter":   early_stop_counter,
                "config":               vars(args),
            }

            # Save latest checkpoint every N epochs (critical for resume)
            if epoch % args.save_every == 0:
                save_checkpoint(ckpt_state, exp_ckpt_dir, "latest_checkpoint.pth")
                logger.info(f"  → Latest checkpoint saved (epoch {epoch}).")

            # Save best model
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best_epoch   = epoch
                early_stop_counter = 0
                ckpt_state["best_mae"]   = best_val_mae
                ckpt_state["best_epoch"] = best_epoch
                save_checkpoint(ckpt_state, exp_ckpt_dir, "best_model.pth")
                logger.info(
                    f"  ★ New best Val MAE = {best_val_mae:.4f} at epoch {epoch}"
                    f" → best_model.pth saved."
                )
            else:
                early_stop_counter += 1

            # Save per-epoch checkpoint (safety net for Kaggle/Colab time limits)
            save_checkpoint(ckpt_state, exp_ckpt_dir, f"epoch_{epoch:04d}.pth")

            # Flush history to disk after EVERY epoch
            # (if training is killed mid-run, all completed epochs are preserved)
            history.to_json(history_json_path)
            history.to_csv(history_csv_path)

            # --- Early stopping ---
            if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
                logger.info(
                    f"Early stopping triggered: no Val MAE improvement for "
                    f"{early_stop_counter} epochs."
                )
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        logger.warning("Saving emergency checkpoint before exit ...")
        if "ckpt_state" in locals():
            save_checkpoint(ckpt_state, exp_ckpt_dir, "emergency_checkpoint.pth")
            history.to_json(history_json_path)
            history.to_csv(history_csv_path)
            logger.warning(f"Emergency checkpoint saved at epoch {epoch}.")
        logger.warning(f"To resume: python train.py --resume --exp_name {args.exp_name}")
        return

    # ------------------------------------------------------------------
    # 6. Save training history & plots
    # ------------------------------------------------------------------
    logger.info("Saving training history and plots …")
    history.to_csv(history_csv_path)
    history.to_json(history_json_path)
    plot_training_curves(history, exp_art_dir, experiment_name=args.exp_name)
    logger.info(f"Artifacts saved to: {exp_art_dir}")

    # ------------------------------------------------------------------
    # 7. Final evaluation on test set
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 60)

    best_ckpt_path = os.path.join(exp_ckpt_dir, "best_model.pth")
    if not os.path.isfile(best_ckpt_path):
        logger.warning("best_model.pth not found. Using current model weights.")
    else:
        load_checkpoint(best_ckpt_path, model, device=device, logger=logger)
    model.eval()

    test_metrics = evaluate(model, test_loader, criterion, device, desc="Test",
                            max_batches=args.max_batches)

    logger.info("[TEST RESULTS]")
    logger.info(f"  - Loss ({args.loss.upper()}): {test_metrics['loss']:.5f}")
    logger.info(f"  - MAE  : {test_metrics['mae']:.4f} years")
    logger.info(f"  - MSE  : {test_metrics['mse']:.4f}")
    logger.info(f"  - RMSE : {test_metrics['rmse']:.4f}")
    r2_str = (
        f"{test_metrics['r2']:.4f}" if not math.isnan(test_metrics['r2']) else "N/A"
    )
    logger.info(f"  - R²   : {r2_str}")
    logger.info(f"  - Best checkpoint from epoch {best_epoch}")

    # ------------------------------------------------------------------
    # 8. Save per-sample predictions CSV
    # ------------------------------------------------------------------
    logger.info("Saving per-sample test predictions …")
    _save_predictions_csv(
        model, test_loader, device,
        out_path=os.path.join(exp_art_dir, "test_predictions.csv"),
        logger=logger,
        max_batches=args.max_batches,
    )

    # Summary JSON
    summary = {
        "experiment":       args.exp_name,
        "loss_fn":          args.loss,
        "augment_strategy": args.augment_strategy,
        "epochs_trained":   best_epoch if best_epoch > 0 else args.epochs,
        "best_epoch":       best_epoch,
        "best_val_mae":     best_val_mae,
        "test_mae":         test_metrics["mae"],
        "test_mse":         test_metrics["mse"],
        "test_rmse":        test_metrics["rmse"],
        "test_r2":          test_metrics["r2"],
    }
    with open(os.path.join(exp_art_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete.")
    logger.info(f"Best Val MAE : {best_val_mae:.4f} (epoch {best_epoch})")
    logger.info(f"Test MAE     : {test_metrics['mae']:.4f}")


# ===========================================================================
# Save predictions CSV
# ===========================================================================

@torch.no_grad()
def _save_predictions_csv(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: str,
    logger,
    max_batches: int = 0,
) -> None:
    """Save actual age, predicted age, and error to a CSV file."""
    model.eval()
    rows = []
    for batch_idx, (images, metadata, targets) in enumerate(
        tqdm(loader, desc="  [Predict]", leave=False, ncols=100)
    ):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images   = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        preds    = model(images, metadata).squeeze(1).cpu().numpy()
        tgts     = targets.numpy()
        for pred, tgt in zip(preds, tgts):
            rows.append({
                "actual_age":    round(float(tgt), 2),
                "predicted_age": round(float(pred), 4),
                "error":         round(float(pred - tgt), 4),
                "abs_error":     round(abs(float(pred - tgt)), 4),
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["actual_age", "predicted_age", "error", "abs_error"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"  Predictions saved: {out_path} ({len(rows)} samples)")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
