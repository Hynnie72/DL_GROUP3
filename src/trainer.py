
"""Trainer: training loop, validation, checkpoint management and prediction."""

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .model import MLP
from .utils import DEVICE


class Trainer:
    def __init__(
        self,
        model: MLP,
        config: ExperimentConfig,
        device: torch.device = DEVICE
    ):
        self.model = model
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "evaluation_loss": [],
            "train_accuracy": [],
            "evaluation_accuracy": []
        }

        # Best checkpoint tracking
        self.best_loss = float("inf")
        self.best_epoch = 0

        self.checkpoint_paths = {}

    def _build_optimizer(self) -> torch.optim.Optimizer:
        c = self.config

        if c.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=c.lr
            )

        if c.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=c.lr,
                momentum=c.momentum
            )

        raise ValueError(f"Optimizer không hợp lệ: {c.optimizer}")

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool
    ) -> Tuple[float, float]:

        self.model.train(train)

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * labels.size(0)
                correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()

                total += labels.size(0)

        return total_loss / total, 100.0 * correct / total

    # ---------------------------------------------------------
    # CHECKPOINT MANAGEMENT
    # ---------------------------------------------------------

    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        best_loss: float
    ) -> None:
        """Save complete training state."""

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": best_loss,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }

        torch.save(checkpoint, path)

    def fit(
        self,
        train_loader: DataLoader,
        evaluation_loader: DataLoader,
        log_every: int = 5,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:

        """Train and save best/last checkpoints."""

        epochs = self.config.epochs

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

            best_path = os.path.join(
                checkpoint_dir, "best_checkpoint.pth"
            )

            last_path = os.path.join(
                checkpoint_dir, "last_checkpoint.pth"
            )

            self.checkpoint_paths = {
                "best": best_path,
                "last": last_path
            }

        for epoch in range(1, epochs + 1):

            # 1. Training
            train_loss, train_acc = self._run_epoch(
                train_loader, train=True
            )

            # 2. Evaluation
            evaluation_loss, evaluation_acc = self._run_epoch(
                evaluation_loader, train=False
            )

            # 3. Update history
            self.history["train_loss"].append(train_loss)
            self.history["evaluation_loss"].append(evaluation_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["evaluation_accuracy"].append(evaluation_acc)

            # 4. Check whether current epoch is the best
            is_best = evaluation_loss < self.best_loss

            if is_best:
                self.best_loss = evaluation_loss
                self.best_epoch = epoch

                if checkpoint_dir is not None:
                    self._save_checkpoint(
                        path=best_path,
                        epoch=epoch,
                        best_loss=self.best_loss
                    )

                print(
                    f"  [BEST] Epoch {epoch:02d} | "
                    f"Evaluation Loss: {evaluation_loss:.6f}"
                )

            # 5. Always update last checkpoint
            if checkpoint_dir is not None:
                self._save_checkpoint(
                    path=last_path,
                    epoch=epoch,
                    best_loss=self.best_loss
                )

            # 6. Logging
            if (
                epoch == 1
                or epoch == epochs
                or epoch % log_every == 0
            ):
                print(
                    f"  Epoch {epoch:02d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Evaluation Loss: {evaluation_loss:.4f} | "
                    f"Evaluation Acc: {evaluation_acc:.2f}%"
                )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print(f"Best Epoch       : {self.best_epoch}")
        print(f"Best Eval Loss   : {self.best_loss:.6f}")

        if checkpoint_dir is not None:
            print(f"Best Checkpoint  : {best_path}")
            print(f"Last Checkpoint  : {last_path}")

        print("=" * 60)

        return self.history

    # ---------------------------------------------------------
    # CHECKPOINT LOADING
    # ---------------------------------------------------------

    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = False
    ) -> dict:
        """Load a checkpoint and optionally restore optimizer state."""

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if load_optimizer:
            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        self.best_loss = checkpoint["best_loss"]
        self.best_epoch = checkpoint["best_epoch"]
        self.history = checkpoint["history"]

        return checkpoint

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        self.model.eval()

        labels_all = []
        preds_all = []
        probs_all = []

        for images, labels in loader:

            probs = torch.softmax(
                self.model(images.to(self.device)),
                dim=1
            ).cpu()

            probs_all.append(probs.numpy())
            preds_all.append(probs.argmax(dim=1).numpy())
            labels_all.append(labels.numpy())

        return (
            np.concatenate(labels_all),
            np.concatenate(preds_all),
            np.concatenate(probs_all)
        )
