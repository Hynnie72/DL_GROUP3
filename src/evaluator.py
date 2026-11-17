"""Evaluator: metrics and plots for the predictions of one trained model."""

import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .utils import SEED


class Evaluator:
    def __init__(self, class_names: List[str], evaluation_images: np.ndarray):
        self.class_names = class_names
        self.evaluation_images = evaluation_images

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Accuracy (%) and macro-averaged Precision, Recall and F1-score."""
        return {
            "Accuracy (%)": accuracy_score(y_true, y_pred) * 100,
            "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "Macro F1-score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def per_class_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Share of each class that is predicted correctly (%), i.e. per-class recall."""
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        return np.diag(cm) / cm.sum(axis=1) * 100

    def plot_learning_curves(self, history: Dict[str, List[float]], title: str) -> None:
        """Loss and accuracy per epoch for train and evaluation, with the best evaluation epoch marked."""
        epochs = range(1, len(history["train_loss"]) + 1)
        best_epoch = int(np.argmax(history["evaluation_accuracy"])) + 1

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(epochs, history["train_loss"], marker="o", markersize=3, label="Train Loss")
        axes[0].plot(epochs, history["evaluation_loss"], marker="o", markersize=3, label="Evaluation Loss")
        axes[0].set_title("Loss theo epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        axes[1].plot(epochs, history["train_accuracy"], marker="o", markersize=3, label="Train Accuracy")
        axes[1].plot(epochs, history["evaluation_accuracy"], marker="o", markersize=3, label="Evaluation Accuracy")
        axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.7,
                        label=f"Evaluation Acc cao nhất (epoch {best_epoch})")
        axes[1].set_title("Accuracy theo epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.suptitle(f"Đường cong học - {title}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
        """Confusion matrix as counts and row-normalised (%), then the 5 most frequent mistakes."""
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        cm_norm = cm / cm.sum(axis=1, keepdims=True) * 100

        fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
        for ax, matrix, sub, is_count in [
            (axes[0], cm, "Số lượng ảnh", True),
            (axes[1], cm_norm, "Chuẩn hóa theo hàng (%)", False),
        ]:
            im = ax.imshow(matrix, cmap="Blues")
            ax.set_title(f"{title}\nMa trận nhầm lẫn - {sub}", fontsize=13)
            ax.set_xlabel("Nhãn dự đoán")
            ax.set_ylabel("Nhãn thật")
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.set_xticklabels(self.class_names, rotation=45, ha="right")
            ax.set_yticklabels(self.class_names)
            thresh = matrix.max() / 2
            for r in range(10):
                for c in range(10):
                    v = matrix[r, c]
                    text = f"{int(v)}" if is_count else f"{v:.1f}"
                    ax.text(c, r, text, ha="center", va="center", fontsize=9,
                            color="white" if v > thresh else "black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

        pairs = sorted(
            ((cm[r, c], self.class_names[r], self.class_names[c])
             for r in range(10) for c in range(10) if r != c),
            reverse=True
        )
        print("5 cặp nhầm lẫn nhiều nhất (nhãn thật -> dự đoán):")
        for count, true_name, pred_name in pairs[:5]:
            print(f"  {true_name} -> {pred_name}: {count} ảnh")

    def plot_predictions(self, y_true, y_pred, y_prob, title: str,
                         n_correct: int = 6, n_wrong: int = 6, seed: int = SEED) -> None:
        """Random correct (green) and wrong (red) evaluation predictions with their confidence."""
        rng = random.Random(seed)
        correct = np.where(y_pred == y_true)[0].tolist()
        wrong = np.where(y_pred != y_true)[0].tolist()
        selected = (rng.sample(correct, min(n_correct, len(correct)))
                    + rng.sample(wrong, min(n_wrong, len(wrong))))
        rng.shuffle(selected)

        n_cols = 4
        n_rows = int(np.ceil(len(selected) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.7 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, i in zip(axes, selected):
            ok = y_true[i] == y_pred[i]
            ax.imshow(self.evaluation_images[i], cmap="gray")
            ax.set_title(
                f"Thật: {self.class_names[y_true[i]]}\n"
                f"Dự đoán: {self.class_names[y_pred[i]]}\n"
                f"Xác suất: {y_prob[i][y_pred[i]] * 100:.2f}%",
                color="green" if ok else "red", fontsize=10
            )
            ax.axis("off")
        for ax in axes[len(selected):]:
            ax.axis("off")
        plt.suptitle(f"Dự đoán trên tập kiểm tra - {title}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()
