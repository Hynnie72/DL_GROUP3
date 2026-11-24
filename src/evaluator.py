"""Exercise 2 — Model evaluator on the held-out test set + Confusion Matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from transformers import Trainer


class ModelEvaluator:
    """Evaluate the fine-tuned model on an independent test dataset (Step 7).

    Parameters
    ----------
    trainer : Trainer
        The HF Trainer instance after ``train()`` has been called, **or** a
        Trainer that wraps a saved model loaded from disk.
    """

    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer

    # ------------------------------------------------------------------
    # Step 7 (EX2-07) — Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_dataset) -> dict[str, float]:
        """Run evaluation on *test_dataset* and return metrics dict.

        Keys: ``test_loss``, ``test_accuracy``, ``test_precision``,
        ``test_recall``, ``test_f1``.
        """
        return self.trainer.evaluate(
            eval_dataset=test_dataset,
            metric_key_prefix="test",
        )

    def get_predictions(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        """Return (predicted_labels, true_labels) arrays for *dataset*.

        Used to build the Confusion Matrix without loading a separate library.
        """
        output   = self.trainer.predict(dataset)
        preds    = np.argmax(output.predictions, axis=-1)
        labels   = output.label_ids
        return preds, labels

    def confusion_matrix(
        self,
        dataset,
        figures_dir: str | Path = "artifacts/figures",
    ) -> np.ndarray:
        """Compute and save a Confusion Matrix PNG.

        Parameters
        ----------
        dataset : HF Dataset
            Tokenised dataset (usually the test split).
        figures_dir : str or Path
            Directory where the PNG is saved.

        Returns
        -------
        np.ndarray
            2 × 2 confusion matrix [[TN, FP], [FN, TP]].
        """
        from datetime import datetime

        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        preds, labels = self.get_predictions(dataset)
        cm = confusion_matrix(labels, preds)

        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["NEGATIVE", "POSITIVE"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix — Test Set")
        fig.tight_layout()

        out_dir = Path(figures_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{ts}_confusion_matrix.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Evaluator] Confusion matrix saved → {path.resolve()}")
        return cm
