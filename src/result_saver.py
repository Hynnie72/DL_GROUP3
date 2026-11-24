"""Save inference and training results to JSON, CSV and PNG."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .sentiment_analyzer import SentimentResult


class ResultSaver:
    """Persist all experiment outputs in a structured, traceable way.

    Files produced (all in *reports_dir*)
    ---------------------------------------
    inference_results.json   — per-sentence inference details
    inference_results.csv    — flat tabular version
    training_metrics.json    — raw Trainer evaluation dict
    training_metrics.csv     — flat tabular version

    PNG figures go to *figures_dir* (usually ``artifacts/figures/``).
    """

    def __init__(
        self,
        reports_dir: str | Path = "reports",
        figures_dir: str | Path = "artifacts/figures",
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.figures_dir = Path(figures_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Inference results (Exercise 1)
    # ------------------------------------------------------------------

    def save_inference(self, results: list[SentimentResult]) -> None:
        """Write inference results to JSON + CSV."""
        records = [self._result_to_dict(r) for r in results]
        self._write_json(records, "inference_results.json")
        self._write_csv(
            records,
            "inference_results.csv",
            ["timestamp", "sentence", "sentiment", "confidence",
             "prob_negative", "prob_positive", "tokens"],
        )
        print("[ResultSaver] Inference results saved to reports/")

    @staticmethod
    def _result_to_dict(r: SentimentResult) -> dict[str, Any]:
        return {
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "sentence":     r.sentence,
            "sentiment":    r.sentiment,
            "confidence":   round(r.confidence, 6),
            "prob_negative": round(r.probabilities.get("NEGATIVE", 0.0), 6),
            "prob_positive": round(r.probabilities.get("POSITIVE", 0.0), 6),
            "tokens":       r.tokens,
        }

    # ------------------------------------------------------------------
    # Training metrics (Exercise 2)
    # ------------------------------------------------------------------

    def save_training_metrics(self, metrics: dict[str, Any]) -> None:
        """Write Trainer evaluation metrics to JSON + CSV."""
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **metrics,
        }
        self._write_json(payload, "training_metrics.json")
        rows = [{"metric": k, "value": v} for k, v in payload.items()]
        self._write_csv(rows, "training_metrics.csv", ["metric", "value"])
        print("[ResultSaver] Training metrics saved to reports/")

    # ------------------------------------------------------------------
    # Training curves (Loss + Accuracy)
    # ------------------------------------------------------------------

    def save_training_curves(self, history_csv: str | Path) -> None:
        """Read *history_csv* and save Loss + Accuracy curves as PNG.

        Expected CSV columns:
            epoch, train_loss, train_accuracy, val_loss, val_accuracy

        Output filename (timestamped):
            artifacts/figures/YYYYMMDD_HHMMSS_training_curves.png
        """
        import pandas as pd

        df = pd.read_csv(history_csv).dropna(subset=["train_loss"])
        x  = df["epoch"].tolist()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # --- Loss curve ---
        axes[0].plot(x, df["train_loss"], marker="o", label="Train")
        if "val_loss" in df.columns:
            axes[0].plot(x, df["val_loss"], marker="s", label="Validation")
        axes[0].set(title="Loss by Epoch", xlabel="Epoch", ylabel="Loss")
        axes[0].set_xticks(x)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # --- Accuracy curve ---
        if "train_accuracy" in df.columns:
            axes[1].plot(
                x, df["train_accuracy"] * 100, marker="o", label="Train"
            )
        if "val_accuracy" in df.columns:
            axes[1].plot(
                x, df["val_accuracy"] * 100, marker="s", label="Validation"
            )
        axes[1].set(
            title="Accuracy by Epoch", xlabel="Epoch", ylabel="Accuracy (%)"
        )
        axes[1].set_xticks(x)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.suptitle(
            "IMDb Sentiment Classification — DistilBERT Fine-tuning",
            fontsize=13,
        )
        fig.tight_layout()

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.figures_dir / f"{ts}_training_curves.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[ResultSaver] Training curves saved → {path.resolve()}")

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _write_json(self, data: Any, filename: str) -> None:
        path = self.reports_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    def _write_csv(
        self,
        rows: list[dict],
        filename: str,
        fieldnames: list[str],
    ) -> None:
        path = self.reports_dir / filename
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
