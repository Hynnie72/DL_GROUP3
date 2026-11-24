"""Exercise 2 — Hugging Face Trainer wrapper with per-epoch logging."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .config import TrainingConfig


# ---------------------------------------------------------------------------
# Step 5 (EX2-05) — Metric function used by HF Trainer
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """Compute Accuracy, Precision, Recall and F1 from logits.

    Called automatically by HF Trainer at every evaluation step.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy":  float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
    }


# ---------------------------------------------------------------------------
# Custom callback — print Train + Val metrics after every epoch
# ---------------------------------------------------------------------------

class EpochMetricsCallback(TrainerCallback):
    """Print loss and accuracy on Train and Validation after each epoch.

    HF Trainer only logs eval metrics when ``eval_strategy`` fires.
    This callback forces evaluation on both splits at epoch end so we
    can display a clean per-epoch table without touching the test set.
    """

    def __init__(self, train_dataset, validation_dataset) -> None:
        self._train_ds = train_dataset
        self._val_ds   = validation_dataset
        self.trainer: Trainer | None = None   # injected after Trainer init

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control

        train_m = self.trainer.evaluate(
            eval_dataset=self._train_ds,
            metric_key_prefix="train",
        )
        val_m = self.trainer.evaluate(
            eval_dataset=self._val_ds,
            metric_key_prefix="val",
        )

        epoch = int(state.epoch or 0)
        total = int(args.num_train_epochs)
        print(
            f"\nEpoch [{epoch}/{total}]"
            f"  Train  Loss={train_m.get('train_loss', 0):.4f}"
            f"  Acc={train_m.get('train_accuracy', 0)*100:.2f}%"
            f"  |  Val  Loss={val_m.get('val_loss', 0):.4f}"
            f"  Acc={val_m.get('val_accuracy', 0)*100:.2f}%",
            flush=True,
        )
        return control


# ---------------------------------------------------------------------------
# Step 6 (EX2-06) — FineTuningTrainer
# ---------------------------------------------------------------------------

class FineTuningTrainer:
    """OOP wrapper around Hugging Face ``Trainer`` (Exercise 2, Step 6).

    Parameters
    ----------
    config : TrainingConfig
    model  : pre-built model from ``TextClassificationModel.build()``
    tokenizer : HF tokenizer (from ``IMDbDataModule``)
    train_dataset, validation_dataset : tokenized HF Datasets
    """

    def __init__(
        self,
        config: TrainingConfig,
        model,
        tokenizer,
        train_dataset,
        validation_dataset,
    ) -> None:
        self.config    = config
        self.model     = model
        self.tokenizer = tokenizer
        self._start_time: float | None = None

        self._epoch_cb = EpochMetricsCallback(train_dataset, validation_dataset)
        data_collator  = DataCollatorWithPadding(tokenizer=tokenizer)

        # Step 5 — Training Arguments
        training_args = TrainingArguments(
            output_dir                  = config.output_dir,
            num_train_epochs            = config.epochs,
            per_device_train_batch_size = config.batch_size,
            per_device_eval_batch_size  = config.batch_size,
            learning_rate               = config.learning_rate,
            weight_decay                = config.weight_decay,
            max_grad_norm               = config.max_grad_norm,
            eval_strategy               = "no",       # handled by callback
            save_strategy               = "epoch",
            logging_strategy            = "steps",
            logging_steps               = config.logging_steps,
            load_best_model_at_end      = False,
            save_total_limit            = 2,
            report_to                   = "none",
            seed                        = config.seed,
        )

        self.trainer = Trainer(
            model            = self.model,
            args             = training_args,
            train_dataset    = train_dataset,
            eval_dataset     = validation_dataset,
            processing_class = tokenizer,
            data_collator    = data_collator,
            compute_metrics  = compute_metrics,
            callbacks        = [self._epoch_cb],
        )
        self._epoch_cb.trainer = self.trainer

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        """Run fine-tuning; returns HF TrainOutput."""
        self._start_time = time.time()
        result = self.trainer.train()
        elapsed = time.time() - self._start_time
        print(f"\n[Trainer] Training finished in {elapsed/60:.1f} min.")
        return result

    # ------------------------------------------------------------------
    # Persistence — save model + tokenizer with timestamp
    # ------------------------------------------------------------------

    def save(self) -> Path:
        """Save fine-tuned model and tokenizer with a timestamp subdirectory.

        Output path:
            saved_model/YYYYMMDD_HHMMSS_distilbert_imdb_last/

        Returns
        -------
        Path
            Absolute path of the saved model directory.
        """
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_distilbert_imdb_last"
        out  = Path(self.config.saved_model_dir) / name
        out.mkdir(parents=True, exist_ok=True)

        self.trainer.save_model(str(out))
        self.tokenizer.save_pretrained(str(out))
        print(f"[Trainer] Model saved → {out.resolve()}")
        return out
