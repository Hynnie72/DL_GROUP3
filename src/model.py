"""Exercise 2 — DistilBERT model builder with backbone-freeze support.

Freeze Strategy
---------------
DistilBERT for Sequence Classification has this structure:

    DistilBertForSequenceClassification
    ├── distilbert          ← backbone  (embeddings + 6 transformer layers)
    │   ├── embeddings
    │   └── transformer
    ├── pre_classifier      ← Linear(768 → 768) + ReLU   ← HEAD
    ├── classifier          ← Linear(768 → num_labels)   ← HEAD
    └── dropout

``freeze_strategy = "backbone"``
    All backbone parameters are frozen (requires_grad = False).
    Only pre_classifier and classifier are trainable.

    Trainable parameter count:
        pre_classifier  : 768 × 768 (weight) + 768 (bias)  = 590,592
        classifier      : 768 × 2   (weight) + 2   (bias)  =   1,538
        ─────────────────────────────────────────────────────────────
        TOTAL trainable :   592,130  out of ~66.9 M  (~0.88 %)

    Benefits:
    - ~100× fewer parameters to update → faster training.
    - Backbone already encodes rich English semantics (pre-trained on
      BookCorpus + Wikipedia).
    - Reduces risk of catastrophic forgetting and overfitting.

``freeze_strategy = "none"``
    All ~66.9 M parameters are trainable (standard full fine-tuning).
    The existing saved_model/ was trained with this strategy.
"""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification

from .config import TrainingConfig


class TextClassificationModel:
    """Build a DistilBERT sequence-classification model.

    Parameters
    ----------
    config : TrainingConfig
        Provides ``model_name``, ``num_labels`` and ``freeze_strategy``.
    """

    ID2LABEL: dict[int, str] = {0: "NEGATIVE", 1: "POSITIVE"}
    LABEL2ID: dict[str, int] = {"NEGATIVE": 0, "POSITIVE": 1}

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def build(self) -> AutoModelForSequenceClassification:
        """Load pre-trained weights, attach classification head, apply freeze.

        Returns
        -------
        AutoModelForSequenceClassification
            Model ready for training (or inference).
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            id2label=self.ID2LABEL,
            label2id=self.LABEL2ID,
        )

        if self.config.freeze_strategy == "backbone":
            self._freeze_backbone(model)
        # "none" → all parameters remain trainable

        self._print_param_summary(model)
        return model

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _freeze_backbone(
        model: AutoModelForSequenceClassification,
    ) -> None:
        """Freeze all DistilBERT transformer layers; keep only head trainable.

        Implementation detail
        ---------------------
        1. Freeze every parameter (requires_grad = False).
        2. Unfreeze pre_classifier (Linear 768→768).
        3. Unfreeze classifier   (Linear 768→num_labels).

        This is equivalent to using DistilBERT as a fixed feature extractor
        and training only the downstream classification head.
        """
        # Step 1 — freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Step 2 — unfreeze classification head
        for param in model.pre_classifier.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    @staticmethod
    def _print_param_summary(
        model: AutoModelForSequenceClassification,
    ) -> None:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen    = total - trainable
        pct       = trainable / total * 100
        print(
            f"[Model] Parameters — "
            f"total: {total:,} | "
            f"trainable: {trainable:,} ({pct:.2f}%) | "
            f"frozen: {frozen:,}"
        )
