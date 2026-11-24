"""Exercise 1 — OOP Sentiment Analysis (inference only, no fine-tuning)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification

from .config import InferenceConfig, load_inference_config
from .text_processor import TextProcessor
from .tokenizer import TextTokenizer


# ---------------------------------------------------------------------------
# Value object — returned by SentimentAnalyzer.analyze()
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SentimentResult:
    """Immutable structured result from one inference call."""

    sentence: str
    sentiment: str          # "POSITIVE" or "NEGATIVE"
    confidence: float       # probability of the predicted class ∈ [0, 1]
    tokens: list[str]       # decoded subword tokens including [CLS] / [SEP]
    probabilities: dict[str, float]   # {label: probability}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """Load a Hugging Face sentiment model and run inference.

    This class implements Exercise 1 requirements:
    1. Load ``transformers`` model from the Hub.
    2. Tokenize an input sentence.
    3. Perform sentiment analysis (inference only — weights are never updated).

    Parameters
    ----------
    model_name : str
        Hugging Face model ID or path to a local ``saved_model/`` directory.
    """

    def __init__(self, model_name: str | None = None) -> None:
        cfg: InferenceConfig = load_inference_config()
        self.model_name: str = model_name or cfg.model_name
        self._expected_labels: frozenset[str] = cfg.expected_labels

        self._tokenizer = TextTokenizer(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self._model.eval()          # inference mode — disable dropout
        self._validate_labels()

    # ------------------------------------------------------------------
    # Exercise 1 public methods
    # ------------------------------------------------------------------

    def tokenize(self, sentence: str) -> dict[str, Any]:
        """Return ``input_ids`` and ``attention_mask`` tensors.

        Parameters
        ----------
        sentence : str
            Raw English text.
        """
        return self._tokenizer.encode(TextProcessor.clean(sentence))

    def analyze(self, sentence: str) -> SentimentResult:
        """Classify *sentence* without updating model weights.

        Parameters
        ----------
        sentence : str
            Raw English text.

        Returns
        -------
        SentimentResult
            Contains sentiment label, confidence, decoded tokens and
            full class probability distribution.
        """
        clean = TextProcessor.clean(sentence)
        inputs = self._tokenizer.encode(clean)

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probs))
        label_map = self._model.config.id2label

        probabilities = {
            label_map[i]: float(p) for i, p in enumerate(probs)
        }
        sentiment = label_map[predicted_id]

        result = SentimentResult(
            sentence=clean,
            sentiment=sentiment,
            confidence=probabilities[sentiment],
            tokens=self._tokenizer.decode_tokens(inputs["input_ids"][0]),
            probabilities=probabilities,
        )
        self._validate_result(result)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path = "saved_model") -> None:
        """Save model and tokenizer to *output_dir* (created if absent).

        Reload later with::

            analyzer = SentimentAnalyzer("./saved_model")
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(out))
        self._tokenizer._tok.save_pretrained(str(out))
        print(f"[SentimentAnalyzer] Saved → {out.resolve()}")

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _validate_labels(self) -> None:
        actual = set(self._model.config.id2label.values())
        if actual != self._expected_labels:
            raise ValueError(
                f"Expected labels {sorted(self._expected_labels)}, "
                f"got {sorted(actual)}."
            )

    def _validate_result(self, result: SentimentResult) -> None:
        if result.sentiment not in self._expected_labels:
            raise ValueError(f"Unexpected sentiment: {result.sentiment!r}.")
        if not 0.0 <= result.confidence <= 1.0:
            raise ValueError("Confidence out of [0, 1].")
        if abs(sum(result.probabilities.values()) - 1.0) >= 1e-5:
            raise ValueError("Probabilities do not sum to 1.")
        if result.tokens[0] != self._tokenizer.cls_token:
            raise ValueError("First token must be CLS.")
        if result.tokens[-1] != self._tokenizer.sep_token:
            raise ValueError("Last token must be SEP.")
