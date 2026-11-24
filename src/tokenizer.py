"""Hugging Face tokenizer adapter for Exercise 1 (inference)."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


class TextTokenizer:
    """Thin wrapper around ``AutoTokenizer`` that exposes only what
    ``SentimentAnalyzer`` needs."""

    def __init__(self, model_name: str) -> None:
        self._tok = AutoTokenizer.from_pretrained(model_name)

    # ------------------------------------------------------------------
    # Special tokens
    # ------------------------------------------------------------------

    @property
    def cls_token(self) -> str | None:
        """Classification start token (e.g. ``[CLS]``)."""
        return self._tok.cls_token

    @property
    def sep_token(self) -> str | None:
        """Separator / end token (e.g. ``[SEP]``)."""
        return self._tok.sep_token

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, sentence: str) -> dict[str, Any]:
        """Tokenize *sentence* and return PyTorch tensors.

        Returns a dict with keys ``input_ids`` and ``attention_mask``.
        """
        return self._tok(sentence, return_tensors="pt")

    def decode_tokens(self, input_ids: Any) -> list[str]:
        """Convert a 1-D tensor of token IDs to human-readable subword tokens."""
        return self._tok.convert_ids_to_tokens(input_ids)
