"""Text validation and normalization — shared by Exercise 1 and Exercise 2."""

from __future__ import annotations


class TextProcessor:
    """Validate and normalise raw user-provided text before tokenization."""

    @staticmethod
    def clean(sentence: str) -> str:
        """Strip whitespace and reject empty or non-string inputs.

        Parameters
        ----------
        sentence : str
            Raw input text.

        Returns
        -------
        str
            Stripped, non-empty string.

        Raises
        ------
        TypeError
            If *sentence* is not a string.
        ValueError
            If *sentence* is empty after stripping.
        """
        if not isinstance(sentence, str):
            raise TypeError(f"Expected str, got {type(sentence).__name__}.")
        cleaned = sentence.strip()
        if not cleaned:
            raise ValueError("sentence must not be empty after stripping.")
        return cleaned
