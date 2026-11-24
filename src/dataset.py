"""Exercise 2 — IMDb dataset loader, splitter and tokenizer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from .config import TrainingConfig


class IMDbDataModule:
    """Read the local IMDb CSV, encode labels, split and tokenize.

    Usage
    -----
    >>> data = IMDbDataModule(config).load_data()
    >>> data.preprocess()
    >>> data.train_dataset, data.validation_dataset, data.test_dataset
    """

    LABEL_MAP: dict[str, int] = {"negative": 0, "positive": 1}

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        self.train_dataset: Dataset | None = None
        self.validation_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    # ------------------------------------------------------------------
    # Step 2 (EX2-02) — Load dataset
    # ------------------------------------------------------------------

    def load_data(self) -> "IMDbDataModule":
        """Read and validate the IMDb CSV; apply stratified split.

        Returns
        -------
        IMDbDataModule
            self — for method chaining.
        """
        path = Path(self.config.data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path.resolve()}\n"
                "Copy 'IMDB Dataset.csv' to the data/ directory."
            )

        df = pd.read_csv(path)
        required = {"review", "sentiment"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        df = df[["review", "sentiment"]].dropna().copy()
        df["sentiment"] = df["sentiment"].str.strip().str.lower()

        invalid = set(df["sentiment"].unique()) - set(self.LABEL_MAP)
        if invalid:
            raise ValueError(f"Unexpected labels in CSV: {invalid}")

        df["labels"] = df["sentiment"].map(self.LABEL_MAP).astype(int)
        df = df.rename(columns={"review": "text"})[["text", "labels"]]

        # Optional: use a small random subset for quick tests
        if self.config.use_small_sample:
            n = min(self.config.small_sample_size, len(df))
            df, _ = train_test_split(
                df, train_size=n,
                random_state=self.config.seed,
                stratify=df["labels"],
            )

        # Stratified split: train / validation / test
        train_val, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.seed,
            stratify=df["labels"],
        )
        relative_val = self.config.validation_size / (
            self.config.train_size + self.config.validation_size
        )
        train_df, val_df = train_test_split(
            train_val,
            test_size=relative_val,
            random_state=self.config.seed,
            stratify=train_val["labels"],
        )

        self.train_dataset      = Dataset.from_pandas(train_df.reset_index(drop=True))
        self.validation_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
        self.test_dataset       = Dataset.from_pandas(test_df.reset_index(drop=True))

        pos = int((df["labels"] == 1).sum())
        neg = len(df) - pos
        print(
            f"[Dataset] Total: {len(df):,} | "
            f"POSITIVE: {pos:,} ({pos/len(df):.1%}) | "
            f"NEGATIVE: {neg:,} ({neg/len(df):.1%})"
        )
        print(
            f"[Dataset] Split  train: {len(self.train_dataset):,} | "
            f"val: {len(self.validation_dataset):,} | "
            f"test: {len(self.test_dataset):,}"
        )
        return self

    # ------------------------------------------------------------------
    # Step 4 (EX2-04) — Preprocess / tokenize
    # ------------------------------------------------------------------

    def preprocess(self) -> "IMDbDataModule":
        """Tokenize all three splits in-place (batched map).

        Returns
        -------
        IMDbDataModule
            self — for method chaining.
        """
        def _tokenize(batch: dict) -> dict:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.config.max_length,
            )

        for split_name in ("train_dataset", "validation_dataset", "test_dataset"):
            ds = getattr(self, split_name)
            ds = ds.map(_tokenize, batched=True, remove_columns=["text"])
            setattr(self, split_name, ds)

        print(f"[Dataset] Tokenization done  (max_length={self.config.max_length})")
        return self
