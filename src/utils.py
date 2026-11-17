"""Shared constants and helper functions used by every other module."""

import random
import re
import unicodedata

import numpy as np
import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
CLASS_NAMES_VI = [
    "Áo thun", "Quần dài", "Áo len chui đầu", "Váy", "Áo khoác",
    "Dép xăng-đan", "Áo sơ mi", "Giày thể thao", "Túi xách", "Bốt cổ thấp"
]


def slugify(text: str) -> str:
    """Turn a Vietnamese experiment name into an ASCII file name.

    Example: 'Cơ sở: 2 lớp' -> 'co_so_2_lop'.
    """
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def set_seed(seed: int) -> None:
    """Fix the random seed for Python, NumPy and PyTorch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
