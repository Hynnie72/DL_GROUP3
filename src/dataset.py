"""
dataset.py – NIH ChestX-ray14 Dataset for Age Regression

Columns in Data_Entry_2017_v2020.csv:
  Image Index, Finding Labels, Follow-up #, Patient ID,
  Patient Age, Patient Sex, View Position,
  OriginalImage[Width, Height], OriginalImagePixelSpacing[x, y]

Metadata used as model input:
  - Patient Sex       (binary:  0=M, 1=F)
  - View Position     (binary:  0=PA, 1=AP)
  - Finding Labels    (multi-hot: 14-dim vector, one entry per disease)

Target: Patient Age (continuous, regression)

Metadata tensor layout (dim=16):
  [0]    sex_enc          (0 or 1)
  [1]    view_enc         (0 or 1)
  [2..15] multi-hot finding labels (order = FINDING_LABELS list)
"""

import os
import tarfile
import io
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEX_MAP:  Dict[str, int] = {"M": 0, "F": 1}
VIEW_MAP: Dict[str, int] = {"PA": 0, "AP": 1}

# 14 NIH ChestX-ray14 disease labels (alphabetical – fixed order for reproducibility)
FINDING_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]
NUM_FINDING_LABELS: int = len(FINDING_LABELS)          # 14
_LABEL_TO_IDX: Dict[str, int] = {l: i for i, l in enumerate(FINDING_LABELS)}

# Total metadata dimension: sex(1) + view(1) + findings(14) = 16
METADATA_DIM: int = 2 + NUM_FINDING_LABELS             # 16

IMG_MEAN: Tuple[float, ...] = (0.5006,)   # single-channel (grayscale) mean
IMG_STD:  Tuple[float, ...] = (0.2520,)   # single-channel (grayscale) std
# (approximate NIH ChestX-ray14 statistics, single-channel)


# ---------------------------------------------------------------------------
# Multi-hot encoder helper
# ---------------------------------------------------------------------------
def encode_finding_labels(finding_str: str) -> np.ndarray:
    """
    Convert a pipe-separated Finding Labels string to a 14-dim multi-hot vector.

    Parameters
    ----------
    finding_str : str
        e.g. "Cardiomegaly|Effusion" or "No Finding"

    Returns
    -------
    np.ndarray of shape (14,), dtype float32
        1.0 at position i if disease i is present, 0.0 otherwise.
        "No Finding" → all-zeros vector.
    """
    vec = np.zeros(NUM_FINDING_LABELS, dtype=np.float32)
    for label in str(finding_str).split("|"):
        label = label.strip()
        if label in _LABEL_TO_IDX:
            vec[_LABEL_TO_IDX[label]] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Helper: locate image file
# ---------------------------------------------------------------------------
def _find_image_path(image_name: str, images_root: str) -> Optional[str]:
    """Search for image under images_root/images/ subfolder (after extraction)."""
    # Standard extraction layout: images_root/images/<filename>
    candidate = os.path.join(images_root, "images", image_name)
    if os.path.isfile(candidate):
        return candidate
    # Fallback: directly under images_root
    candidate2 = os.path.join(images_root, image_name)
    if os.path.isfile(candidate2):
        return candidate2
    return None


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------
def get_train_transforms(
    image_size: int = 224,
    augment_strategy: str = "standard",
    grayscale: bool = True,
) -> T.Compose:
    """
    Build training transforms.

    augment_strategy options:
        "none"     – only resize + normalise (no augmentation)
        "standard" – flip + small rotation + colour jitter
        "strong"   – flip + larger rotation + affine + jitter
    """
    to_tensor_norm = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMG_MEAN * (1 if grayscale else 3),
                    std=IMG_STD   * (1 if grayscale else 3)),
    ]

    if augment_strategy == "none":
        transforms_list = [T.Grayscale(1)] + to_tensor_norm if grayscale else to_tensor_norm

    elif augment_strategy == "standard":
        aug = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        if grayscale:
            transforms_list = [T.Grayscale(1)] + aug + to_tensor_norm
        else:
            transforms_list = aug + to_tensor_norm

    elif augment_strategy == "strong":
        aug = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]
        if grayscale:
            transforms_list = [T.Grayscale(1)] + aug + to_tensor_norm
        else:
            transforms_list = aug + to_tensor_norm

    else:
        raise ValueError(f"Unknown augment_strategy: '{augment_strategy}'. "
                         f"Choose from 'none', 'standard', 'strong'.")

    return T.Compose(transforms_list)


def get_val_transforms(
    image_size: int = 224,
    grayscale: bool = True,
) -> T.Compose:
    """Deterministic transforms for validation / test (no augmentation)."""
    transforms_list = []
    if grayscale:
        transforms_list.append(T.Grayscale(1))
    transforms_list += [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMG_MEAN * (1 if grayscale else 3),
                    std=IMG_STD   * (1 if grayscale else 3)),
    ]
    return T.Compose(transforms_list)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ChestXrayAgeDataset(Dataset):
    """
    NIH ChestX-ray14 dataset for patient-age regression.

    Parameters
    ----------
    csv_path : str
        Path to Data_Entry_2017_v2020.csv.
    images_root : str
        Root directory that contains the extracted 'images/' subfolder.
    split_list_path : str
        Path to train_val_list.txt or test_list.txt (one filename per line).
    transform : optional callable
        Image transform pipeline.
    target_col : str
        Column name for the regression target (default: 'Patient Age').
    min_age : int
        Drop records with age below this threshold (removes age=0 artifacts).
    max_age : int
        Drop records with age above this threshold.
    """

    def __init__(
        self,
        csv_path: str,
        images_root: str,
        split_list_path: str,
        transform=None,
        target_col: str = "Patient Age",
        min_age: int = 1,
        max_age: int = 100,
    ):
        super().__init__()
        self.images_root = images_root
        self.transform = transform
        self.target_col = target_col

        # --- Load metadata CSV ---
        df_full = pd.read_csv(csv_path)

        # --- Load split list ---
        with open(split_list_path, "r") as f:
            split_names = set(line.strip() for line in f if line.strip())

        # --- Filter to this split ---
        df = df_full[df_full["Image Index"].isin(split_names)].copy()

        # --- Drop invalid ages ---
        df = df[(df[target_col] >= min_age) & (df[target_col] <= max_age)].copy()

        # --- Drop rows with NaN in key columns ---
        required_cols = ["Image Index", target_col, "Patient Sex", "View Position"]
        df = df.dropna(subset=required_cols).copy()

        # --- Encode metadata ---
        df["sex_enc"]  = df["Patient Sex"].map(SEX_MAP).fillna(0).astype(np.float32)
        df["view_enc"] = df["View Position"].map(VIEW_MAP).fillna(0).astype(np.float32)

        self.df = df.reset_index(drop=True)

        # --- Multi-hot encode Finding Labels (14-dim) ---
        # Pre-compute as a contiguous numpy matrix for fast __getitem__ lookup
        self._finding_matrix: np.ndarray = np.stack(
            self.df["Finding Labels"].apply(encode_finding_labels).values
        ).astype(np.float32)   # shape: (N, 14)

        # Pre-build image path cache for speed
        self._path_cache: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        image    : FloatTensor [C, H, W]
        metadata : FloatTensor [16]  (sex_enc, view_enc, + 14 finding multi-hot)
        target   : FloatTensor []    (scalar age)
        """
        row = self.df.iloc[idx]
        img_name = row["Image Index"]

        # --- Load image ---
        if img_name not in self._path_cache:
            self._path_cache[img_name] = _find_image_path(img_name, self.images_root)
        img_path = self._path_cache[img_name]

        if img_path is None:
            raise FileNotFoundError(
                f"Image '{img_name}' not found under '{self.images_root}'. "
                "Make sure all tar.gz archives have been extracted."
            )

        image = Image.open(img_path).convert("L")  # grayscale

        if self.transform is not None:
            image = self.transform(image)
        # If transform is None, return the PIL Image as-is so that
        # _TransformSubset in train.py can apply its own transform pipeline.

        # --- Metadata: [sex(1), view(1), finding_multihot(14)] = dim 16 ---
        demo_vec    = np.array([row["sex_enc"], row["view_enc"]], dtype=np.float32)
        finding_vec = self._finding_matrix[idx]                     # shape (14,)
        metadata    = torch.from_numpy(np.concatenate([demo_vec, finding_vec]))

        # --- Target ---
        target = torch.tensor(float(row[self.target_col]), dtype=torch.float32)

        return image, metadata, target

    # ------------------------------------------------------------------
    def get_age_stats(self) -> Dict[str, float]:
        """Return mean/std of target ages in this split (for normalisation reference)."""
        ages = self.df[self.target_col].values.astype(float)
        return {
            "mean": float(np.mean(ages)),
            "std":  float(np.std(ages)),
            "min":  float(np.min(ages)),
            "max":  float(np.max(ages)),
            "count": int(len(ages)),
        }

    # ------------------------------------------------------------------
    def get_finding_labels(self) -> pd.Series:
        """Return Finding Labels series for this split (for analysis only)."""
        return self.df["Finding Labels"]
