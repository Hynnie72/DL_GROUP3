"""NIH ChestX-ray14 dataset loader, augmentations, and synthetic data generator for regression."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from .config import RegressionConfig, AugmentationConfig


def get_transforms(
    aug_config: AugmentationConfig,
    image_size: Tuple[int, int] = (224, 224),
    is_train: bool = True,
    in_channels: int = 3,
) -> T.Compose:
    """Build Torchvision transform pipeline based on augmentation configuration.

    Parameters
    ----------
    aug_config : AugmentationConfig
        Augmentation parameters.
    image_size : Tuple[int, int]
        Target (Height, Width) for the model.
    is_train : bool
        Whether to apply data augmentation (train only).
    in_channels : int
        1 (grayscale) or 3 (RGB).
    """
    mean = [0.485, 0.456, 0.406] if in_channels == 3 else [0.5]
    std = [0.229, 0.224, 0.225] if in_channels == 3 else [0.25]

    transform_list: List[Any] = []

    if is_train and aug_config.type != "none":
        if aug_config.type == "standard":
            transform_list.extend([
                T.RandomHorizontalFlip(p=aug_config.horizontal_flip_prob),
                T.RandomRotation(degrees=aug_config.rotation_degrees),
            ])
        elif aug_config.type == "advanced":
            transform_list.extend([
                T.RandomHorizontalFlip(p=aug_config.horizontal_flip_prob),
                T.RandomRotation(degrees=aug_config.rotation_degrees),
                T.RandomAffine(
                    degrees=0,
                    translate=aug_config.random_affine_translate,
                    scale=(0.95, 1.05),
                ),
                T.ColorJitter(
                    brightness=aug_config.brightness_jitter,
                    contrast=aug_config.contrast_jitter,
                ),
            ])

    transform_list.extend([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(transform_list)


def generate_synthetic_nih_data(data_dir: str | Path, num_samples: int = 300, seed: int = 42) -> Path:
    """Generate synthetic NIH ChestX-ray14 data structure with simulated X-ray images
    and realistic metadata (Patient Age, Tumor Size, etc.).

    This ensures complete execution and testing even when the full 42GB NIH dataset
    has not yet been downloaded.
    """
    np.random.seed(seed)
    root = Path(data_dir)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = root / "Data_Entry_2017.csv"

    records = []
    findings_list = [
        "No Finding", "Infiltration", "Atelectasis", "Effusion",
        "Nodule", "Pneumothorax", "Mass", "Consolidation"
    ]

    for i in range(num_samples):
        img_id = f"000{i+1:05d}_000.png"
        img_file = images_dir / img_id

        # Generate realistic medical simulated chest X-ray if not exists
        if not img_file.exists():
            # Create a base grayscale canvas with lung-like elliptical shapes
            img = Image.new("L", (256, 256), color=25)
            draw = ImageDraw.Draw(img)

            # Mediastinum (spine/center column)
            draw.rectangle([115, 20, 140, 240], fill=160)

            # Left and right lungs (darker areas with ribs)
            draw.ellipse([30, 45, 110, 200], fill=65)
            draw.ellipse([145, 45, 225, 200], fill=65)

            # Heart shadow (left lower zone)
            draw.ellipse([110, 120, 175, 195], fill=180)

            # Simulate clavicles / ribs
            for y_rib in range(60, 200, 25):
                draw.arc([20, y_rib, 120, y_rib + 30], start=10, end=170, fill=130, width=3)
                draw.arc([135, y_rib, 235, y_rib + 30], start=10, end=170, fill=130, width=3)

            # Optional simulated nodule / lesion
            age = int(np.clip(np.random.normal(loc=52, scale=16), 18, 92))
            tumor_size_mm = round(float(np.clip(np.random.exponential(scale=12.0), 2.0, 65.0)), 2)

            if "Nodule" in findings_list or "Mass" in findings_list:
                cx, cy = np.random.randint(60, 200), np.random.randint(70, 180)
                draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=210)

            # Apply slight blur to simulate radiographic soft tissue
            img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

            # Add gaussian noise
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 5, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(img_file)
        else:
            age = int(np.clip(np.random.normal(loc=52, scale=16), 18, 92))
            tumor_size_mm = round(float(np.clip(np.random.exponential(scale=12.0), 2.0, 65.0)), 2)

        finding = np.random.choice(findings_list, p=[0.5, 0.1, 0.1, 0.1, 0.08, 0.04, 0.04, 0.04])
        gender = np.random.choice(["M", "F"])
        patient_id = np.random.randint(1, 1500)

        records.append({
            "Image Index": img_id,
            "Finding Labels": finding,
            "Follow-up #": np.random.randint(0, 5),
            "Patient ID": patient_id,
            "Patient Age": age,
            "Patient Gender": gender,
            "View Position": np.random.choice(["PA", "AP"]),
            "OriginalImage[Width": 1024,
            "Height]": 1024,
            "OriginalImagePixelSpacing[x": 0.143,
            "y]": 0.143,
            "Tumor Size": tumor_size_mm,
        })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    return csv_path


class NIHChestXrayDataset(Dataset):
    """PyTorch Dataset for NIH ChestX-ray14 continuous regression tasks."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path | str,
        target_col: str = "Patient Age",
        transform: Optional[T.Compose] = None,
        target_transform: Optional[Any] = None,
        in_channels: int = 3,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.target_col = target_col
        self.transform = transform
        self.target_transform = target_transform
        self.in_channels = in_channels

        if self.target_col not in self.df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in dataframe columns: {list(self.df.columns)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = self.images_dir / img_name

        if not img_path.exists():
            # If not found directly in images_dir, check recursive subfolders
            matches = list(self.images_dir.glob(f"**/{img_name}"))
            if matches:
                img_path = matches[0]
            else:
                raise FileNotFoundError(f"Image {img_name} not found under {self.images_dir}")

        image = Image.open(img_path)
        if self.in_channels == 3:
            image = image.convert("RGB")
        else:
            image = image.convert("L")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        target_val = float(row[self.target_col])
        if self.target_transform is not None:
            target_val = self.target_transform(target_val)

        target = torch.tensor(target_val, dtype=torch.float32)
        return image, target


class NIHDataModule:
    """Manages loading, splitting, transform injection, and DataLoader creation."""

    def __init__(self, config: RegressionConfig) -> None:
        self.config = config
        self.data_cfg = config.data
        self.aug_cfg = config.augmentation
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

        self.train_dataset: Optional[NIHChestXrayDataset] = None
        self.val_dataset: Optional[NIHChestXrayDataset] = None
        self.test_dataset: Optional[NIHChestXrayDataset] = None

    def setup(self) -> "NIHDataModule":
        """Prepare dataframe, apply split, setup datasets."""
        csv_file = Path(self.data_cfg.metadata_file)
        images_dir = Path(self.data_cfg.images_dir)

        # Auto-create synthetic data if requested or if files are missing
        if self.data_cfg.use_synthetic or not csv_file.exists() or not images_dir.exists():
            generate_synthetic_nih_data(
                data_dir=self.data_cfg.data_dir,
                num_samples=self.data_cfg.num_synthetic_samples,
                seed=self.config.seed,
            )

        df = pd.read_csv(csv_file)
        target = self.config.target_name

        if target not in df.columns:
            # Fallback if specific target is missing
            if "Patient Age" in df.columns:
                target = "Patient Age"
            elif "Tumor Size" in df.columns:
                target = "Tumor Size"
            else:
                raise ValueError(f"Neither '{self.config.target_name}', 'Patient Age' nor 'Tumor Size' found in CSV")
            self.config.target_name = target

        # Clean invalid values (e.g. age > 120 or negative)
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df.dropna(subset=[target, "Image Index"])
        if target == "Patient Age":
            df = df[(df[target] > 0) & (df[target] < 110)]

        # Calculate target stats for normalization if enabled
        self.target_mean = float(df[target].mean())
        self.target_std = float(df[target].std()) if float(df[target].std()) > 1e-5 else 1.0

        target_norm = None
        if self.data_cfg.normalize_target:
            target_norm = lambda y: (y - self.target_mean) / self.target_std

        # Train / Val / Test split
        val_test_ratio = self.data_cfg.validation_size + self.data_cfg.test_size
        train_df, temp_df = train_test_split(
            df,
            test_size=val_test_ratio,
            random_state=self.config.seed,
        )

        val_rel_ratio = self.data_cfg.validation_size / val_test_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_rel_ratio),
            random_state=self.config.seed,
        )

        train_transform = get_transforms(
            self.aug_cfg,
            image_size=self.data_cfg.image_size,
            is_train=True,
            in_channels=self.data_cfg.in_channels,
        )
        eval_transform = get_transforms(
            self.aug_cfg,
            image_size=self.data_cfg.image_size,
            is_train=False,
            in_channels=self.data_cfg.in_channels,
        )

        self.train_dataset = NIHChestXrayDataset(
            train_df, images_dir, target, transform=train_transform,
            target_transform=target_norm, in_channels=self.data_cfg.in_channels
        )
        self.val_dataset = NIHChestXrayDataset(
            val_df, images_dir, target, transform=eval_transform,
            target_transform=target_norm, in_channels=self.data_cfg.in_channels
        )
        self.test_dataset = NIHChestXrayDataset(
            test_df, images_dir, target, transform=eval_transform,
            target_transform=target_norm, in_channels=self.data_cfg.in_channels
        )

        return self

    def denormalize(self, val: np.ndarray | torch.Tensor | float) -> np.ndarray | float:
        """Denormalize scaled target predictions back to original physical unit."""
        if not self.data_cfg.normalize_target:
            return val
        if isinstance(val, torch.Tensor):
            return val * self.target_std + self.target_mean
        return np.array(val) * self.target_std + self.target_mean

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Return PyTorch DataLoader instances for train, val, and test."""
        if self.train_dataset is None:
            self.setup()

        batch_size = self.config.training.batch_size
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 0 is safe on Windows
            drop_last=False,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        return train_loader, val_loader, test_loader
