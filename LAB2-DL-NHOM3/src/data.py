"""
data.py - Chuẩn bị dataset Intel Image Classification.

Bao gồm: tìm dataset root, tạo Dataset, tách train/val, tạo DataLoader.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError


# ──────────────────────────────────────────────
# 1. Tìm dataset root
# ──────────────────────────────────────────────

def find_dataset_root(local_fallback: Optional[str] = None) -> str:
    """
    Tự động tìm thư mục gốc của Intel Image Classification dataset.

    Chiến lược tìm kiếm:
    1. Tìm trong /kaggle/input/ bằng os.walk (phù hợp cấu trúc mới của Kaggle)
    2. Fallback sang đường dẫn local từ config

    Returns:
        Đường dẫn đến thư mục chứa seg_train/, seg_test/, seg_pred/

    Raises:
        FileNotFoundError: Nếu không tìm thấy dataset ở bất kỳ đâu.
    """
    # Thử tìm trên Kaggle
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, dirs, _ in os.walk(str(kaggle_input)):
            if "seg_train" in dirs and "seg_test" in dirs:
                print(f"✅ Tìm thấy dataset tại: {root}")
                return root

    # Fallback sang local path từ config
    if local_fallback and Path(local_fallback).exists():
        local = Path(local_fallback)
        if (local / "seg_train").exists():
            print(f"✅ Dùng dataset local: {local_fallback}")
            return local_fallback

    raise FileNotFoundError(
        "❌ Không tìm thấy dataset Intel Image Classification.\n"
        "   Trên Kaggle: Add Input → Intel Image Classification (puneet6060)\n"
        "   Local: Tải dataset từ Kaggle và cập nhật 'local_root' trong configs/base.yaml"
    )


# ──────────────────────────────────────────────
# 2. Kiểm tra và thống kê dataset
# ──────────────────────────────────────────────

def check_images(folder: str) -> Tuple[int, int, List[str]]:
    """
    Kiểm tra ảnh trong thư mục: đếm tổng, đếm lỗi, liệt kê ảnh lỗi.

    Returns:
        (total, error_count, error_files)
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    total = 0
    errors: List[str] = []

    for root, _, files in os.walk(folder):
        for fname in files:
            if Path(fname).suffix.lower() in extensions:
                total += 1
                fpath = os.path.join(root, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                except Exception:
                    errors.append(fpath)

    return total, len(errors), errors


def count_samples_per_class(folder: str) -> Dict[str, int]:
    """Đếm số ảnh mỗi lớp trong thư mục có cấu trúc <folder>/<class>/*.jpg."""
    counts: Dict[str, int] = {}
    folder_path = Path(folder)

    if not folder_path.exists():
        return counts

    for class_dir in sorted(folder_path.iterdir()):
        if class_dir.is_dir():
            count = sum(
                1 for f in class_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            counts[class_dir.name] = count

    return counts


def resolve_split_path(dataset_root: str, split_name: str) -> str:
    """
    Tìm đường dẫn thực tế cho một split (hỗ trợ cấu trúc lồng 2 cấp cùng tên).

    Ví dụ: seg_train/seg_train/<class>/ hoặc seg_train/<class>/
    """
    base = Path(dataset_root)
    # Thử lồng 2 cấp (cấu trúc Kaggle mới)
    nested = base / split_name / split_name
    single = base / split_name
    if nested.exists():
        return str(nested)
    if single.exists():
        return str(single)
    raise FileNotFoundError(f"Không tìm thấy thư mục split '{split_name}' trong {dataset_root}")


def print_dataset_stats(dataset_root: str) -> Dict:
    """In thống kê đầy đủ về dataset và trả về dict kết quả."""
    print("\n" + "=" * 60)
    print("📊 THỐNG KÊ DATASET Intel Image Classification")
    print("=" * 60)

    stats: Dict = {}

    for split_name in ["seg_train", "seg_test"]:
        try:
            folder = resolve_split_path(dataset_root, split_name)
        except FileNotFoundError:
            print(f"\n  [{split_name}] ❌ Không tìm thấy thư mục")
            continue

        counts = count_samples_per_class(folder)
        total = sum(counts.values())
        label = "train+val" if split_name == "seg_train" else "test"
        print(f"\n  [{label}] Tổng: {total:,} ảnh")
        print(f"  {'Lớp':<15} {'Số ảnh':>8}")
        print(f"  {'-' * 24}")
        for cls, cnt in sorted(counts.items()):
            print(f"  {cls:<15} {cnt:>8,}")
        if counts:
            max_c, min_c = max(counts.values()), min(counts.values())
            ratio = max_c / min_c if min_c > 0 else float("inf")
            print(f"  → Tỉ lệ mất cân bằng: {ratio:.2f}x")
        stats[split_name] = {"total": total, "per_class": counts}

    # seg_pred (không nhãn)
    try:
        pred_path = resolve_split_path(dataset_root, "seg_pred")
        n_pred = sum(
            1 for f in Path(pred_path).iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"\n  [seg_pred] {n_pred:,} ảnh (không nhãn, chỉ dùng demo)")
        stats["seg_pred"] = n_pred
    except FileNotFoundError:
        print("\n  [seg_pred] ❌ Không tìm thấy")

    print("\n" + "=" * 60)
    return stats


# ──────────────────────────────────────────────
# 3. Transforms
# ──────────────────────────────────────────────

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    Tạo transforms theo split.

    - Train: RandomResizedCrop + RandomHorizontalFlip + RandomRotation(10)
             + ColorJitter + RandomErasing (data augmentation)
    - Val/Test: Resize(256) + CenterCrop(224) (deterministic)

    RandomErasing đặt sau Normalize (theo khuyến nghị của torchvision) và chỉ
    áp dụng cho train; val/test giữ nguyên để đánh giá ổn định.

    Lưu ý: Ảnh gốc 150x150, resize lên 224x224 là phóng to (~1.5x).
    Điều này có thể làm ảnh bị mờ nhẹ, nhưng là cách chuẩn để phù hợp
    với kiến trúc ImageNet. Thực tế cho thấy mạng pre-trained vẫn học tốt.

    Args:
        split: 'train', 'val', hoặc 'test'
        image_size: Kích thước đầu vào (mặc định 224)
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), value="random"),
        ])
    else:  # val hoặc test
        resize_to = int(image_size * 256 / 224)  # ≈256 nếu image_size=224
        return transforms.Compose([
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


# ──────────────────────────────────────────────
# 4. Dataset class cho seg_pred
# ──────────────────────────────────────────────

class PredDataset(Dataset):
    """
    Dataset cho seg_pred (không có nhãn).
    Chỉ dùng để demo dự đoán.
    """

    def __init__(self, root: str, transform=None, max_samples: Optional[int] = None):
        extensions = {".jpg", ".jpeg", ".png"}
        self.files = sorted([
            str(f) for f in Path(root).iterdir()
            if f.suffix.lower() in extensions
        ])
        if max_samples:
            self.files = self.files[:max_samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            img = self.transform(img)

        return img, path  # Trả về ảnh và đường dẫn (thay cho nhãn)


# ──────────────────────────────────────────────
# 5. Tách train/val và tạo DataLoader
# ──────────────────────────────────────────────

def get_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
    val_ratio: float = 0.15,
    seed: int = 42,
    smoke_test: bool = False,
    samples_per_class: int = 8,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Tạo DataLoader cho train, val, test.

    - train + val: từ seg_train (stratified split 85/15, seed 42)
    - test: nguyên seg_test (không dùng để chọn model)
    - seg_pred: KHÔNG được tạo ở đây, chỉ dùng trong demo

    Args:
        dataset_root: Thư mục gốc của dataset (chứa seg_train, seg_test, seg_pred).
        batch_size: Batch size.
        num_workers: Số worker cho DataLoader.
        image_size: Kích thước ảnh đầu vào.
        val_ratio: Tỉ lệ validation (mặc định 0.15 = 15%).
        seed: Random seed.
        smoke_test: Nếu True, chỉ lấy samples_per_class ảnh mỗi lớp.
        samples_per_class: Số ảnh mỗi lớp cho smoke test.

    Returns:
        (train_loader, val_loader, test_loader, class_to_idx)
    """
    train_root = resolve_split_path(dataset_root, "seg_train")
    test_root  = resolve_split_path(dataset_root, "seg_test")

    # Nạp full train dataset để lấy labels (dùng transform train tạm)
    full_train_dataset = datasets.ImageFolder(
        root=train_root,
        transform=get_transforms("train", image_size),
    )
    class_to_idx: Dict[str, int] = full_train_dataset.class_to_idx

    targets = [s[1] for s in full_train_dataset.samples]
    indices = list(range(len(targets)))

    # Smoke test: lấy ít ảnh mỗi lớp
    if smoke_test:
        from collections import defaultdict
        class_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)

        smoke_indices: List[int] = []
        for label_idx_list in class_indices.values():
            smoke_indices.extend(label_idx_list[:samples_per_class])
        indices = smoke_indices
        targets = [targets[i] for i in indices]

    # Stratified split
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=targets,
    )

    # Tạo 2 dataset với transform tương ứng
    train_dataset_full = datasets.ImageFolder(
        root=train_root, transform=get_transforms("train", image_size)
    )
    val_dataset_full = datasets.ImageFolder(
        root=train_root, transform=get_transforms("val", image_size)
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset   = Subset(val_dataset_full,   val_indices)

    # Test dataset
    test_dataset = datasets.ImageFolder(
        root=test_root, transform=get_transforms("test", image_size)
    )

    # DataLoader settings
    pin_mem = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_mem,
    }

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    print(f"\n📦 DataLoader đã sẵn sàng:")
    print(f"  Train: {len(train_dataset):,} ảnh, {len(train_loader)} batch")
    print(f"  Val:   {len(val_dataset):,} ảnh, {len(val_loader)} batch")
    print(f"  Test:  {len(test_dataset):,} ảnh, {len(test_loader)} batch")
    print(f"  Batch size: {batch_size}, Workers: {num_workers}")
    print(f"  class_to_idx: {class_to_idx}")

    return train_loader, val_loader, test_loader, class_to_idx


def get_pred_loader(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
    max_samples: Optional[int] = 20,
) -> DataLoader:
    """DataLoader cho seg_pred (không nhãn, chỉ dùng demo)."""
    pred_root = resolve_split_path(dataset_root, "seg_pred")
    pred_dataset = PredDataset(
        root=pred_root,
        transform=get_transforms("test", image_size),
        max_samples=max_samples,
    )
    return DataLoader(
        pred_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
