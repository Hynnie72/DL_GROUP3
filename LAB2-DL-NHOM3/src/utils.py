"""
utils.py - Các hàm tiện ích dùng chung cho Lab 2.

Bao gồm: seed, đếm tham số, freeze/unfreeze, device, checkpoint, logger, metrics resume.
"""

import os
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Đặt seed cho tính tái lập (reproducibility) trên CPU và GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Đảm bảo deterministic operations (có thể chậm hơn một chút)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Tự chọn GPU nếu có, fallback CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Sử dụng GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  Không có GPU. Đang chạy trên CPU (chậm hơn nhiều).")
    return device


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Đếm số tham số của model.

    Returns:
        Dict với 'total', 'trainable', 'frozen'
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def model_size_mb(model: nn.Module) -> float:
    """Tính kích thước model tính bằng MB (dựa trên số byte tham số float32)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


def print_parameter_summary(model: nn.Module, model_name: str = "") -> None:
    """In bảng tóm tắt số tham số của model."""
    stats = count_parameters(model)
    size = model_size_mb(model)
    header = f"=== Thông tin tham số: {model_name} ===" if model_name else "=== Thông tin tham số ==="
    print(header)
    print(f"  Tổng tham số:      {stats['total']:>12,}")
    print(f"  Tham số trainable: {stats['trainable']:>12,}")
    print(f"  Tham số đóng băng: {stats['frozen']:>12,}")
    print(f"  Kích thước model:  {size:>11.2f} MB")
    print("=" * len(header))


def freeze_all(model: nn.Module) -> None:
    """Đóng băng toàn bộ tham số (requires_grad = False)."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Mở khóa toàn bộ tham số (requires_grad = True)."""
    for param in model.parameters():
        param.requires_grad = True


def freeze_module(module: nn.Module) -> None:
    """Đóng băng một module cụ thể."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Mở khóa một module cụ thể."""
    for param in module.parameters():
        param.requires_grad = True


def set_bn_eval(model: nn.Module) -> None:
    """
    Đưa các lớp BatchNorm bị đóng băng về eval() mode.

    Lý do: Khi đóng băng backbone, các lớp BN không nên cập nhật running_mean/running_var.
    Nếu để ở train() mode, chúng vẫn cập nhật statistics dù weight không đổi,
    gây ra kết quả không ổn định khi batch size nhỏ.

    Cần gọi sau model.train() ở đầu mỗi epoch nếu có frozen BN.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Chỉ set eval() cho BN có tham số đóng băng
            if module.weight is not None and not module.weight.requires_grad:
                module.eval()


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Tạo logger với format chuẩn."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File handler (tùy chọn)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_checkpoint(
    state: Dict[str, Any],
    path: str,
    is_best: bool = False,
    best_path: Optional[str] = None,
) -> None:
    """
    Lưu checkpoint của model.

    Args:
        state: Dict chứa model state_dict, optimizer, epoch, metrics, class_to_idx, v.v.
        path: Đường dẫn lưu checkpoint hiện tại.
        is_best: Nếu True, cũng lưu vào best_path.
        best_path: Đường dẫn checkpoint tốt nhất.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best and best_path:
        import shutil
        shutil.copy2(path, best_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Nạp checkpoint vào model (và optimizer nếu có).

    Returns:
        Dict thông tin checkpoint (epoch, metrics, v.v.)
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """Lưu metrics ra file JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def load_metrics(path: str) -> Optional[Dict[str, Any]]:
    """Nạp metrics từ file JSON. Trả về None nếu file không tồn tại."""
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def experiment_exists(
    exp_name: str,
    outputs_dir: str,
    config_signature: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Kiểm tra thí nghiệm đã có kết quả chưa (cơ chế resume).

    Kiểm tra file metrics JSON có tồn tại và hợp lệ không.
    Nếu truyền config_signature, chỉ coi là đã xong khi metrics lưu đúng
    signature đó — tránh resume nhầm kết quả train bằng cấu hình khác.
    """
    metrics_path = Path(outputs_dir) / exp_name / "metrics.json"
    if not metrics_path.exists():
        return False
    metrics = load_metrics(str(metrics_path))
    if metrics is None or not metrics.get("completed", False):
        return False
    if config_signature is None:
        return True
    saved_sig = metrics.get("config_signature")
    if saved_sig is None:
        # metrics cũ (train bằng code trước) không có signature → coi là đã xong
        return True
    return saved_sig == config_signature


def get_peak_memory_mb() -> Dict[str, float]:
    """Lấy đỉnh RAM/VRAM đã dùng (tính bằng MB)."""
    result: Dict[str, float] = {}
    if torch.cuda.is_available():
        result["vram_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        result["vram_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 ** 2)
    else:
        result["vram_mb"] = 0.0
        result["vram_reserved_mb"] = 0.0

    try:
        import psutil
        process = psutil.Process(os.getpid())
        result["ram_mb"] = process.memory_info().rss / (1024 ** 2)
    except ImportError:
        result["ram_mb"] = 0.0

    return result


class Timer:
    """Context manager đo thời gian thực thi."""

    def __init__(self, description: str = ""):
        self.description = description
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.description:
            print(f"⏱  {self.description}: {self.elapsed:.2f}s")
