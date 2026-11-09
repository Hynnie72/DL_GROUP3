"""
smoke_test.py - Kiểm thử nhanh (smoke test) cho Lab 2.

Kiểm tra với dữ liệu giả (không cần dataset thật, không cần Internet):
1. Build được 3 model (pretrained=False)
2. Output shape đúng [batch, 6]
3. Forward + backward không lỗi
4. Loss hữu hạn (không NaN/Inf)
5. Khi freeze thì chỉ layer trainable có gradient
6. Mỗi freeze_mode cho đúng số tham số trainable
7. TensorBoard tạo được file event

Mục tiêu: chạy dưới 2 phút trên CPU, không cần GPU.
"""

import sys
import os
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# Thêm project root vào sys.path để import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model, list_models, VALID_FREEZE_MODES
from src.utils import count_parameters, set_seed


# ──────────────────────────────────────────────
# Cấu hình smoke test
# ──────────────────────────────────────────────

NUM_CLASSES = 6
BATCH_SIZE  = 2
IMAGE_SIZE  = 224

# Số tham số trainable kỳ vọng (xấp xỉ) cho từng model x freeze_mode
# Dùng để kiểm tra "đúng số tham số trainable kỳ vọng"
# Giá trị chính xác sẽ được in ra khi chạy; đây là threshold tối thiểu
MIN_TRAINABLE: Dict[str, Dict[str, int]] = {
    "resnet18": {
        "freeze_backbone": 3_000,       # chỉ fc: 512*6 + 6 = 3078
        "finetune_last":   2_000_000,   # layer4 + fc: ~2.1M
        "full_finetune":   11_000_000,  # tất cả: ~11.7M
    },
    "vgg16": {
        "freeze_backbone": 24_000,      # classifier toàn bộ: ~4096*6 + 6 = 24582
        "finetune_last":   100_000_000, # block5 + classifier: ~100M+
        "full_finetune":   130_000_000, # tất cả: ~138M
    },
    "densenet121": {
        "freeze_backbone": 6_000,       # chỉ classifier: 1024*6 + 6 = 6150
        "finetune_last":   2_100_000,   # denseblock4 + norm5 + classifier: 2,166,278
        "full_finetune":   6_900_000,   # tất cả: 6,960,006
    },
}


# ──────────────────────────────────────────────
# Hàm tiện ích
# ──────────────────────────────────────────────

def make_fake_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """Tạo batch dữ liệu giả để test (không cần dataset thật)."""
    images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return images, labels


def print_result(test_name: str, passed: bool, detail: str = "") -> None:
    """In kết quả một test."""
    icon = "✅" if passed else "❌"
    msg  = f"{icon} {test_name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ──────────────────────────────────────────────
# Các bài test
# ──────────────────────────────────────────────

def test_build_model(model_name: str, freeze_mode: str) -> Tuple[bool, str]:
    """Test 1: Build model thành công (không cần Internet, pretrained=False)."""
    try:
        model = build_model(model_name, NUM_CLASSES, freeze_mode, pretrained=False)
        return True, f"build_model OK"
    except Exception as e:
        return False, f"FAIL: {e}"


def test_output_shape(model: nn.Module, model_name: str, freeze_mode: str) -> Tuple[bool, str]:
    """Test 2: Output shape = [batch, NUM_CLASSES]."""
    model.eval()
    images, _ = make_fake_batch()
    with torch.no_grad():
        outputs = model(images)
    expected = (BATCH_SIZE, NUM_CLASSES)
    ok = tuple(outputs.shape) == expected
    return ok, f"shape={tuple(outputs.shape)} (expected {expected})"


def test_forward_backward(model: nn.Module, model_name: str, freeze_mode: str) -> Tuple[bool, str]:
    """Test 3: Forward + backward không lỗi."""
    try:
        model.train()
        criterion = nn.CrossEntropyLoss()
        images, labels = make_fake_batch()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        return True, f"loss={loss.item():.4f}"
    except Exception as e:
        return False, f"FAIL: {e}"


def test_loss_finite(model: nn.Module, model_name: str, freeze_mode: str) -> Tuple[bool, str]:
    """Test 4: Loss hữu hạn (không NaN / Inf)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    images, labels = make_fake_batch()
    with torch.no_grad():
        outputs = model(images)
    loss = criterion(outputs, labels)
    ok = torch.isfinite(loss).item()
    return ok, f"loss={loss.item():.4f} finite={ok}"


def test_gradients(model: nn.Module, model_name: str, freeze_mode: str) -> Tuple[bool, str]:
    """
    Test 5: Khi freeze → chỉ layer trainable có gradient.
    Kiểm tra: tham số đóng băng có grad=None sau backward.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    images, labels = make_fake_batch()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    frozen_with_grad   = 0
    unfrozen_with_grad = 0
    unfrozen_no_grad   = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                unfrozen_with_grad += 1
            else:
                unfrozen_no_grad += 1
        else:
            if param.grad is not None:
                frozen_with_grad += 1

    ok = frozen_with_grad == 0 and unfrozen_with_grad > 0 and unfrozen_no_grad == 0
    detail = (
        f"frozen_with_grad={frozen_with_grad} (phải=0), "
        f"unfrozen_with_grad={unfrozen_with_grad} (phải>0), "
        f"unfrozen_no_grad={unfrozen_no_grad} (phải=0)"
    )
    return ok, detail


def test_trainable_params(model: nn.Module, model_name: str, freeze_mode: str) -> Tuple[bool, str]:
    """
    Test 6: Số tham số trainable >= ngưỡng tối thiểu kỳ vọng
    và nhỏ hơn full_finetune (trừ khi là full_finetune).
    """
    stats = count_parameters(model)
    trainable = stats["trainable"]
    total     = stats["total"]
    min_expected = MIN_TRAINABLE.get(model_name, {}).get(freeze_mode, 0)

    ok = trainable >= min_expected
    if freeze_mode != "full_finetune":
        ok = ok and trainable < total  # phải có tham số bị đóng băng

    detail = (
        f"trainable={trainable:,} / total={total:,} "
        f"(min_expected={min_expected:,})"
    )
    return ok, detail


def test_tensorboard(model_name: str, freeze_mode: str, tb_dir: str) -> Tuple[bool, str]:
    """Test 7: TensorBoard tạo được file event."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = str(Path(tb_dir) / f"{model_name}_{freeze_mode}")
        writer  = SummaryWriter(log_dir=log_dir)
        writer.add_scalar("test/loss", 1.0, 0)
        writer.close()

        # Kiểm tra file event đã được tạo
        event_files = list(Path(log_dir).glob("events.out.tfevents.*"))
        ok = len(event_files) > 0
        return ok, f"{len(event_files)} event file(s) tại {log_dir}"
    except Exception as e:
        return False, f"FAIL: {e}"


# ──────────────────────────────────────────────
# Chạy tất cả test
# ──────────────────────────────────────────────

def run_smoke_tests() -> Dict:
    """Chạy toàn bộ smoke test và trả về tóm tắt."""
    set_seed(42)
    start_time = time.time()

    print("=" * 70)
    print("🚀 SMOKE TEST - Lab 2: Pre-trained CNN")
    print("=" * 70)
    print(f"  Thiết bị: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"  NUM_CLASSES={NUM_CLASSES}, BATCH_SIZE={BATCH_SIZE}, IMAGE_SIZE={IMAGE_SIZE}")
    print()

    results: Dict = {"passed": 0, "failed": 0, "details": []}

    # Thư mục tạm cho TensorBoard test
    tb_dir = tempfile.mkdtemp(prefix="lab2_smoke_tb_")

    models_to_test = list_models()
    freeze_modes   = sorted(VALID_FREEZE_MODES)

    for model_name in models_to_test:
        for freeze_mode in freeze_modes:
            tag = f"[{model_name} | {freeze_mode}]"
            print(f"\n{'─'*60}")
            print(f"  {tag}")
            print(f"{'─'*60}")

            # Test 1: Build
            ok, detail = test_build_model(model_name, freeze_mode)
            print_result("T1 Build model", ok, detail)
            results["details"].append({"test": "build", "model": model_name, "mode": freeze_mode, "ok": ok})
            if not ok:
                results["failed"] += 1
                print(f"  ⚠️  Bỏ qua các test còn lại cho {tag}")
                continue
            results["passed"] += 1

            # Tạo model (pretrained=False để không cần Internet)
            model = build_model(model_name, NUM_CLASSES, freeze_mode, pretrained=False)

            # Test 2: Output shape
            ok, detail = test_output_shape(model, model_name, freeze_mode)
            print_result("T2 Output shape", ok, detail)
            results["passed" if ok else "failed"] += 1

            # Test 3: Forward + backward
            ok, detail = test_forward_backward(model, model_name, freeze_mode)
            print_result("T3 Forward+backward", ok, detail)
            results["passed" if ok else "failed"] += 1

            # Test 4: Loss finite
            ok, detail = test_loss_finite(model, model_name, freeze_mode)
            print_result("T4 Loss finite", ok, detail)
            results["passed" if ok else "failed"] += 1

            # Test 5: Gradients
            ok, detail = test_gradients(model, model_name, freeze_mode)
            print_result("T5 Gradients freeze", ok, detail)
            results["passed" if ok else "failed"] += 1

            # Test 6: Trainable params
            ok, detail = test_trainable_params(model, model_name, freeze_mode)
            print_result("T6 Trainable params", ok, detail)
            results["passed" if ok else "failed"] += 1

            # Test 7: TensorBoard
            ok, detail = test_tensorboard(model_name, freeze_mode, tb_dir)
            print_result("T7 TensorBoard event", ok, detail)
            results["passed" if ok else "failed"] += 1

    # Dọn dẹp thư mục TensorBoard tạm
    shutil.rmtree(tb_dir, ignore_errors=True)

    elapsed = time.time() - start_time
    total   = results["passed"] + results["failed"]

    print(f"\n{'='*70}")
    print(f"📋 KẾT QUẢ SMOKE TEST")
    print(f"{'='*70}")
    print(f"  Đạt:   {results['passed']:3d}/{total}")
    print(f"  Thất bại: {results['failed']:3d}/{total}")
    print(f"  Thời gian: {elapsed:.1f}s")

    if elapsed > 120:
        print(f"  ⚠️  Cảnh báo: chạy quá 2 phút ({elapsed:.0f}s)")
    else:
        print(f"  ✅ Trong giới hạn 2 phút")

    if results["failed"] == 0:
        print(f"\n  🎉 Tất cả test đạt! Môi trường sẵn sàng.")
    else:
        print(f"\n  ❌ Có {results['failed']} test thất bại. Kiểm tra log ở trên.")

    print(f"{'='*70}")

    results["elapsed_s"] = elapsed
    return results


if __name__ == "__main__":
    result = run_smoke_tests()
    sys.exit(0 if result["failed"] == 0 else 1)
