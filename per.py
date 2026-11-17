
"""per.py — Benchmark inference trên MỘT ảnh duy nhất cho cả 4 mô hình.

Với mỗi model, đo:
    • Accuracy             — đúng / sai so với nhãn thật (100% hoặc 0%)
    • Time processing      — thời gian forward pass (ms)
    • VRAM Peak            — VRAM GPU tối đa được sử dụng (MB)
    • VRAM Allocated       — VRAM đang được cấp phát sau inference (MB)

Kết quả được in console và vẽ bảng bằng Matplotlib rồi lưu PNG.
Thiết bị: CUDA (tự động fallback CPU nếu không có GPU).
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

# ─── Cấu hình ─────────────────────────────────────────────────────────────

ALL_MODELS = ["lenet", "resnet18", "alexnet", "vgg16"]

CHECKPOINT_DIR = Path(__file__).resolve().parent / "outputs" / "metrics"
DATASET_ROOT = Path(__file__).resolve().parent / "datastet" / "raw"

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "figures"
    / "performance_table.png"
)

RANDOM_SEED = 42


# ─── Import nội bộ ────────────────────────────────────────────────────────

from data import CatsDogsDataset, get_transforms, CLASS_TO_IDX
from evaluate import predict_one_img
from utils import get_device, device_info


# ─── Load model ───────────────────────────────────────────────────────────

def load_model(name: str, device: torch.device) -> torch.nn.Module:
    """Load model từ checkpoint."""

    ckpt = CHECKPOINT_DIR / f"{name}_best.pth"

    if not ckpt.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {ckpt}"
        )

    if name == "lenet":
        from models.lenet import lenet
        model = lenet(num_classes=2)

    elif name == "resnet18":
        from models.resnet18 import resnet18
        model = resnet18(num_classes=2)

    elif name == "alexnet":
        from models.alexnet import alexnet
        model = alexnet(num_classes=2)

    elif name == "vgg16":
        from models.vgg import vgg16
        model = vgg16(num_classes=2)

    else:
        raise ValueError(f"Không hỗ trợ model: {name}")

    state = torch.load(
        ckpt,
        map_location=device,
        weights_only=False
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model


# ─── Benchmark một model ──────────────────────────────────────────────────

def benchmark_model(
    name: str,
    image_path: Path,
    true_class: str,
    device: torch.device,
) -> dict[str, Any]:

    """Chạy inference và đo Accuracy / Time / VRAM."""

    transform = get_transforms(
        name,
        split="test",
        augment=False
    )

    model = load_model(name, device)

    # ── Warm-up CUDA ──────────────────────────────────────────────────────

    if device.type == "cuda":

        import PIL.Image

        with PIL.Image.open(image_path) as img:
            tensor = transform(
                img.convert("RGB")
            ).unsqueeze(0).to(device)

        with torch.inference_mode():
            _ = model(tensor)

        torch.cuda.synchronize()

        del tensor

        # Xóa bộ nhớ tạm sau warm-up
        torch.cuda.empty_cache()

        # Reset peak để chỉ đo inference chính thức
        torch.cuda.reset_peak_memory_stats(device)

        # Ghi nhận VRAM trước inference
        vram_before = torch.cuda.memory_allocated(device)

    else:
        vram_before = 0

    # ── Inference + đo thời gian ──────────────────────────────────────────

    if device.type == "cuda":

        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        result = predict_one_img(
            model,
            image_path,
            transform,
            device
        )

        end_event.record()

        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)

        # Peak VRAM trong inference
        vram_peak = torch.cuda.max_memory_allocated(device)

        # VRAM hiện đang được cấp phát
        vram_after = torch.cuda.memory_allocated(device)

        # VRAM tăng thêm tại thời điểm cuối
        vram_delta = max(
            0,
            vram_after - vram_before
        )

        # Chuyển byte sang MB
        vram_peak_mb = vram_peak / (1024 ** 2)
        vram_allocated_mb = vram_after / (1024 ** 2)
        vram_delta_mb = vram_delta / (1024 ** 2)

    else:

        t0 = time.perf_counter()

        result = predict_one_img(
            model,
            image_path,
            transform,
            device
        )

        elapsed_ms = (
            time.perf_counter() - t0
        ) * 1000

        vram_peak_mb = 0.0
        vram_allocated_mb = 0.0
        vram_delta_mb = 0.0

    # ── Accuracy ──────────────────────────────────────────────────────────

    is_correct = (
        result["predicted_class"] == true_class
    )

    accuracy = 100.0 if is_correct else 0.0

    # ── Giải phóng model ──────────────────────────────────────────────────

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": name.upper(),

        "predicted": result["predicted_class"],

        "confidence": result["confidence"],

        "accuracy": accuracy,

        "time_ms": elapsed_ms,

        "vram_peak_mb": vram_peak_mb,

        "vram_allocated_mb": vram_allocated_mb,

        "vram_delta_mb": vram_delta_mb,

        "correct": is_correct,
    }


# ─── Vẽ bảng Matplotlib ───────────────────────────────────────────────────

def draw_table(
    rows: list[dict],
    true_class: str,
    image_name: str,
    device_str: str,
) -> None:

    """Vẽ bảng benchmark và lưu thành PNG."""

    col_labels = [
        "Model",
        "Predicted",
        "Confidence (%)",
        "Accuracy (%)",
        "Time (ms)",
        "VRAM Peak (MB)",
        "VRAM Δ (MB)",
    ]

    cell_data = []
    row_colors = []

    GREEN = "#d4edda"
    RED = "#f8d7da"
    HEADER = "#343a40"

    for r in rows:

        cell_data.append([
            r["model"],
            r["predicted"],
            f'{r["confidence"]:.2f}',
            f'{r["accuracy"]:.1f}',
            f'{r["time_ms"]:.3f}',
            f'{r["vram_peak_mb"]:.2f}',
            f'{r["vram_delta_mb"]:.2f}',
        ])

        color = GREEN if r["correct"] else RED

        row_colors.append(
            [color] * len(col_labels)
        )

    fig, ax = plt.subplots(
        figsize=(14, 3.5)
    )

    ax.set_axis_off()

    title = (
        "Inference Performance — Single Image Benchmark\n"
        f"Image: {image_name}   |   "
        f"Ground Truth: {true_class.upper()}   |   "
        f"Device: {device_str}"
    )

    fig.suptitle(
        title,
        fontsize=11,
        fontweight="bold",
        y=1.02
    )

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)

    # Header
    for col_idx in range(len(col_labels)):

        cell = tbl[0, col_idx]

        cell.set_facecolor(HEADER)

        cell.set_text_props(
            color="white",
            fontweight="bold"
        )

    # Legend
    legend_text = (
        "GREEN = Correct Prediction     "
        "RED = Incorrect Prediction"
    )

    fig.text(
        0.5,
        -0.04,
        legend_text,
        ha="center",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#e9ecef",
            edgecolor="#adb5bd"
        )
    )

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\n  Bảng đã lưu → {OUTPUT_PATH}"
    )


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> None:

    device = get_device()

    dev_str = device_info(device)

    IDX2CLASS = {
        v: k for k, v in CLASS_TO_IDX.items()
    }

    print("=" * 85)

    print(f"  Device  : {dev_str}")

    print(
        f"  Models  : {', '.join(m.upper() for m in ALL_MODELS)}"
    )

    # ── Chọn một ảnh ngẫu nhiên từ tập test ────────────────────────────────

    dataset = CatsDogsDataset(
        root=DATASET_ROOT,
        split="test",
        transform=None
    )

    if len(dataset) == 0:
        raise RuntimeError("Test dataset không có ảnh.")

    random.seed(RANDOM_SEED)

    idx = random.randint(
        0,
        len(dataset) - 1
    )

    image_path, true_label = dataset.samples[idx]

    true_class = IDX2CLASS[true_label]

    print(f"  Image   : {image_path.name}")

    print(f"  Label   : {true_class.upper()}")

    if device.type == "cuda":

        print(
            f"  GPU     : {torch.cuda.get_device_name(device)}"
        )

    else:

        print("  GPU     : Not available (CPU mode)")

    print("=" * 85)

    # ── Benchmark từng model ──────────────────────────────────────────────

    results: list[dict] = []

    for name in ALL_MODELS:

        print(
            f"  → Benchmarking {name.upper()} ...",
            end=" ",
            flush=True
        )

        try:

            r = benchmark_model(
                name,
                image_path,
                true_class,
                device
            )

            results.append(r)

            verdict = "CORRECT" if r["correct"] else "INCORRECT"

            print(
                f"{verdict} | "
                f"Time: {r['time_ms']:.3f} ms | "
                f"VRAM Peak: {r['vram_peak_mb']:.2f} MB | "
                f"VRAM Δ: {r['vram_delta_mb']:.2f} MB | "
                f"Confidence: {r['confidence']:.2f}%"
            )

        except FileNotFoundError as e:

            print(f"[SKIP] {e}")

    if not results:

        print(
            "\n  Không có kết quả nào để hiển thị."
        )

        return

    # ── Vẽ bảng ───────────────────────────────────────────────────────────

    draw_table(
        results,
        true_class,
        image_path.name,
        dev_str
    )

    # ── Tóm tắt ───────────────────────────────────────────────────────────

    print("\n" + "=" * 85)

    print("  BENCHMARK SUMMARY")

    print("=" * 85)

    for r in results:

        print(
            f"  {r['model']:<10} "
            f"Accuracy={r['accuracy']:5.1f}% | "
            f"Time={r['time_ms']:8.3f} ms | "
            f"VRAM Peak={r['vram_peak_mb']:8.2f} MB | "
            f"VRAM Δ={r['vram_delta_mb']:8.2f} MB"
        )

    print("=" * 85)


if __name__ == "__main__":
    main()
