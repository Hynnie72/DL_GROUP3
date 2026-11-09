"""
evaluate.py - Đánh giá model trên TEST set.

Tính: accuracy, precision/recall/F1, confusion matrix, thời gian inference.
Phân tích lỗi: cặp lớp hay nhầm, ảnh dự đoán sai.
Xuất: CSV/JSON tổng hợp, ảnh confusion matrix.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (cho Kaggle và server)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.tensorboard import SummaryWriter

from src.utils import (
    get_device, save_metrics, load_metrics, model_size_mb, count_parameters,
)


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Đánh giá model trên test set, trả về dict kết quả đầy đủ.

    Args:
        model: Model đã nạp checkpoint tốt nhất.
        test_loader: DataLoader cho test set.
        class_names: Danh sách tên lớp (theo thứ tự index).
        device: Thiết bị tính toán.

    Returns:
        Dict chứa accuracy, F1, per-class metrics, confusion matrix, inference time, v.v.
    """
    model.eval()
    all_preds:  List[int]       = []
    all_labels: List[int]       = []
    all_probs:  List[List[float]] = []

    total_images = 0
    inference_start = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
            total_images += images.size(0)

    inference_time = time.time() - inference_start
    ms_per_image   = (inference_time / max(total_images, 1)) * 1000
    images_per_sec = total_images / max(inference_time, 1e-9)

    np_preds  = np.array(all_preds)
    np_labels = np.array(all_labels)
    np_probs  = np.array(all_probs)

    # Metrics tổng hợp
    accuracy = float(accuracy_score(np_labels, np_preds))
    precision, recall, f1, support = precision_recall_fscore_support(
        np_labels, np_preds,
        average=None,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        np_labels, np_preds, average="macro", zero_division=0,
    )

    report = classification_report(
        np_labels, np_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(np_labels, np_preds)

    # Per-class metrics
    per_class: Dict[str, Dict] = {}
    for i, cls in enumerate(class_names):
        per_class[cls] = {
            "precision": float(precision[i]),
            "recall":    float(recall[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i]),
        }

    # Phân tích lỗi
    wrong_indices = np.where(np_preds != np_labels)[0]
    error_pairs: Dict[str, int] = {}
    for idx in wrong_indices:
        true_cls = class_names[np_labels[idx]]
        pred_cls = class_names[np_preds[idx]]
        key = f"{true_cls} → {pred_cls}"
        error_pairs[key] = error_pairs.get(key, 0) + 1
    top_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\n📊 Kết quả đánh giá trên TEST set:")
    print(f"  Accuracy:        {accuracy:.4f}")
    print(f"  Macro F1:        {macro_f1:.4f}")
    print(f"  Macro Precision: {macro_precision:.4f}")
    print(f"  Macro Recall:    {macro_recall:.4f}")
    print(f"  Inference:       {ms_per_image:.2f} ms/ảnh | {images_per_sec:.1f} ảnh/giây")
    print(f"\n  Classification Report:\n{report}")
    if top_errors:
        print(f"  Top lỗi phân loại:")
        for pair, cnt in top_errors:
            print(f"    {pair}: {cnt} lần")

    return {
        "accuracy":               accuracy,
        "macro_f1":               float(macro_f1),
        "macro_precision":        float(macro_precision),
        "macro_recall":           float(macro_recall),
        "per_class":              per_class,
        "classification_report":  report,
        "confusion_matrix":       cm.tolist(),
        "top_error_pairs":        top_errors,
        "wrong_indices":          wrong_indices.tolist(),
        "all_preds":              np_preds.tolist(),
        "all_labels":             np_labels.tolist(),
        "inference_time_s":       float(inference_time),
        "ms_per_image":           float(ms_per_image),
        "images_per_sec":         float(images_per_sec),
        "total_images":           total_images,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Vẽ và lưu confusion matrix dưới dạng ảnh PNG."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("Nhãn thật", fontsize=12)
    ax.set_xlabel("Nhãn dự đoán", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Confusion matrix lưu tại: {save_path}")


def run_full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    exp_name: str,
    model_name: str,
    outputs_dir: str,
    runs_dir: str,
    metrics_path: str,
) -> Dict[str, Any]:
    """
    Đánh giá đầy đủ: metrics + confusion matrix + cập nhật metrics JSON + TensorBoard hparams.

    Args:
        model: Model đã nạp checkpoint tốt nhất.
        test_loader: DataLoader cho test set.
        class_names: Danh sách tên lớp.
        exp_name: Tên thí nghiệm.
        model_name: Tên model.
        outputs_dir: Thư mục lưu kết quả.
        runs_dir: Thư mục TensorBoard.
        metrics_path: Đường dẫn file metrics JSON (để cập nhật).

    Returns:
        Dict kết quả đánh giá đầy đủ (merge với metrics train).
    """
    device = get_device()
    model  = model.to(device)

    param_stats = count_parameters(model)
    size_mb     = model_size_mb(model)

    eval_result = evaluate(model, test_loader, class_names, device)

    # Vẽ confusion matrix
    cm      = np.array(eval_result["confusion_matrix"])
    cm_path = str(Path(outputs_dir) / exp_name / "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path, title=f"Confusion Matrix — {exp_name}")

    # Cập nhật metrics JSON với kết quả test
    existing = load_metrics(metrics_path) or {}
    existing.update({
        "test_acc":           eval_result["accuracy"],
        "test_f1":            eval_result["macro_f1"],
        "test_precision":     eval_result["macro_precision"],
        "test_recall":        eval_result["macro_recall"],
        "per_class_metrics":  eval_result["per_class"],
        "inference_ms_per_image": eval_result["ms_per_image"],
        "images_per_sec":     eval_result["images_per_sec"],
        "model_size_mb":      size_mb,
        "total_params":       param_stats["total"],
        "trainable_params":   param_stats["trainable"],
        "confusion_matrix":   eval_result["confusion_matrix"],
        "top_error_pairs":    eval_result["top_error_pairs"],
        "cm_path":            cm_path,
    })
    save_metrics(existing, metrics_path)

    # Cập nhật TensorBoard hparams với test_acc thật
    writer = SummaryWriter(log_dir=str(Path(runs_dir) / exp_name))
    writer.add_hparams(
        {
            "model":       model_name,
            "freeze_mode": existing.get("freeze_mode", ""),
            "lr":          float(existing.get("lr", 0)),
            "batch_size":  int(existing.get("batch_size", 32)),
            "optimizer":   existing.get("optimizer", ""),
            "epochs_run":  int(existing.get("epochs_run", 0)),
        },
        {
            "hparam/test_acc":     eval_result["accuracy"],
            "hparam/test_f1":      eval_result["macro_f1"],
            "hparam/best_val_acc": existing.get("best_val_acc", 0),
            "hparam/train_time_h": existing.get("train_time_h", 0),
            "hparam/peak_vram_mb": existing.get("peak_vram_mb", 0),
            "hparam/inference_ms": eval_result["ms_per_image"],
            "hparam/model_size_mb": size_mb,
        },
    )
    writer.close()

    return {**existing, **eval_result}


def build_summary_table(outputs_dir: str) -> List[Dict]:
    """
    Đọc tất cả metrics JSON trong outputs_dir và tạo bảng tổng hợp.

    Returns:
        List of dicts, mỗi dict là một thí nghiệm.
    """
    import json
    rows: List[Dict] = []
    outputs_path = Path(outputs_dir)

    if not outputs_path.exists():
        return rows

    for exp_dir in sorted(outputs_path.iterdir()):
        metrics_file = exp_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, encoding="utf-8") as f:
                m = json.load(f)
            rows.append({
                "exp_name":      m.get("exp_name", exp_dir.name),
                "model":         m.get("model_name", ""),
                "freeze_mode":   m.get("freeze_mode", ""),
                "optimizer":     m.get("optimizer", ""),
                "lr":            m.get("lr", None),
                "batch_size":    m.get("batch_size", None),
                "best_val_acc":  m.get("best_val_acc", None),
                "test_acc":      m.get("test_acc", None),
                "test_f1":       m.get("test_f1", None),
                "epochs_run":    m.get("epochs_run", None),
                "train_time_h":  m.get("train_time_h", None),
                "peak_vram_mb":  m.get("peak_vram_mb", None),
                "inference_ms":  m.get("inference_ms_per_image", None),
                "model_size_mb": m.get("model_size_mb", None),
            })
    return rows
