"""
train.py - Vòng lặp huấn luyện cho Lab 2.

Hỗ trợ:
- AMP (Automatic Mixed Precision)
- TensorBoard logging (loss, accuracy, lr, hparams, graph, images, RAM/VRAM)
- Early stopping theo val accuracy (patience 8)
- Cơ chế resume (bỏ qua thí nghiệm đã có kết quả)
- Đo thời gian epoch và tổng thời gian train
- Đo RAM/VRAM peak
- Lưu checkpoint tốt nhất và metrics JSON
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.utils import (
    set_seed, get_device, save_checkpoint, save_metrics, load_metrics,
    experiment_exists, get_peak_memory_mb, count_parameters, model_size_mb,
)


# ──────────────────────────────────────────────
# Hàm tạo optimizer và scheduler
# ──────────────────────────────────────────────

def build_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Tạo optimizer từ config. Chỉ nhận tham số có requires_grad=True."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    opt_name = config.get("name", "sgd").lower()
    lr = float(config.get("lr", 1e-3))
    wd = float(config.get("weight_decay", 5e-4))

    if opt_name == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=wd,
        )
    elif opt_name == "adam":
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Optimizer '{opt_name}' không được hỗ trợ. Chọn: sgd, adam, adamw")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict,
) -> Optional[Any]:
    """Tạo learning rate scheduler từ config."""
    sched_name = config.get("name", "cosine").lower()

    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config.get("T_max", 30))
        )
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 10)),
            gamma=float(config.get("gamma", 0.1)),
        )
    elif sched_name in ("none", ""):
        return None
    else:
        raise ValueError(f"Scheduler '{sched_name}' không được hỗ trợ.")


# ──────────────────────────────────────────────
# Một epoch train
# ──────────────────────────────────────────────

def _get_model_module(model_name: str):
    """Nạp module tương ứng để gọi set_train_mode."""
    if model_name == "resnet18":
        import src.models.resnet18 as m
    elif model_name == "vgg16":
        import src.models.vgg16 as m
    elif model_name == "densenet121":
        import src.models.densenet121 as m
    else:
        raise ValueError(f"model_name không hợp lệ: {model_name}")
    return m


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    freeze_mode: str,
    model_module: Any,
) -> Tuple[float, float]:
    """
    Chạy một epoch train.

    Returns:
        (avg_loss, accuracy)
    """
    model_module.set_train_mode(model, freeze_mode)

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=90)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Đánh giá model trên tập val.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="  Val  ", leave=False, ncols=90)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ──────────────────────────────────────────────
# Hàm train chính
# ──────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    exp_name: str,
    freeze_mode: str,
    model_name: str,
    class_to_idx: Dict[str, int],
    outputs_dir: str,
    runs_dir: str,
    first_model_timing: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Hàm train chính. Hỗ trợ early stopping, AMP, TensorBoard, resume, checkpoint.

    Args:
        model: Model đã được build và freeze.
        train_loader, val_loader: DataLoader.
        config: Dict cấu hình (training, optimizer, scheduler, early_stopping, tensorboard).
        exp_name: Tên thí nghiệm (dùng cho đường dẫn lưu).
        freeze_mode: Chế độ freeze (để truyền set_train_mode).
        model_name: Tên model ('resnet18', 'vgg16', 'densenet121').
        class_to_idx: Ánh xạ tên lớp → index.
        outputs_dir: Thư mục lưu checkpoint và metrics.
        runs_dir: Thư mục TensorBoard logs.
        first_model_timing: Dict để ghi thời gian epoch đầu (ước tính tổng thời gian).

    Returns:
        Dict kết quả: best_val_acc, best_epoch, epochs_run, train_time_s, ckpt_path, v.v.
    """
    # ── Chữ ký cấu hình: đổi regularization/hyperparameter thì không resume kết quả cũ ──
    _opt_cfg   = config.get("optimizer", {})
    _loss_cfg  = config.get("loss", {})
    config_signature = {
        "optimizer":       str(_opt_cfg.get("name", "sgd")).lower(),
        "lr":              float(_opt_cfg.get("lr", 1e-3)),
        "weight_decay":    float(_opt_cfg.get("weight_decay", 0.0)),
        "label_smoothing": float(_loss_cfg.get("label_smoothing", 0.0)),
    }

    # ── Kiểm tra resume ──
    exp_outputs = Path(outputs_dir) / exp_name
    metrics_path = str(exp_outputs / "metrics.json")

    if experiment_exists(exp_name, outputs_dir, config_signature):
        print(f"⏭️  Bỏ qua '{exp_name}' (đã có kết quả, cơ chế resume)")
        return load_metrics(metrics_path)

    exp_outputs.mkdir(parents=True, exist_ok=True)

    # ── Thiết lập ──
    set_seed(config.get("seed", 42))
    device = get_device()
    model = model.to(device)

    model_module = _get_model_module(model_name)

    # Loss, optimizer, scheduler
    loss_config    = config.get("loss", {})
    label_smoothing = float(loss_config.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"  🎯 Label smoothing: {label_smoothing}")
    opt_config   = config.get("optimizer",      {"name": "sgd", "lr": 1e-3})
    sched_config = config.get("scheduler",      {"name": "cosine", "T_max": 30})
    train_config = config.get("training",       {})
    es_config    = config.get("early_stopping", {"patience": 8})
    tb_config    = config.get("tensorboard",    {})

    optimizer = build_optimizer(model, opt_config)
    scheduler = build_scheduler(optimizer, sched_config)

    # AMP
    use_amp = train_config.get("amp", True) and device.type == "cuda"
    scaler  = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  ⚡ AMP (Automatic Mixed Precision) đang bật")

    # Early stopping
    patience = int(es_config.get("patience", 8))

    # TensorBoard
    tb_log_dir = str(Path(runs_dir) / exp_name)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"  📊 TensorBoard log dir: {tb_log_dir}")

    # Ghi graph (add_graph)
    if tb_config.get("log_graph", True):
        try:
            dummy = torch.zeros(1, 3, 224, 224).to(device)
            writer.add_graph(model, dummy)
        except Exception as e:
            print(f"  ⚠️  add_graph thất bại: {e}")

    # Ghi ảnh mẫu đầu vào
    if tb_config.get("log_images", True):
        try:
            sample_images, _ = next(iter(train_loader))
            writer.add_images("train/sample_images", sample_images[:8], 0)
        except Exception:
            pass

    # ── Vòng lặp train ──
    max_epochs = int(train_config.get("epochs", 30))

    best_val_acc      = 0.0
    best_val_loss     = float("inf")
    best_epoch        = 0
    epochs_no_improve = 0
    history: Dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [], "lr": [],
    }

    ckpt_latest = str(exp_outputs / "checkpoint_latest.pth")
    ckpt_best   = str(exp_outputs / "checkpoint_best.pth")

    train_start = time.time()
    final_epoch = 0

    for epoch in range(1, max_epochs + 1):
        final_epoch = epoch
        epoch_start = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, freeze_mode, model_module,
        )

        # Validation
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        mem = get_peak_memory_mb()

        print(
            f"  Epoch [{epoch:2d}/{max_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        # Ước tính sau epoch 2: epoch 1 thường chậm hơn nhiều do warmup/cuDNN autotune,
        # nên lấy epoch 1 sẽ phóng đại thời gian (vd 431s vs 39s).
        if epoch == 1 and first_model_timing is not None:
            first_model_timing["epoch_time_s"] = epoch_time
        if epoch == 2 and first_model_timing is not None:
            first_model_timing["steady_epoch_time_s"] = epoch_time
            est_model_h = epoch_time * max_epochs / 3600
            print(f"\n  📊 Epoch 2 (steady): {epoch_time:.1f}s "
                  f"→ đủ {max_epochs} epoch ≈ {est_model_h:.2f}h cho model này.")
            print("     Early stopping (patience=8) thường dừng sớm hơn; "
                  "model khác có thời gian/epoch khác nhau.\n")

        # TensorBoard
        writer.add_scalar("Loss/train",       train_loss, epoch)
        writer.add_scalar("Loss/val",         val_loss,   epoch)
        writer.add_scalar("Accuracy/train",   train_acc,  epoch)
        writer.add_scalar("Accuracy/val",     val_acc,    epoch)
        writer.add_scalar("LR/lr",            current_lr, epoch)
        writer.add_scalar("Perf/epoch_time_s", epoch_time, epoch)
        if mem.get("vram_mb", 0) > 0:
            writer.add_scalar("Memory/VRAM_peak_MB", mem["vram_mb"], epoch)
        writer.add_scalar("Memory/RAM_MB", mem.get("ram_mb", 0), epoch)

        if tb_config.get("log_histograms", False):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f"weights/{name}", param, epoch)

        # History
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Checkpoint
        param_stats = count_parameters(model)
        state = {
            "epoch": epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc":    val_acc,
            "val_loss":   val_loss,
            "train_acc":  train_acc,
            "train_loss": train_loss,
            "class_to_idx": class_to_idx,
            "freeze_mode":  freeze_mode,
            "model_name":   model_name,
            "param_stats":  param_stats,
        }

        # Chọn checkpoint tốt nhất theo val_acc; nếu val_acc bằng nhau (accuracy bị
        # lượng tử hoá theo số mẫu val) thì ưu tiên val_loss thấp hơn.
        acc_improved = val_acc > best_val_acc
        is_best = acc_improved or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc  = max(best_val_acc, val_acc)
            best_val_loss = val_loss
            best_epoch    = epoch
            save_checkpoint(state, ckpt_latest, is_best=True, best_path=ckpt_best)
            print(f"  💾 Best checkpoint lưu tại epoch {epoch} "
                  f"(val_acc={val_acc:.4f}, val_loss={val_loss:.4f})")
        else:
            save_checkpoint(state, ckpt_latest)

            # Early stopping theo val_acc (strict).
        # patience=8 có nghĩa: nếu 8 epoch liên tiếp val_acc không tăng → dừng.
        # Giá trị mặc định patience=8 trên dataset 14k ảnh là hợp lý:
        #   - Quá thấp (< 5): dừng sớm khi model đang học, bỏ lỡ cải thiện muộn.
        #   - Quá cao (> 12): kéo dài thời gian vô ích sau khi đã plateau.
        # Với CosineAnnealingLR T_max=30, lr giảm dần → model có thể cải thiện chậm ở
        # epoch cuối; patience=8 cho đủ cơ hội nhưng không lãng phí quota GPU.
        if acc_improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n  🛑 Early stopping tại epoch {epoch} "
                  f"(val_acc không tăng trong {patience} epoch liên tiếp)")
            break

    total_train_time = time.time() - train_start
    mem_final = get_peak_memory_mb()

    # TensorBoard HParams (ghi tạm với val_acc; sẽ cập nhật test_acc sau evaluate)
    hparam_dict = {
        "model":       model_name,
        "freeze_mode": freeze_mode,
        "lr":          float(opt_config.get("lr", 1e-3)),
        "batch_size":  int(train_config.get("batch_size", 32)),
        "optimizer":   opt_config.get("name", "sgd"),
        "label_smoothing": label_smoothing,
        "epochs_run":  final_epoch,
    }
    metric_dict = {
        "hparam/best_val_acc":  best_val_acc,
        "hparam/train_time_h":  total_train_time / 3600,
        "hparam/peak_vram_mb":  mem_final.get("vram_mb", 0),
    }
    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()

    # Lưu metrics JSON
    result: Dict[str, Any] = {
        "exp_name":    exp_name,
        "model_name":  model_name,
        "freeze_mode": freeze_mode,
        "lr":          float(opt_config.get("lr", 1e-3)),
        "batch_size":  int(train_config.get("batch_size", 32)),
        "optimizer":   opt_config.get("name", "sgd"),
        "weight_decay": float(opt_config.get("weight_decay", 0.0)),
        "label_smoothing": label_smoothing,
        "config_signature": config_signature,
        "best_val_acc":  best_val_acc,
        "best_val_loss": best_val_loss if best_epoch else None,
        "best_epoch":    best_epoch,
        "epochs_run":    final_epoch,
        "train_time_s":  total_train_time,
        "train_time_h":  total_train_time / 3600,
        "ckpt_path":     ckpt_best,
        "history":       history,
        "peak_vram_mb":  mem_final.get("vram_mb", 0),
        "peak_ram_mb":   mem_final.get("ram_mb", 0),
        "completed":     True,
        "test_acc":      None,
        "test_f1":       None,
        "inference_ms_per_image": None,
    }
    save_metrics(result, metrics_path)

    print(f"\n  ✅ Train xong '{exp_name}'")
    print(f"     Best val_acc={best_val_acc:.4f} (val_loss={best_val_loss:.4f}) "
          f"tại epoch {best_epoch}/{final_epoch}")
    print(f"     Tổng thời gian: {total_train_time/3600:.2f}h ({total_train_time:.0f}s)")

    return result
