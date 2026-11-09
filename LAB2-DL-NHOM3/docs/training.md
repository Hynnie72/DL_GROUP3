# Lab 2: Training — Pipeline, optimizer, scheduler và artifact

---

## 1. Luồng một thí nghiệm

`configs/base.yaml + experiments.yaml` → `build_model` → `train()` → `checkpoint + metrics.json` → `run_full_evaluation()` → `metrics.json` có test metric

- `train()` **không** tự đánh giá test; notebook gọi evaluate ở bước riêng sau khi train xong để test set không tham gia chọn model.
- Nếu `outputs/{exp_name}/metrics.json` đã có `completed=True` **và** `config_signature` khớp cấu hình hiện tại, `train()` bỏ qua thí nghiệm (resume mức thí nghiệm, không phải resume giữa epoch).

---

## 2. Loss và optimizer

- **Loss:** `nn.CrossEntropyLoss(label_smoothing=0.1)` — model trả logits `[batch, 6]`, không softmax trước.
- **Label smoothing 0.1:** giảm overconfidence, cải thiện calibration; tăng thêm regularization ngoài weight decay.
- **Optimizer:**
- SGD: `momentum=0.9`, `weight_decay=5e-4`, `nesterov=True`.
- Adam: `betas=(0.9, 0.999)`, `weight_decay=5e-4`.
- Chỉ truyền tham số `requires_grad=True` vào optimizer; tham số đóng băng không cập nhật và không chiếm bộ nhớ optimizer.

---

## 3. Scheduler — Cosine Annealing

`CosineAnnealingLR(T_max=epochs, eta_min=1e-6)` — lr giảm dần theo đường cosine từ `lr_init` về `eta_min` trong suốt quá trình train. Không có warmup riêng; scheduler bước sau mỗi epoch.

- Ưu điểm: tốc độ học cao ban đầu giúp thoát local minima; giảm cuối tránh dao động.
- Không dùng `ReduceLROnPlateau` để giữ scheduler deterministic và dễ tái lập.

---

## 4. Early stopping

| Thông số | Giá trị |
|---|---|
| Metric theo dõi | `val_accuracy` |
| Patience | 8 epoch |
| Delta | 0 (bất kỳ cải thiện nào đều reset counter) |
| Tie-break | `val_loss` thấp hơn khi `val_accuracy` bằng nhau |
| Tác dụng | Lưu `checkpoint_best.pth` khi tìm thấy val_acc mới tốt hơn |

- Early stopping dừng sớm khi val_acc không cải thiện, tiết kiệm quota Kaggle.
- `checkpoint_latest.pth` lưu sau mỗi epoch để resume nếu session bị ngắt (không dùng cho evaluate).

---

## 5. AMP — Automatic Mixed Precision

```python
from torch.amp import autocast, GradScaler
scaler = GradScaler("cuda")

with autocast("cuda"):
outputs = model(inputs)
loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

- AMP chỉ bật khi `torch.cuda.is_available()`. Trên CPU, `autocast("cuda")` tự bỏ qua.
- Giảm khoảng 30–40% VRAM; tăng tốc 1,5–2× trên Tesla T4.

---

## 6. Regularization

| Kỹ thuật | Giá trị | Vị trí |
|---|---|---|
| `weight_decay` | 5e-4 | Optimizer (SGD và Adam) |
| `label_smoothing` | 0.1 | `CrossEntropyLoss` |
| `RandomErasing` | p=0.1 | Data transform (train only) |
| `Dropout` VGG16 | p=0.5 | Kiến trúc gốc, giữ nguyên |
| Early stopping | patience=8 | Training loop |

---

## 7. TensorBoard

Ghi sau mỗi epoch:
- **Scalars:** `train/loss`, `train/acc`, `val/loss`, `val/acc`, `lr/epoch`, `time/epoch_s`, `memory/peak_vram_mb`
- **HParams:** sau evaluate — `hparam/test_acc`, `hparam/test_f1`, `hparam/best_val_acc`, `hparam/train_time_h`, `hparam/inference_ms`, `hparam/model_size_mb`
- **Graph:** model graph ghi một lần ở epoch đầu

```bash
# Local
tensorboard --logdir runs/
# http://localhost:6006

# Kaggle
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/runs/
```

---

## 8. Resume

- `experiment_exists()` kiểm tra `metrics.json` tồn tại và `completed=True`; nếu đúng, `train()` in thông báo bỏ qua và trả metrics cũ.
- Trên Kaggle, notebook copy các thư mục từ `RESUME_FROM` vào `OUTPUTS_DIR` trước khi train. Trỏ `RESUME_FROM` tới thư mục **chứa trực tiếp các thư mục thí nghiệm**, không trỏ lên cấp trên để tránh `outputs/outputs`.
- `completed=True` được ghi trước evaluate. Nếu train xong nhưng evaluate bị ngắt, lần sau train bị bỏ qua; bước evaluate vẫn chạy lại được vì notebook kiểm tra `test_acc` riêng.

---

## 9. Tái lập

`set_seed(42)` đặt seed cho `random`, NumPy, PyTorch và CUDA, đồng thời bật `cudnn.deterministic` và tắt `cudnn.benchmark`. Kết quả tái lập tốt nhất khi dùng cùng seed, cùng split, cùng phiên bản thư viện và cùng loại GPU; không đảm bảo tuyệt đối trên mọi môi trường.

---

## 10. Trạng thái

**9/9 P0 đã train và evaluate trên Kaggle (Tesla T4).** Kết quả tốt nhất: ResNet18 `finetune_last` lr=1e-3 bs=32 SGD — test_acc=**0.9463**, F1=**0.9471**.

| Nhóm | Nội dung | Priority | Trạng thái |
|---|---|---|---|
| E1 | 3 model × freeze mode | 4 P0 + 5 P1 | 4/4 P0 hoàn thành |
| E2 | Learning rate 1e-2, 1e-4 | P0 | 2/2 hoàn thành |
| E3 | Batch size 16, 64 | P0 | 2/2 hoàn thành |
| E4 | Adam lr=1e-4 | 1 P0 + 1 P1 | 1/1 P0 hoàn thành |
| E5 | E2+E3 cho VGG16/DenseNet121 | P1 | Chưa chạy |

---

**Tham chiếu:** `src/train.py` · `src/utils.py` · `configs/base.yaml` · [model.md](model.md) · [evaluation.md](evaluation.md) · [demo.md](demo.md)
