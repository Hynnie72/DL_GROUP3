# Lab 2: Transfer Learning với Pretrained CNNs

Phân loại ảnh phong cảnh Intel thành 6 lớp bằng ResNet18, VGG16 và DenseNet121; so sánh freeze backbone với fine-tune lớp cuối, learning rate, batch size và optimizer.

---

## Thông tin dự án

- **Dataset:** Intel Image Classification (Kaggle `puneet6060`), 6 lớp
- **Models:** ResNet18 · VGG16 · DenseNet121, weights ImageNet-1K
- **Stack:** Python 3.9+, PyTorch 2.x, torchvision, TensorBoard, scikit-learn
- **Platform:** Kaggle · Tesla T4 · 14,6 GB VRAM
- **Trạng thái:** Smoke test 63/63 · **9/9 thí nghiệm P0 đã train và evaluate** trên Kaggle (Tesla T4) 
- **Kết quả tốt nhất:** ResNet18 `finetune_last` lr=1e-3 bs=32 SGD — test_acc=**0.9463**, F1=**0.9471**

---

## Mục tiêu

1. So sánh 3 kiến trúc pretrained trên cùng ngân sách train
2. Đánh giá ảnh hưởng của `freeze_backbone` và `finetune_last` (`full_finetune` là P1)
3. Khảo sát learning rate (1e-2, 1e-3, 1e-4), batch size (16, 32, 64), SGD và Adam
4. Theo dõi huấn luyện bằng TensorBoard: loss, accuracy, lr, thời gian, bộ nhớ
5. Đánh giá test bằng accuracy, macro F1, confusion matrix và phân tích lỗi
6. Demo dự đoán trên `seg_pred` và lưu bảng chạy

---

## Trạng thái hiện tại

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Tài liệu `docs/` | Hoàn thành | Đủ 7 file: Define + Design và tài liệu chuyên đề |
| Mã nguồn `src/` | Hoàn thành | Data, models, training, evaluation, utils |
| Cấu hình `configs/` | Hoàn thành | 23 thí nghiệm E1–E5 |
| Smoke test | 63/63 | `tests/smoke_test.py`, 25,3 giây |
| Notebook | Hoàn thành, đã chạy | 9/9 P0 đã train và evaluate trên Kaggle |
| Thực nghiệm | 9/9 P0 đã train | `outputs/` và `runs/` đã copy về project |
| Test metric | Có đủ | test_acc=**0.9463**, F1=**0.9471** (best) |

---

## Cấu trúc thư mục

```
LAB2-DL-NHOM3/
docs/
labs2_description.md # Define, Design, yêu cầu REQ
data.md # Dataset, transforms, split, DataLoader
model.md # Kiến trúc, freeze mode, BatchNorm
training.md # Loss, optimizer, scheduler, AMP, resume
evaluation.md # Metric, kết quả thực tế, kết luận
demo.md # Runbook local và Kaggle
plan_progress_status.md # Tiến độ, known issues
configs/
base.yaml # Baseline
smoke.yaml # Smoke test
experiments.yaml # 23 thí nghiệm E1–E5
resnet18.yaml
vgg16.yaml
densenet121.yaml
src/
utils.py # Seed, device, checkpoint, metrics, freeze helper
data.py # Dataset, transforms, split, DataLoader
train.py # Training loop, AMP, early stopping, TensorBoard
evaluate.py # Metrics, confusion matrix, error analysis
models/
__init__.py # Registry, build_model
resnet18.py
vgg16.py
densenet121.py
tests/smoke_test.py # 3 model × 3 freeze mode, không cần dataset/Internet
scripts/
run_smoke.sh
run_normal.sh
check_vs_description.py
notebooks/lab2_report.ipynb # Báo cáo và thuyết trình chính
outputs/ # Checkpoint, metrics, hình (sinh khi chạy)
runs/ # TensorBoard logs (sinh khi chạy)
requirements.txt
README.md
```

---

## Cài đặt

```bash
python -m venv venv
# Windows
source venv/Scripts/activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

Kiểm tra nhanh:
```bash
python -c "import torch, torchvision, tensorboard; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## Smoke test

```bash
PYTHONIOENCODING=utf-8 python tests/smoke_test.py
```

Smoke test chạy 3 model × 3 freeze mode với `pretrained=False`: build model, output shape `[batch, 6]`, forward/backward, loss hữu hạn, gradient chỉ ở tham số trainable, số tham số trainable và TensorBoard event. Không cần dataset thật, không cần Internet.

**Kết quả hiện tại: 63/63 đạt, 25,3 giây.**

---

## Kết quả thí nghiệm P0

| # | Thí nghiệm | Test Acc | F1 | Train (h) | ms/img |
|---|---|---:|---:|---:|---:|
| 1 | ResNet18 finetune_last lr=1e-3 bs=32 SGD | **0.9463** | **0.9471** | 0,39 | 2,4 |
| 2 | ResNet18 finetune_last lr=1e-3 bs=16 SGD | 0.9413 | 0.9422 | 0,20 | 2,4 |
| 3 | ResNet18 finetune_last lr=1e-2 bs=32 SGD | 0.9373 | 0.9382 | 0,20 | 2,4 |
| 4 | ResNet18 finetune_last lr=1e-4 bs=32 Adam | 0.9337 | 0.9350 | 0,30 | 2,4 |
| 5 | ResNet18 finetune_last lr=1e-3 bs=64 SGD | 0.9323 | 0.9335 | 0,38 | 2,4 |
| 6 | ResNet18 finetune_last lr=1e-4 bs=32 SGD | 0.9203 | 0.9214 | 0,38 | 2,4 |
| 7 | DenseNet121 freeze_backbone lr=1e-3 bs=32 SGD | 0.9183 | 0.9200 | 0,41 | 3,6 |
| 8 | VGG16 freeze_backbone lr=1e-3 bs=32 SGD | 0.9170 | 0.9190 | 0,42 | 5,2 |
| 9 | ResNet18 freeze_backbone lr=1e-3 bs=32 SGD | 0.9067 | 0.9087 | 0,50 | 38,5 |

---

## Dataset

| Thông tin | Giá trị |
|---|---|
| Tên | Intel Image Classification |
| Nguồn | Kaggle: `puneet6060/intel-image-classification` |
| Số lớp | 6: `buildings` · `forest` · `glacier` · `mountain` · `sea` · `street` |
| Ảnh gốc | khoảng 150×150 |
| Input model | 224×224 |
| Train + val | `seg_train/seg_train/<class>/*.jpg` |
| Test | `seg_test/seg_test/<class>/*.jpg` |
| Demo | `seg_pred/seg_pred/*.jpg` — không nhãn |
| Chia val | 15% stratified, seed 42 |

---

## Models

| Model | Tổng tham số | Trainable `freeze_backbone` | Trainable `finetune_last` | Kích thước FP32 |
|---|---:|---:|---:|---:|
| ResNet18 | 11 179 590 | 3 078 | 8 396 806 | 42,6 MB |
| VGG16 | 134 285 126 | 24 582 | 126 649 862 | 512,3 MB |
| DenseNet121 | 6 960 006 | 6 150 | 2 166 278 | 26,6 MB |

---

## Thí nghiệm

| Nhóm | Nội dung | Số lần | Priority |
|---|---|---:|---|
| E1 | 3 model × 3 freeze mode (P0: 3 model `freeze_backbone` + ResNet18 `finetune_last`) | 9 | 4 P0 + 5 P1 |
| E2 | Learning rate 1e-2, 1e-4 | 2 | P0 |
| E3 | Batch size 16, 64 | 2 | P0 |
| E4 | Optimizer Adam lr=1e-4, 1e-3 | 2 | 1 P0 + 1 P1 |
| E5 | E2 + E3 cho VGG16, DenseNet121 | 8 | P1 |
| **Tổng** | | **23** | 9 P0 + 14 P1 |

Danh sách đầy đủ: [`configs/experiments.yaml`](configs/experiments.yaml).
Tên thí nghiệm: `{model}__{freeze_mode}__lr{lr}__bs{batch_size}__{optimizer}`

---

## TensorBoard

```bash
tensorboard --logdir runs/
# http://localhost:6006
```

Các tab: Scalars (loss, accuracy, lr, epoch time, memory), Graphs, HParams, Images.

---

## Tài liệu

| File | Nội dung |
|---|---|
| [`docs/labs2_description.md`](docs/labs2_description.md) | Define + Design, yêu cầu REQ, kiến trúc hệ thống |
| [`docs/data.md`](docs/data.md) | Dataset card, transforms, split, DataLoader |
| [`docs/model.md`](docs/model.md) | Kiến trúc, classifier head, 3 freeze mode, BatchNorm |
| [`docs/training.md`](docs/training.md) | Loss, optimizer, scheduler, early stopping, AMP, resume |
| [`docs/evaluation.md`](docs/evaluation.md) | Metric, kết quả thực tế, confusion matrix, kết luận |
| [`docs/demo.md`](docs/demo.md) | Runbook local và Kaggle từng bước |
| [`docs/plan_progress_status.md`](docs/plan_progress_status.md) | Tiến độ, known issues, nhật ký |

---

## Nhóm thực hiện

**Nhóm 3 — Deep Learning, 2026.**

---

## Tài liệu tham khảo

- [Deep Residual Learning for Image Recognition — He et al., 2015](https://arxiv.org/abs/1512.03385)
- [Very Deep Convolutional Networks — Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556)
- [Densely Connected Convolutional Networks — Huang et al., 2017](https://arxiv.org/abs/1608.06993)
- [PyTorch torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
