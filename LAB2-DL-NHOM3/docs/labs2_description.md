# Lab 2: Transfer Learning - Intel Scene Classification

| Truong        | Noi dung                        |
|---------------|---------------------------------|
| Module        | docs/labs2_description.md       |
| Author        | Nhom 3 - LAB2-DL-NHOM3         |
| Created date  | 2026-10                         |
| Updated date  | 2026-11-09                      |

---

## Purpose

Phan loai anh phong canh Intel 6 lop (buildings, forest, glacier, mountain, sea, street)
bang Transfer Learning tu 3 CNN pretrained tren ImageNet-1K: ResNet18, VGG16, DenseNet121.
So sanh anh huong cua freeze mode, learning rate, batch size va optimizer den do chinh xac.

---

## Input & Output

- **Input**
  - Nguon    : Kaggle `puneet6060/intel-image-classification`
  - Dinh dang: JPEG ~150x150, resize ve 224x224, chuan hoa ImageNet mean/std
  - Shape    : `[B, 3, 224, 224]`
  - Config   : `configs/*.yaml` | Ma nguon: `src/`

- **Output**
  - `outputs/<exp>/checkpoint_best.pth`   -- trong so tot nhat
  - `outputs/<exp>/metrics.json`          -- test_acc, F1, history
  - `outputs/<exp>/confusion_matrix.png`  -- ma tran nham lan
  - `outputs/figures/chart_*.png`         -- bieu do tong hop
  - `runs/<exp>/`                         -- TensorBoard logs

---

## Model Architecture

| Thanh phan  | Mo ta                                              |
|-------------|----------------------------------------------------|
| Backbone    | ResNet18 (11.2M) / VGG16 (134.3M) / DenseNet121 (7M) |
| Head        | `nn.Linear(feature_dim, 6)` thay the head cu      |
| Num classes | 6                                                  |
| Input size  | 224 x 224 x 3 (RGB, chuan hoa ImageNet)            |
| Output      | Logits `[B, 6]`, chua softmax                      |

---

## Success Criteria

- Hoan thanh >= 9 thi nghiem P0
- Test Accuracy >= 90% (dat: 94.63% - ResNet18 finetune_last)
- Notebook chay tu dau den cuoi khong loi (Kaggle Run All)
- Smoke test < 2 phut tren CPU (dat: 26s, 63/63)
- Tat ca 13 REQ co artifact bang chung

---

## How to do

| Buoc | Noi dung |
|------|----------|
| 1    | Chuan bi data: load Intel dataset, transform, chia train/val (85/15, seed=42) |
| 2    | Build model: load pretrained weights, thay head 6 lop, ap dung freeze mode |
| 3    | Train: forward/backward, AMP, cosine scheduler, early stopping, TensorBoard, checkpoint |
| 4    | Experiment: chay 9 cau hinh P0 theo `configs/experiments.yaml` |
| 5    | Evaluate: test accuracy, macro F1, confusion matrix, phan tich loi |
| 6    | Report: `lab2_report.ipynb` tong hop ket qua; `scripts/py_report.py` tu dong hoa |

---

## Cau truc du an

```
LAB2-DL-NHOM3/
|
|-- README.md                          # Gioi thieu du an, huong dan chay nhanh
|-- requirements.txt                   # Thu vien Python can thiet
|
|-- configs/                           # Cau hinh thi nghiem (YAML)
|   |-- base.yaml                      # Cau hinh chung: dataset, training, seed
|   |-- experiments.yaml               # 23 thi nghiem E1-E5 day du
|   |-- smoke.yaml                     # Cau hinh smoke test (nhanh, CPU)
|   |-- resnet18.yaml                  # Cau hinh rieng ResNet18
|   |-- vgg16.yaml                     # Cau hinh rieng VGG16
|   |-- densenet121.yaml               # Cau hinh rieng DenseNet121
|
|-- docs/                              # Tai lieu mo ta du an (7 file)
|   |-- labs2_description.md           # [FILE NAY] Define + Design + Cau truc
|   |-- model.md                       # Kien truc model, freeze mode, ket qua
|   |-- data.md                        # Dataset, transform, split, thong ke
|   |-- training.md                    # Pipeline train, optimizer, scheduler
|   |-- evaluation.md                  # Metric, ket qua 9 TN, phan tich
|   |-- demo.md                        # Huong dan chay local va tren Kaggle
|   |-- plan_progress_status.md        # Ke hoach, tien do, REQ checklist
|
|-- notebooks/
|   |-- lab2_report.ipynb              # Notebook bao cao chinh (chay tren Kaggle)
|
|-- outputs/                           # Ket qua thi nghiem (sinh ra sau khi train)
|   |-- experiment_summary.csv         # Bang tong hop tat ca thi nghiem
|   |-- report_summary.json            # JSON ranking va chi so chinh
|   |-- summary_results.csv            # Ket qua chi tiet
|   |-- py_report.log                  # Log chay py_report.py
|   |-- sample_images.png              # Luoi anh mau tu dataset
|   |-- demo_predictions.png           # Anh demo du doan tren seg_pred
|   |-- learning_curves.png            # Learning curves tong hop
|   |-- model_layers_resnet18.csv      # Chi tiet layer ResNet18
|   |-- model_layers_vgg16.csv         # Chi tiet layer VGG16
|   |-- model_layers_densenet121.csv   # Chi tiet layer DenseNet121
|   |
|   |-- figures/                       # Bieu do so sanh (PNG)
|   |   |-- chart_accuracy_f1.png      # So sanh Accuracy & F1
|   |   |-- chart_perclass_f1.png      # F1 theo tung lop
|   |   |-- chart_resource.png         # RAM/VRAM/Size model
|   |   |-- chart_time.png             # Thoi gian train & inference
|   |   |-- chart_tradeoff.png         # Accuracy vs Toc do (scatter)
|   |   |-- chart_errors.png           # Phan tich loi nham
|   |   |-- cmp_acc_f1.png             # So sanh Acc & F1 (py_report)
|   |   |-- cmp_perclass_f1.png        # Per-class F1 (py_report)
|   |   |-- cmp_size_vs_acc.png        # Size vs Accuracy (py_report)
|   |   |-- cmp_learning_curves_top3.png  # Learning curves Top 3
|   |
|   |-- resnet18__finetune_last__lr1e-3__bs32__sgd/    # [BEST] Test Acc=0.9463
|   |   |-- checkpoint_best.pth        # Trong so model tot nhat
|   |   |-- checkpoint_latest.pth      # Trong so epoch cuoi
|   |   |-- metrics.json               # Tat ca chi so: acc, F1, history...
|   |   |-- confusion_matrix.png       # Ma tran nham lan
|   |
|   |-- resnet18__finetune_last__lr1e-3__bs16__sgd/    # Test Acc=0.9413
|   |-- resnet18__finetune_last__lr1e-2__bs32__sgd/    # Test Acc=0.9373
|   |-- resnet18__finetune_last__lr1e-4__bs32__adam/   # Test Acc=0.9337
|   |-- resnet18__finetune_last__lr1e-3__bs64__sgd/    # Test Acc=0.9323
|   |-- resnet18__finetune_last__lr1e-4__bs32__sgd/    # Test Acc=0.9203
|   |-- densenet121__freeze_backbone__lr1e-3__bs32__sgd/ # Test Acc=0.9183
|   |-- vgg16__freeze_backbone__lr1e-3__bs32__sgd/     # Test Acc=0.9170
|   |-- resnet18__freeze_backbone__lr1e-3__bs32__sgd/  # Test Acc=0.9067
|
|-- runs/                              # TensorBoard event logs (1 folder / TN)
|   |-- resnet18__finetune_last__lr1e-3__bs32__sgd/
|   |-- resnet18__finetune_last__lr1e-3__bs16__sgd/
|   |-- ... (9 folder tuong ung 9 thi nghiem)
|
|-- scripts/                           # Script tien ich
|   |-- py_report.py                   # Tu dong hoa bao cao: CSV, JSON, 4 PNG
|   |-- run_all.py                     # Pipeline chay 1 lenh: smoke -> report
|   |-- run_normal.sh                  # Script bash chay thi nghiem (Kaggle/Linux)
|   |-- run_smoke.sh                   # Script bash chay smoke test
|   |-- check_vs_description.py        # Kiem tra artifact vs REQ
|
|-- src/                               # Ma nguon chinh
|   |-- __init__.py
|   |-- data.py                        # Dataset, transform, split, DataLoader
|   |-- train.py                       # Training loop, AMP, early stopping, TensorBoard
|   |-- evaluate.py                    # Metric, confusion matrix, error analysis
|   |-- utils.py                       # Seed, device, checkpoint, model size
|   |-- models/
|       |-- __init__.py                # Registry: build_model(name, freeze_mode)
|       |-- resnet18.py                # ResNet18: load pretrained + thay head
|       |-- vgg16.py                   # VGG16: load pretrained + thay head
|       |-- densenet121.py             # DenseNet121: load pretrained + thay head
|
|-- tests/
    |-- smoke_test.py                  # 63 test cases, < 2 phut tren CPU
                                       # Ket qua: 63/63 PASS trong 26s
```

---

## Define

- **Problem**: Phan loai anh phong canh 6 lop bang transfer learning
- **Models**: ResNet18, VGG16, DenseNet121 (pretrained ImageNet)
- **Experiments**: 9 cau hinh P0 — so sanh model, freeze mode, LR, batch size, optimizer
- **Data**: Kaggle Intel Image Classification, 14 034 train, 3 000 test

## Design

**4-layer architecture:**
- Data layer    -- `src/data.py`: load dataset, transform, split, DataLoader
- Model layer   -- `src/models/`: 3 kien truc, freeze mode
- Training layer-- `src/train.py`: training loop, AMP, early stopping, TensorBoard
- Eval layer    -- `src/evaluate.py`: metric, confusion matrix, error analysis
- Entry point   -- `notebooks/lab2_report.ipynb`

**Data flow:**
`configs` -> `find_dataset_root` -> `ImageFolder` -> `split 85/15`
-> `DataLoader` -> `build_model` -> `train()` -> `checkpoint_best.pth`
-> `run_full_evaluation()` -> `metrics.json + confusion_matrix.png`

**Naming convention:** `{model}__{freeze_mode}__lr{lr}__bs{bs}__{optimizer}`
Vi du: `resnet18__finetune_last__lr1e-3__bs32__sgd`

## Yeu cau (REQ)

| Ma     | Mo ta |
|--------|-------|
| REQ-01 | >= 3 model pretrained (ResNet18, VGG16, DenseNet121) |
| REQ-02 | >= 2 freeze mode (freeze_backbone, finetune_last) |
| REQ-03 | Thi nghiem learning rate (1e-2, 1e-3, 1e-4) |
| REQ-04 | Thi nghiem batch size (16, 32, 64) |
| REQ-05 | So sanh optimizer (SGD vs Adam) |
| REQ-06 | Log TensorBoard moi lan train |
| REQ-07 | Danh gia tren TEST set (accuracy, F1, confusion matrix) |
| REQ-08 | Du 7 file trong `docs/` |
| REQ-09 | Smoke test < 2 phut tren CPU |
| REQ-10 | Notebook chay duoc tu dau den cuoi tren Kaggle |
| REQ-11 | Co che resume |
| REQ-12 | Phan tich loi |
| REQ-13 | Demo du doan tren `seg_pred` |

---

**Tai lieu lien quan:** [data.md](data.md) | [model.md](model.md) | [training.md](training.md) | [evaluation.md](evaluation.md)
