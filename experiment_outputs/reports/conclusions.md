# Experiment Conclusions

Baseline model 'Cơ sở: 2 lớp ẩn (256-128)': Best Accuracy 88.70%, Best Evaluation Loss 0.312283, Macro F1 0.8862.

Group 1. Số lớp ẩn:
- Highest: 3 lớp ẩn (256-128-64) (88.53%, -0.17 pp vs baseline)
- Lowest: 1 lớp ẩn (256) (88.45%, -0.25 pp vs baseline)

Group 2. Độ rộng lớp ẩn:
- Highest: Rộng: 2 lớp ẩn (512-256) (88.67%, -0.03 pp vs baseline)
- Lowest: Hẹp: 2 lớp ẩn (64-32) (86.77%, -1.93 pp vs baseline)

Group 3. Dropout:
- Highest: Dropout 0.0 (88.40%, -0.30 pp vs baseline)
- Lowest: Dropout 0.5 (87.10%, -1.60 pp vs baseline)

Group 4. Hàm kích hoạt:
- Highest: Activation RELU (88.70%, +0.00 pp vs baseline)
- Lowest: Activation Tanh (87.05%, -1.65 pp vs baseline)

Group 5. Bộ tối ưu & learning rate:
- Highest: SGD lr=0.01 momentum=0.9 (89.00%, +0.30 pp vs baseline)
- Lowest: Adam lr=0.01 (82.03%, -6.67 pp vs baseline)

Group 6. BatchNorm:
- Có BatchNorm: 89.40% (+0.70 pp vs baseline)

Group 7. Batch size:
- Highest: Batch size 256 (89.16%, +0.46 pp vs baseline)
- Lowest: Batch size 32 (88.16%, -0.54 pp vs baseline)

Experiment with the highest Best Accuracy: Có BatchNorm (89.40%, Macro F1 0.8937).

Experiment with the lowest Best Evaluation Loss: Có BatchNorm (0.289237, epoch 34).

Experiment with the largest Train - Evaluation gap at its best epoch: Dropout 0.0 (0.78 points).

Hardest class (mean over experiments): Shirt (62.1%); easiest: Trouser (97.1%).

Increasing Số lớp ẩn: 1: 88.45% -> 2: 88.70% -> 2: 88.70% -> 3: 88.53%
- Best: Số lớp ẩn = 2 (88.70%, +0.00 pp vs baseline)
- Worst: Số lớp ẩn = 1 (88.45%, -0.25 pp vs baseline)
- Train-Evaluation gap at best epoch: from -1.26 to -0.88 points
- Training time: from 1239s to 1498s
- Number of parameters: from 203,530 to 242,762

Increasing Learning rate (Adam): 0.0001: 88.73% -> 0.001: 88.70% -> 0.001: 88.70% -> 0.01: 82.03%
- Best: Learning rate (Adam) = 0.0001 (88.73%, +0.03 pp vs baseline)
- Worst: Learning rate (Adam) = 0.01 (82.03%, -6.67 pp vs baseline)
- Train-Evaluation gap at best epoch: from -7.84 to -0.90 points
- Training time: from 1237s to 1437s
- Number of parameters: from 235,146 to 235,146

Increasing Batch size: 32: 88.16% -> 64: 88.70% -> 64: 88.70% -> 256: 89.16%
- Best: Batch size = 256 (89.16%, +0.46 pp vs baseline)
- Worst: Batch size = 32 (88.16%, -0.54 pp vs baseline)
- Train-Evaluation gap at best epoch: from -1.36 to -1.17 points
- Training time: from 1239s to 4192s
- Number of parameters: from 235,146 to 235,146
