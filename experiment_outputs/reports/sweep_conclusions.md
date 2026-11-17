# Parameter Sweep Conclusions

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
