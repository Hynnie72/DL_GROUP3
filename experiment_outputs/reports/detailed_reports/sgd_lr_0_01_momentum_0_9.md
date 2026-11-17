# Detailed Analysis: SGD lr=0.01 momentum=0.9
## Configuration

- **name**: SGD lr=0.01 momentum=0.9
- **group**: 5. Bộ tối ưu & learning rate
- **hidden_sizes**: 256, 128
- **activation**: relu
- **dropout**: 0.2
- **batch_norm**: False
- **optimizer**: sgd
- **lr**: 0.01
- **momentum**: 0.9
- **batch_size**: 64
- **epochs**: 35
- **seed**: 42
- **architecture**: 784 -> 256 -> 128 -> 10
- **optimizer_text**: SGD (lr=0.01, momentum=0.9)
## BEST checkpoint (lowest evaluation loss)

- **Path**: ./saved_models\sgd_lr_0_01_momentum_0_9\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 35
- **Best evaluation loss**: 0.301729
- **Train accuracy at best epoch (%)**: 88.31
- **Evaluation accuracy (%)**: 89.00
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\sgd_lr_0_01_momentum_0_9\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.301729
- **Train accuracy (%)**: 88.31
- **Evaluation accuracy (%)**: 89.00
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 89.0000
- **Macro Precision**: 0.8895
- **Macro Recall**: 0.8900
- **Macro F1-score**: 0.8883
- **Number of parameters**: 235,146
- **Training time (s)**: 1224.62
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.8075    0.8850    0.8445      1000
     Trouser     0.9849    0.9800    0.9825      1000
    Pullover     0.7905    0.8340    0.8117      1000
       Dress     0.8857    0.8990    0.8923      1000
        Coat     0.7963    0.8250    0.8104      1000
      Sandal     0.9886    0.9510    0.9694      1000
       Shirt     0.7755    0.6080    0.6816      1000
     Sneaker     0.9246    0.9810    0.9520      1000
         Bag     0.9683    0.9770    0.9726      1000
  Ankle boot     0.9726    0.9600    0.9663      1000

    accuracy                         0.8900     10000
   macro avg     0.8895    0.8900    0.8883     10000
weighted avg     0.8895    0.8900    0.8883     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\sgd_lr_0_01_momentum_0_9_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\sgd_lr_0_01_momentum_0_9_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\sgd_lr_0_01_momentum_0_9_sample_predictions.png
