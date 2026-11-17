# Detailed Analysis: Adam lr=0.0001
## Configuration

- **name**: Adam lr=0.0001
- **group**: 5. Bộ tối ưu & learning rate
- **hidden_sizes**: 256, 128
- **activation**: relu
- **dropout**: 0.2
- **batch_norm**: False
- **optimizer**: adam
- **lr**: 0.0001
- **momentum**: 0.9
- **batch_size**: 64
- **epochs**: 35
- **seed**: 42
- **architecture**: 784 -> 256 -> 128 -> 10
- **optimizer_text**: Adam (lr=0.0001)
## BEST checkpoint (lowest evaluation loss)

- **Path**: ./saved_models\adam_lr_0_0001\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 34
- **Best evaluation loss**: 0.313997
- **Train accuracy at best epoch (%)**: 87.83
- **Evaluation accuracy (%)**: 88.73
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\adam_lr_0_0001\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.314316
- **Train accuracy (%)**: 87.86
- **Evaluation accuracy (%)**: 88.51
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 88.7300
- **Macro Precision**: 0.8867
- **Macro Recall**: 0.8873
- **Macro F1-score**: 0.8866
- **Number of parameters**: 235,146
- **Training time (s)**: 1237.17
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.8357    0.8290    0.8323      1000
     Trouser     0.9819    0.9750    0.9784      1000
    Pullover     0.8215    0.7870    0.8039      1000
       Dress     0.8513    0.9220    0.8853      1000
        Coat     0.7965    0.8260    0.8110      1000
      Sandal     0.9885    0.9450    0.9663      1000
       Shirt     0.7247    0.6740    0.6984      1000
     Sneaker     0.9343    0.9670    0.9504      1000
         Bag     0.9741    0.9760    0.9750      1000
  Ankle boot     0.9586    0.9720    0.9652      1000

    accuracy                         0.8873     10000
   macro avg     0.8867    0.8873    0.8866     10000
weighted avg     0.8867    0.8873    0.8866     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\adam_lr_0_0001_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\adam_lr_0_0001_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\adam_lr_0_0001_sample_predictions.png
