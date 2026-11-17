# Detailed Analysis: Adam lr=0.01
## Configuration

- **name**: Adam lr=0.01
- **group**: 5. Bộ tối ưu & learning rate
- **hidden_sizes**: 256, 128
- **activation**: relu
- **dropout**: 0.2
- **batch_norm**: False
- **optimizer**: adam
- **lr**: 0.01
- **momentum**: 0.9
- **batch_size**: 64
- **epochs**: 35
- **seed**: 42
- **architecture**: 784 -> 256 -> 128 -> 10
- **optimizer_text**: Adam (lr=0.01)
## BEST checkpoint (lowest evaluation loss)

- **Path**: ./saved_models\adam_lr_0_01\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 20
- **Best evaluation loss**: 0.528743
- **Train accuracy at best epoch (%)**: 74.19
- **Evaluation accuracy (%)**: 82.03
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\adam_lr_0_01\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.569692
- **Train accuracy (%)**: 74.78
- **Evaluation accuracy (%)**: 81.94
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 82.0300
- **Macro Precision**: 0.8196
- **Macro Recall**: 0.8203
- **Macro F1-score**: 0.8152
- **Number of parameters**: 235,146
- **Training time (s)**: 1247.78
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.6860    0.8650    0.7651      1000
     Trouser     0.9823    0.9410    0.9612      1000
    Pullover     0.6504    0.6920    0.6705      1000
       Dress     0.8262    0.8320    0.8291      1000
        Coat     0.7000    0.7840    0.7396      1000
      Sandal     0.9736    0.8840    0.9266      1000
       Shirt     0.6194    0.3710    0.4640      1000
     Sneaker     0.9037    0.9290    0.9162      1000
         Bag     0.9421    0.9430    0.9425      1000
  Ankle boot     0.9127    0.9620    0.9367      1000

    accuracy                         0.8203     10000
   macro avg     0.8196    0.8203    0.8152     10000
weighted avg     0.8196    0.8203    0.8152     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\adam_lr_0_01_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\adam_lr_0_01_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\adam_lr_0_01_sample_predictions.png
