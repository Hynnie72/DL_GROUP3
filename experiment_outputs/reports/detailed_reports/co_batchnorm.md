# Detailed Analysis: Có BatchNorm
## Configuration

- **name**: Có BatchNorm
- **group**: 6. BatchNorm
- **hidden_sizes**: 256, 128
- **activation**: relu
- **dropout**: 0.2
- **batch_norm**: True
- **optimizer**: adam
- **lr**: 0.001
- **momentum**: 0.9
- **batch_size**: 64
- **epochs**: 35
- **seed**: 42
- **architecture**: 784 -> 256 -> 128 -> 10
- **optimizer_text**: Adam (lr=0.001)
## BEST checkpoint (lowest evaluation loss)

- **Path**: ./saved_models\co_batchnorm\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 34
- **Best evaluation loss**: 0.289237
- **Train accuracy at best epoch (%)**: 88.22
- **Evaluation accuracy (%)**: 89.40
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\co_batchnorm\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.291929
- **Train accuracy (%)**: 88.15
- **Evaluation accuracy (%)**: 89.19
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 89.4000
- **Macro Precision**: 0.8941
- **Macro Recall**: 0.8940
- **Macro F1-score**: 0.8937
- **Number of parameters**: 235,914
- **Training time (s)**: 2449.57
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.8488    0.8420    0.8454      1000
     Trouser     0.9879    0.9800    0.9839      1000
    Pullover     0.8433    0.7860    0.8137      1000
       Dress     0.8685    0.9250    0.8959      1000
        Coat     0.8079    0.8370    0.8222      1000
      Sandal     0.9896    0.9520    0.9704      1000
       Shirt     0.7214    0.7070    0.7141      1000
     Sneaker     0.9300    0.9700    0.9496      1000
         Bag     0.9799    0.9770    0.9785      1000
  Ankle boot     0.9630    0.9640    0.9635      1000

    accuracy                         0.8940     10000
   macro avg     0.8941    0.8940    0.8937     10000
weighted avg     0.8941    0.8940    0.8937     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\co_batchnorm_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\co_batchnorm_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\co_batchnorm_sample_predictions.png
