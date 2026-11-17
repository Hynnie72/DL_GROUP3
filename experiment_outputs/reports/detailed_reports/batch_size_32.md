# Detailed Analysis: Batch size 32
## Configuration

- **name**: Batch size 32
- **group**: 7. Batch size
- **hidden_sizes**: 256, 128
- **activation**: relu
- **dropout**: 0.2
- **batch_norm**: False
- **optimizer**: adam
- **lr**: 0.001
- **momentum**: 0.9
- **batch_size**: 32
- **epochs**: 35
- **seed**: 42
- **architecture**: 784 -> 256 -> 128 -> 10
- **optimizer_text**: Adam (lr=0.001)
## BEST checkpoint (lowest evaluation loss)

- **Path**: ./saved_models\batch_size_32\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 34
- **Best evaluation loss**: 0.327832
- **Train accuracy at best epoch (%)**: 86.80
- **Evaluation accuracy (%)**: 88.16
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\batch_size_32\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.336593
- **Train accuracy (%)**: 86.96
- **Evaluation accuracy (%)**: 88.02
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 88.1600
- **Macro Precision**: 0.8817
- **Macro Recall**: 0.8816
- **Macro F1-score**: 0.8792
- **Number of parameters**: 235,146
- **Training time (s)**: 4192.24
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.7801    0.8800    0.8271      1000
     Trouser     0.9878    0.9730    0.9804      1000
    Pullover     0.7706    0.8400    0.8038      1000
       Dress     0.8476    0.9230    0.8837      1000
        Coat     0.7976    0.7840    0.7907      1000
      Sandal     0.9806    0.9580    0.9691      1000
       Shirt     0.7908    0.5670    0.6605      1000
     Sneaker     0.9305    0.9640    0.9470      1000
         Bag     0.9720    0.9720    0.9720      1000
  Ankle boot     0.9598    0.9550    0.9574      1000

    accuracy                         0.8816     10000
   macro avg     0.8817    0.8816    0.8792     10000
weighted avg     0.8817    0.8816    0.8792     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\batch_size_32_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\batch_size_32_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\batch_size_32_sample_predictions.png
