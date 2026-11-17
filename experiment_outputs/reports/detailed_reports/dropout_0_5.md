# Detailed Analysis: Dropout 0.5
## Configuration

- **name**: Dropout 0.5
- **group**: 3. Dropout
- **hidden_sizes**: 256, 128
- **activation**: relu
- **dropout**: 0.5
- **batch_norm**: False
- **optimizer**: adam
- **lr**: 0.001
- **momentum**: 0.9
- **batch_size**: 64
- **epochs**: 35
- **seed**: 42
- **architecture**: 784 -> 256 -> 128 -> 10
- **optimizer_text**: Adam (lr=0.001)
## BEST checkpoint (lowest evaluation loss)

- **Path**: ./saved_models\dropout_0_5\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 35
- **Best evaluation loss**: 0.354618
- **Train accuracy at best epoch (%)**: 83.89
- **Evaluation accuracy (%)**: 87.10
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\dropout_0_5\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.354618
- **Train accuracy (%)**: 83.89
- **Evaluation accuracy (%)**: 87.10
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 87.1000
- **Macro Precision**: 0.8714
- **Macro Recall**: 0.8710
- **Macro F1-score**: 0.8679
- **Number of parameters**: 235,146
- **Training time (s)**: 1245.35
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.7492    0.9080    0.8210      1000
     Trouser     0.9838    0.9710    0.9774      1000
    Pullover     0.8002    0.7730    0.7864      1000
       Dress     0.8703    0.8860    0.8781      1000
        Coat     0.7355    0.8230    0.7768      1000
      Sandal     0.9773    0.9470    0.9619      1000
       Shirt     0.7511    0.5160    0.6117      1000
     Sneaker     0.9264    0.9560    0.9409      1000
         Bag     0.9730    0.9720    0.9725      1000
  Ankle boot     0.9476    0.9580    0.9528      1000

    accuracy                         0.8710     10000
   macro avg     0.8714    0.8710    0.8679     10000
weighted avg     0.8714    0.8710    0.8679     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\dropout_0_5_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\dropout_0_5_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\dropout_0_5_sample_predictions.png
