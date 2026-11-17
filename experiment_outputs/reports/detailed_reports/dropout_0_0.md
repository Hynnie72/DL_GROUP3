# Detailed Analysis: Dropout 0.0
## Configuration

- **name**: Dropout 0.0
- **group**: 3. Dropout
- **hidden_sizes**: 256, 128
- **activation**: relu
- **dropout**: 0.0
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

- **Path**: ./saved_models\dropout_0_0\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 22
- **Best evaluation loss**: 0.317222
- **Train accuracy at best epoch (%)**: 89.18
- **Evaluation accuracy (%)**: 88.40
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\dropout_0_0\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.327488
- **Train accuracy (%)**: 90.39
- **Evaluation accuracy (%)**: 88.96
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 88.4000
- **Macro Precision**: 0.8848
- **Macro Recall**: 0.8840
- **Macro F1-score**: 0.8840
- **Number of parameters**: 235,146
- **Training time (s)**: 1403.62
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.8323    0.8240    0.8281      1000
     Trouser     0.9888    0.9740    0.9814      1000
    Pullover     0.8030    0.7950    0.7990      1000
       Dress     0.9000    0.9000    0.9000      1000
        Coat     0.7477    0.8300    0.7867      1000
      Sandal     0.9776    0.9600    0.9687      1000
       Shirt     0.7114    0.6680    0.6890      1000
     Sneaker     0.9364    0.9710    0.9534      1000
         Bag     0.9837    0.9640    0.9737      1000
  Ankle boot     0.9666    0.9540    0.9602      1000

    accuracy                         0.8840     10000
   macro avg     0.8848    0.8840    0.8840     10000
weighted avg     0.8848    0.8840    0.8840     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\dropout_0_0_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\dropout_0_0_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\dropout_0_0_sample_predictions.png
