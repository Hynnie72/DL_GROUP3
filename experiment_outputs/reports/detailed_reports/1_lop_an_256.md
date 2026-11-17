# Detailed Analysis: 1 lớp ẩn (256)
## Configuration

- **name**: 1 lớp ẩn (256)
- **group**: 1. Số lớp ẩn
- **hidden_sizes**: 256
- **activation**: relu
- **dropout**: 0.2
- **batch_norm**: False
- **optimizer**: adam
- **lr**: 0.001
- **momentum**: 0.9
- **batch_size**: 64
- **epochs**: 35
- **seed**: 42
- **architecture**: 784 -> 256 -> 10
- **optimizer_text**: Adam (lr=0.001)
## BEST checkpoint (lowest evaluation loss)

- **Path**: ./saved_models\1_lop_an_256\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 34
- **Best evaluation loss**: 0.335842
- **Train accuracy at best epoch (%)**: 87.57
- **Evaluation accuracy (%)**: 88.45
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\1_lop_an_256\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.343778
- **Train accuracy (%)**: 87.56
- **Evaluation accuracy (%)**: 88.45
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 88.4500
- **Macro Precision**: 0.8837
- **Macro Recall**: 0.8845
- **Macro F1-score**: 0.8838
- **Number of parameters**: 203,530
- **Training time (s)**: 1490.47
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.8345    0.8320    0.8332      1000
     Trouser     0.9741    0.9770    0.9755      1000
    Pullover     0.8020    0.8100    0.8060      1000
       Dress     0.8507    0.9230    0.8854      1000
        Coat     0.8138    0.7780    0.7955      1000
      Sandal     0.9865    0.9520    0.9690      1000
       Shirt     0.7158    0.6750    0.6948      1000
     Sneaker     0.9334    0.9670    0.9499      1000
         Bag     0.9710    0.9710    0.9710      1000
  Ankle boot     0.9552    0.9600    0.9576      1000

    accuracy                         0.8845     10000
   macro avg     0.8837    0.8845    0.8838     10000
weighted avg     0.8837    0.8845    0.8838     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\1_lop_an_256_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\1_lop_an_256_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\1_lop_an_256_sample_predictions.png
