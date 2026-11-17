# Detailed Analysis: Hẹp: 2 lớp ẩn (64-32)
## Configuration

- **name**: Hẹp: 2 lớp ẩn (64-32)
- **group**: 2. Độ rộng lớp ẩn
- **hidden_sizes**: 64, 32
- **activation**: relu
- **dropout**: 0.2
- **batch_norm**: False
- **optimizer**: adam
- **lr**: 0.001
- **momentum**: 0.9
- **batch_size**: 64
- **epochs**: 35
- **seed**: 42
- **architecture**: 784 -> 64 -> 32 -> 10
- **optimizer_text**: Adam (lr=0.001)
## BEST checkpoint (lowest evaluation loss)

- **Path**: ./saved_models\hep_2_lop_an_64_32\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 31
- **Best evaluation loss**: 0.362571
- **Train accuracy at best epoch (%)**: 84.15
- **Evaluation accuracy (%)**: 86.77
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\hep_2_lop_an_64_32\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.363838
- **Train accuracy (%)**: 84.18
- **Evaluation accuracy (%)**: 86.53
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 86.7700
- **Macro Precision**: 0.8669
- **Macro Recall**: 0.8677
- **Macro F1-score**: 0.8662
- **Number of parameters**: 52,650
- **Training time (s)**: 1491.26
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.7778    0.8260    0.8012      1000
     Trouser     0.9898    0.9660    0.9777      1000
    Pullover     0.7502    0.8140    0.7808      1000
       Dress     0.8663    0.8750    0.8706      1000
        Coat     0.7728    0.8060    0.7890      1000
      Sandal     0.9752    0.9430    0.9588      1000
       Shirt     0.7123    0.5720    0.6345      1000
     Sneaker     0.9306    0.9390    0.9348      1000
         Bag     0.9577    0.9740    0.9658      1000
  Ankle boot     0.9358    0.9620    0.9487      1000

    accuracy                         0.8677     10000
   macro avg     0.8669    0.8677    0.8662     10000
weighted avg     0.8669    0.8677    0.8662     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\hep_2_lop_an_64_32_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\hep_2_lop_an_64_32_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\hep_2_lop_an_64_32_sample_predictions.png
