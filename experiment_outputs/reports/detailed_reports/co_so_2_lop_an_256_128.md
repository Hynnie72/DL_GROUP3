# Detailed Analysis: Cơ sở: 2 lớp ẩn (256-128)
## Configuration

- **name**: Cơ sở: 2 lớp ẩn (256-128)
- **group**: 1. Số lớp ẩn
- **hidden_sizes**: 256, 128
- **activation**: relu
- **dropout**: 0.2
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

- **Path**: ./saved_models\co_so_2_lop_an_256_128\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 34
- **Best evaluation loss**: 0.312283
- **Train accuracy at best epoch (%)**: 87.53
- **Evaluation accuracy (%)**: 88.70
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\co_so_2_lop_an_256_128\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.318632
- **Train accuracy (%)**: 87.67
- **Evaluation accuracy (%)**: 88.61
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 88.7000
- **Macro Precision**: 0.8862
- **Macro Recall**: 0.8870
- **Macro F1-score**: 0.8862
- **Number of parameters**: 235,146
- **Training time (s)**: 1436.53
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.8478    0.8300    0.8388      1000
     Trouser     0.9868    0.9750    0.9809      1000
    Pullover     0.8085    0.8020    0.8052      1000
       Dress     0.8565    0.9250    0.8894      1000
        Coat     0.7787    0.8130    0.7955      1000
      Sandal     0.9756    0.9580    0.9667      1000
       Shirt     0.7384    0.6660    0.7003      1000
     Sneaker     0.9423    0.9630    0.9525      1000
         Bag     0.9721    0.9740    0.9730      1000
  Ankle boot     0.9554    0.9640    0.9597      1000

    accuracy                         0.8870     10000
   macro avg     0.8862    0.8870    0.8862     10000
weighted avg     0.8862    0.8870    0.8862     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\co_so_2_lop_an_256_128_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\co_so_2_lop_an_256_128_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\co_so_2_lop_an_256_128_sample_predictions.png
