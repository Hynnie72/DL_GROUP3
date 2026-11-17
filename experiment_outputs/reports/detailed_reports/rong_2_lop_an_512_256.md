# Detailed Analysis: Rộng: 2 lớp ẩn (512-256)
## Configuration

- **name**: Rộng: 2 lớp ẩn (512-256)
- **group**: 2. Độ rộng lớp ẩn
- **hidden_sizes**: 512, 256
- **activation**: relu
- **dropout**: 0.2
- **batch_norm**: False
- **optimizer**: adam
- **lr**: 0.001
- **momentum**: 0.9
- **batch_size**: 64
- **epochs**: 35
- **seed**: 42
- **architecture**: 784 -> 512 -> 256 -> 10
- **optimizer_text**: Adam (lr=0.001)
## BEST checkpoint (lowest evaluation loss)

- **Path**: ./saved_models\rong_2_lop_an_512_256\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 30
- **Best evaluation loss**: 0.314723
- **Train accuracy at best epoch (%)**: 87.68
- **Evaluation accuracy (%)**: 88.67
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\rong_2_lop_an_512_256\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.326291
- **Train accuracy (%)**: 88.11
- **Evaluation accuracy (%)**: 88.29
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 88.6700
- **Macro Precision**: 0.8875
- **Macro Recall**: 0.8867
- **Macro F1-score**: 0.8869
- **Number of parameters**: 535,818
- **Training time (s)**: 1523.13
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.8378    0.8060    0.8216      1000
     Trouser     0.9939    0.9710    0.9823      1000
    Pullover     0.8037    0.8150    0.8093      1000
       Dress     0.8738    0.9070    0.8901      1000
        Coat     0.8055    0.7950    0.8002      1000
      Sandal     0.9807    0.9650    0.9728      1000
       Shirt     0.6911    0.7070    0.6990      1000
     Sneaker     0.9321    0.9740    0.9526      1000
         Bag     0.9809    0.9740    0.9774      1000
  Ankle boot     0.9754    0.9530    0.9641      1000

    accuracy                         0.8867     10000
   macro avg     0.8875    0.8867    0.8869     10000
weighted avg     0.8875    0.8867    0.8869     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\rong_2_lop_an_512_256_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\rong_2_lop_an_512_256_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\rong_2_lop_an_512_256_sample_predictions.png
