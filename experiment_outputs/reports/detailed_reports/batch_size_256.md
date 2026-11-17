# Detailed Analysis: Batch size 256
## Configuration

- **name**: Batch size 256
- **group**: 7. Batch size
- **hidden_sizes**: 256, 128
- **activation**: relu
- **dropout**: 0.2
- **batch_norm**: False
- **optimizer**: adam
- **lr**: 0.001
- **momentum**: 0.9
- **batch_size**: 256
- **epochs**: 35
- **seed**: 42
- **architecture**: 784 -> 256 -> 128 -> 10
- **optimizer_text**: Adam (lr=0.001)
## BEST checkpoint (lowest evaluation loss)

- **Path**: ./saved_models\batch_size_256\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 28
- **Best evaluation loss**: 0.302868
- **Train accuracy at best epoch (%)**: 87.90
- **Evaluation accuracy (%)**: 89.16
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\batch_size_256\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.305309
- **Train accuracy (%)**: 88.35
- **Evaluation accuracy (%)**: 88.78
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 89.1600
- **Macro Precision**: 0.8907
- **Macro Recall**: 0.8916
- **Macro F1-score**: 0.8905
- **Number of parameters**: 235,146
- **Training time (s)**: 3292.53
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.8197    0.8640    0.8413      1000
     Trouser     0.9849    0.9770    0.9809      1000
    Pullover     0.8167    0.8200    0.8184      1000
       Dress     0.8738    0.9210    0.8968      1000
        Coat     0.8027    0.8260    0.8142      1000
      Sandal     0.9875    0.9500    0.9684      1000
       Shirt     0.7579    0.6510    0.7004      1000
     Sneaker     0.9334    0.9670    0.9499      1000
         Bag     0.9750    0.9750    0.9750      1000
  Ankle boot     0.9554    0.9650    0.9602      1000

    accuracy                         0.8916     10000
   macro avg     0.8907    0.8916    0.8905     10000
weighted avg     0.8907    0.8916    0.8905     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\batch_size_256_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\batch_size_256_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\batch_size_256_sample_predictions.png
