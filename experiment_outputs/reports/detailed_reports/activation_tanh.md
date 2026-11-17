# Detailed Analysis: Activation Tanh
## Configuration

- **name**: Activation Tanh
- **group**: 4. Hàm kích hoạt
- **hidden_sizes**: 256, 128
- **activation**: tanh
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

- **Path**: ./saved_models\activation_tanh\best_checkpoint.pth
- **File exists**: True
- **Best epoch**: 29
- **Best evaluation loss**: 0.346755
- **Train accuracy at best epoch (%)**: 85.61
- **Evaluation accuracy (%)**: 87.05
## LAST checkpoint (final training epoch)

- **Path**: ./saved_models\activation_tanh\last_checkpoint.pth
- **File exists**: True
- **Last epoch**: 35
- **Last evaluation loss**: 0.353761
- **Train accuracy (%)**: 85.50
- **Evaluation accuracy (%)**: 87.45
## Evaluation metrics (BEST checkpoint)

- **Accuracy (%)**: 87.0500
- **Macro Precision**: 0.8701
- **Macro Recall**: 0.8705
- **Macro F1-score**: 0.8696
- **Number of parameters**: 235,146
- **Training time (s)**: 1246.89
## Classification report (BEST checkpoint)

```text
              precision    recall  f1-score   support

 T-shirt/top     0.8241    0.8150    0.8195      1000
     Trouser     0.9918    0.9640    0.9777      1000
    Pullover     0.7948    0.7280    0.7599      1000
       Dress     0.8457    0.9150    0.8790      1000
        Coat     0.7383    0.8210    0.7775      1000
      Sandal     0.9656    0.9530    0.9592      1000
       Shirt     0.6867    0.6270    0.6555      1000
     Sneaker     0.9377    0.9490    0.9433      1000
         Bag     0.9624    0.9720    0.9672      1000
  Ankle boot     0.9543    0.9610    0.9576      1000

    accuracy                         0.8705     10000
   macro avg     0.8701    0.8705    0.8696     10000
weighted avg     0.8701    0.8705    0.8696     10000
```
## Related figures

- **Learning curves**: ..\..\figures\detailed\activation_tanh_learning_curves.png
- **Confusion matrix**: ..\..\figures\detailed\activation_tanh_confusion_matrix.png
- **Sample predictions**: ..\..\figures\detailed\activation_tanh_sample_predictions.png
