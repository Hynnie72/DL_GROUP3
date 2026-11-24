# Lab 3: Deep Learning — Thực Hành Với Hugging Face Transformers & Fine-tuning Mô Hình Phân Loại Cảm Xúc (Sentiment Analysis)

---

## 📌 1. Giới Thiệu Chung & Mục Tiêu Bài Lab

Bài thực hành **Lab 3 (Practice 3: Get Started with Hugging Face)** tập trung vào việc tiếp cận và làm chủ hệ sinh thái **Hugging Face Transformers** trong bài toán Xử lý ngôn ngữ tự nhiên (NLP) thực tế: **Phân loại cảm xúc văn bản (Sentiment Analysis)** thành 2 lớp `POSITIVE` (Tích cực) hoặc `NEGATIVE` (Tiêu cực).

### Mục tiêu chính:
1. **Làm chủ quy trình NLP hiện đại với Hugging Face**: Nắm vững các khái niệm then chốt như Subword Tokenization (WordPiece), cơ chế Attention Mask, Special Tokens (`[CLS]`, `[SEP]`), và biểu diễn dữ liệu dạng PyTorch Tensor.
2. **Khám phá và ứng dụng Pre-trained Models**: Trải nghiệm quy trình suy luận (Inference-only) với mô hình đã được huấn luyện sẵn trên Hugging Face Hub mà không cần huấn luyện lại.
3. **Fine-tuning mô hình ngôn ngữ lớn (LLM/SLM) với tài nguyên tối ưu**: Thực hiện tinh chỉnh mô hình **DistilBERT** trên tập dữ liệu đánh giá phim lớn (**IMDb Dataset — 50.000 mẫu**) áp dụng kỹ thuật **Freeze Backbone** (đóng băng phần trích xuất đặc trưng, chỉ huấn luyện classifier head) nhằm tiết kiệm chi phí tính toán và chống hiện tượng Catastrophic Forgetting.
4. **Chuẩn hóa kiến trúc phần mềm mã nguồn**: Xây dựng toàn bộ mã nguồn theo chuẩn **Object-Oriented Programming (OOP)**, module hóa cao, tách bạch rõ ràng giữa xử lý dữ liệu, định nghĩa mô hình, huấn luyện, đánh giá, cấu hình qua file YAML và hỗ trợ giao diện dòng lệnh (CLI).

---

## 🤖 2. Hai Mô Hình Được Sử Dụng & Fine-Tune

Dự án triển khai và khảo sát 2 mô hình Transformer dựa trên kiến trúc DistilBERT:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        KIẾN TRÚC DISTILBERT                            │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Embeddings : Word(30522, 768) + Position(512, 768) + LayerNorm      │
│ 2. Transformer: 6 Transformer Encoder Layers (12 Attention Heads, 768d)│
│ 3. Head       : Pre-classifier Linear(768->768) + ReLU                 │
│                 + Classifier Linear(768->2)                            │
└────────────────────────────────────────────────────────────────────────┘
```

### 🔹 Mô hình 1: `distilbert-base-uncased-finetuned-sst-2-english` (Exercise 1)
- **Mục đích**: Chạy suy luận (Inference-only) xác định cảm xúc câu văn bản tiếng Anh.
- **Nguồn gốc**: Được Hugging Face tiền huấn luyện và fine-tune sẵn trên tập dữ liệu học thuật **SST-2 (Stanford Sentiment Treebank v2)**.
- **Cơ chế hoạt động**:
  - Chạy ở chế độ `eval()` kết hợp với context manager `torch.no_grad()`.
  - **Không cập nhật bất kỳ trọng số nào** (Weight Update = None).
  - Đầu vào: Câu văn bản đơn lẻ $\rightarrow$ Tokenizer $\rightarrow$ Model $\rightarrow$ Softmax $\rightarrow$ Label & Xác suất tự tin (Confidence).
- **Tổng số tham số**: ~66.955.010 (~66.9M params).

### 🔹 Mô hình 2: `distilbert-base-uncased` Fine-tuned trên IMDb (Exercise 2)
- **Mục đích**: Tinh chỉnh (Fine-tuning) để phân loại 50.000 bài đánh giá phim thực tế từ IMDb.
- **Backbone gốc**: Pre-trained trên tập dữ liệu tổng quát BookCorpus (800M từ) và English Wikipedia (2.500M từ).
- **Chiến lược huấn luyện (Freeze Backbone Strategy)**:
  - **Đóng băng (Freeze)** toàn bộ khối Embeddings và 6 tầng Transformer (`requires_grad = False`): chiếm **66.362.880 tham số (99.12%)**.
  - **Chỉ huấn luyện (Trainable)** phần Classification Head (`pre_classifier` và `classifier`): chỉ **592.130 tham số (0.88%)**.
- **Lý do lựa chọn Freeze Backbone**:
  - Backbone DistilBERT đã học sẵn biểu diễn ngữ nghĩa tiếng Anh phong phú, việc giữ nguyên backbone giúp giảm thiểu nguy cơ **Catastrophic Forgetting** và **Overfitting**.
  - Giảm ~100 lần số tham số cần cập nhật đạo hàm, tăng tốc độ huấn luyện khoảng **3x** so với Full Fine-tuning, phù hợp với tài nguyên phần cứng vừa và nhỏ (Colab GPU T4 hoặc CPU).
- **Trọng số đã huấn luyện hoàn chỉnh**: Đã được lưu trữ và đóng gói độc lập tại thư mục `saved_model/` (đạt **Test Accuracy: 91.57%**, **Test F1: 91.66%**).

---

## 🗂️ 3. Cấu Trúc Dự Án (Project Structure)

Dự án được tổ chức khoa học theo mô hình module hóa chuyên nghiệp:

```text
Lab3_DL/
│
├── configs/
│   └── training_config.yaml         # File cấu hình tập trung (Hyperparameters, Paths, Split)
│
├── data/
│   └── IMDB Dataset.csv             # Bộ dữ liệu 50,000 bài đánh giá phim IMDb
│
├── src/                             # Package mã nguồn chính (Core Modules)
│   ├── __init__.py                  # Expose public API của package
│   ├── config.py                    # Dataclass & Loader đọc file YAML
│   ├── text_processor.py            # Kiểm tra & tiền xử lý chuỗi văn bản đầu vào
│   ├── tokenizer.py                 # Wrapper Tokenizer cho Exercise 1
│   ├── sentiment_analyzer.py        # Pipeline suy luận hướng đối tượng (Exercise 1)
│   ├── dataset.py                   # IMDb DataModule: load, stratified split, tokenize (Ex 2)
│   ├── model.py                     # DistilBERT model builder & logic freeze backbone (Ex 2)
│   ├── trainer.py                   # Hugging Face Trainer wrapper & EpochMetricsCallback (Ex 2)
│   ├── evaluator.py                 # Đánh giá độc lập trên Test Set & vẽ Confusion Matrix
│   └── result_saver.py              # Xuất kết quả có cấu trúc (JSON, CSV, PNG)
│
├── notebooks/ (Gốc thư mục)
│   ├── Ex1_SentimentAnalysis.ipynb  # Notebook thực hành bài tập 1
│   ├── Ex2_Finetuning.ipynb         # Notebook thực hành bài tập 2
│   └── Lab3_Practice3.ipynb         # Notebook tổng hợp báo cáo chi tiết kèm biểu đồ
│
├── scripts/
│   ├── smoke_run.py                 # Kịch bản kiểm thử nhanh pipeline (24 checks verification)
│   └── generate_report.py           # Script tự động xuất báo cáo PDF (dùng fpdf2)
│
├── saved_model/                     # Model DistilBERT đã fine-tune hoàn chỉnh
│   ├── model.safetensors            # Trọng số mô hình (đã fine-tune)
│   ├── config.json                  # Cấu hình kiến trúc mô hình
│   ├── tokenizer.json               # Cấu hình Tokenizer & từ vựng
│   ├── tokenizer_config.json        # Thuộc tính tokenizer
│   ├── special_tokens_map.json      # Bản đồ token đặc biệt
│   └── vocab.txt                    # Bộ từ vựng WordPiece
│
├── artifacts/
│   └── figures/                     # Lưu trữ hình ảnh, biểu đồ trực quan
│       ├── 20261110_214819_confusion_matrix.png
│       └── 20261110_214820_training_curves.png
│
├── reports/                         # Lưu trữ kết quả đo lường và báo cáo
│   ├── test_metrics.json            # Chỉ số chi tiết trên tập test (Acc, F1, Precision, Recall)
│   ├── training_history.csv         # Lịch sử Loss và Accuracy qua các Epoch
│   └── 20261110_215250_lab3_report.pdf # File PDF báo cáo thí nghiệm hoàn chỉnh
│
├── docs/                            # Tài liệu kỹ thuật chi tiết
│   ├── lab3_description.md          # Tổng quan yêu cầu bài lab
│   ├── data.md                      # Đặc tả tập dữ liệu & quy trình tiền xử lý
│   ├── model.md                     # Phân tích kiến trúc mô hình & số lượng tham số
│   ├── training.md                  # Quy trình huấn luyện & hyperparameters
│   ├── evaluation.md                # Phân tích chi tiết kết quả đánh giá & ma trận nhầm lẫn
│   └── demo.md                      # Hướng dẫn chạy & thử nghiệm
│
├── tests/
│   └── test_pipeline.py             # Kiểm thử tự động (Unit Test / PyTest)
│
├── main.py                          # CLI Entry Point đa chức năng (inference, train, eval)
├── requirements.txt                 # Danh sách thư viện phụ thuộc
└── README.md                        # Tài liệu hướng dẫn dự án (File này)
```

---

## ⚙️ 4. Các Modules Đã Xây Dựng Và Triển Khai

Toàn bộ nghiệp vụ được trừu tượng hóa và đóng gói thành các module độc lập trong thư mục `src/`:

| Module | Lớp / Đối tượng chính | Nhiệm vụ & Vai trò triển khai |
|---|---|---|
| `config.py` | `InferenceConfig`<br>`TrainingConfig` | Quản lý tham số có cấu trúc bằng Python `@dataclass`. Hỗ trợ đọc ghi từ file `configs/training_config.yaml`, tránh hardcode các đường dẫn và siêu tham số. |
| `text_processor.py` | `TextProcessor` | Xác thực tính hợp lệ của chuỗi văn bản đầu vào: kiểm tra kiểu chuỗi (`str`), loại bỏ khoảng trắng thừa (`strip()`), và chặn chuỗi rỗng trước khi gửi sang tokenizer. |
| `tokenizer.py` | `TextTokenizer` | Bao bọc `AutoTokenizer` của Hugging Face cho bài toán Exercise 1: hỗ trợ trích xuất special tokens (`[CLS]`, `[SEP]`), encode chuỗi thành tensor PyTorch, và giải mã token subword. |
| `sentiment_analyzer.py`| `SentimentAnalyzer`<br>`SentimentResult` | Triển khai suy luận cảm xúc cho Exercise 1. Thực hiện forward pass không tính gradient, áp dụng Softmax và trả về dataclass bất biến `SentimentResult` gồm nhãn dự đoán, xác suất tự tin (confidence) và chi tiết phân phối xác suất. |
| `dataset.py` | `IMDbDataModule` | Tải dữ liệu từ `IMDB Dataset.csv`, mã hóa nhãn (`negative: 0, positive: 1`), thực hiện **Stratified Split** bảo toàn tỷ lệ nhãn (70% Train — 15% Validation — 15% Test với seed cố định = 42). Tích hợp tokenization theo batch với `max_length=256`. |
| `model.py` | `TextClassificationModel` | Khởi tạo mô hình phân loại chuỗi. Cài đặt thuật toán đóng băng thông minh: nếu `freeze_strategy="backbone"`, lặp qua toàn bộ tham số của `distilbert` và đặt `param.requires_grad = False`, chỉ để lại `pre_classifier` và `classifier` ở trạng thái trainable. |
| `trainer.py` | `FineTuningTrainer`<br>`EpochMetricsCallback`<br>`compute_metrics` | Tích hợp lớp `Trainer` từ Hugging Face. Bổ sung `compute_metrics` (tính Accuracy, Precision, Recall, F1 bằng `scikit-learn`) và Custom Callback in chi tiết thông số huấn luyện / thẩm định sau mỗi epoch. |
| `evaluator.py` | `ModelEvaluator` | Độc lập đánh giá mô hình đã huấn luyện trên tập Test 7.500 mẫu chưa từng thấy (Unseen Data). Hỗ trợ dự đoán hàng loạt và vẽ biểu đồ **Confusion Matrix** trực quan. |
| `result_saver.py` | `ResultSaver` | Phụ trách lưu trữ đầu ra có cấu trúc: xuất kết quả phân loại ra file JSON, CSV trong `reports/`, đồng thời lưu đồ thị đường cong học tập và confusion matrix vào `artifacts/figures/`. |

---

## 📊 5. Quy Trình Dữ Liệu & Kết Quả Thực Nghiệm

### 5.1. Phân chia tập dữ liệu (Stratified Split)
- **Tập dữ liệu**: IMDb Dataset (50.000 bài đánh giá phân bổ đều: 25.000 Positive, 25.000 Negative).
- **Tỷ lệ phân chia**:
  - **Train set (70%)**: 34.999 mẫu — dùng cập nhật trọng số classification head.
  - **Validation set (15%)**: 7.501 mẫu — kiểm soát hiện tượng overfitting qua từng epoch.
  - **Test set (15%)**: 7.500 mẫu — **giữ độc lập hoàn toàn**, chỉ đánh giá 1 lần duy nhất sau khi hoàn tất training.

### 5.2. Kết quả huấn luyện (2 Epochs, Freeze Backbone)

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Val F1-score |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.1547 | 94.42% | 0.2629 | 90.68% | 91.08% |
| **2** | 0.1045 | 97.33% | 0.3223 | 91.75% | 91.80% |

### 5.3. Kết quả đánh giá trên tập Test độc lập (7.500 mẫu)

```json
{
    "test_loss": 0.3274,
    "test_accuracy": 0.9157,
    "test_precision": 0.9077,
    "test_recall": 0.9256,
    "test_f1": 0.9166
}
```

- **Độ chính xác (Test Accuracy)**: **91.57%** (Vượt mục tiêu đề ra $\ge 90\%$).
- **F1-score (Test F1)**: **91.66%**.
- **Kiểm soát Data Leakage**: Độ chênh lệch giữa Validation Accuracy (91.75%) và Test Accuracy (91.57%) chỉ là **0.18%** ($< 1\%$), khẳng định mô hình có khả năng tổng quát hóa (generalization) rất tốt trên dữ liệu mới mà không bị rò rỉ dữ liệu.

---

## 🚀 6. Hướng Dẫn Cài Đặt & Triển Khai (How to Run)

### Bước 1: Chuẩn bị môi trường & cài đặt thư viện
Yêu cầu Python $\ge 3.10$. Khuyến nghị sử dụng môi trường ảo (`venv` hoặc `conda`):

```bash
# Tạo và kích hoạt môi trường ảo (tùy chọn)
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

# Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
```

### Bước 2: Kiểm thử tính toàn vẹn hệ thống (Smoke Run & Unit Test)
Chạy kịch bản kiểm tra nhanh 24 điều kiện kỹ thuật với mẫu 200 dòng dữ liệu:

```bash
# Chạy smoke test (hoàn thành trong < 1 phút)
python scripts/smoke_run.py

# Hoặc chạy kiểm thử tự động với pytest
pytest tests/
```

### Bước 3: Chạy suy luận Exercise 1 qua CLI (Inference)
Chạy phân tích cảm xúc câu bất kỳ bằng mô hình pre-trained:

```bash
# Câu mặc định
python main.py --mode inference

# Câu tùy ý
python main.py --mode inference --text "The movie was absolutely marvelous with brilliant acting!"

# Lưu kết quả suy luận vào reports/
python main.py --mode inference --text "The plot was boring and predictable." --save
```

### Bước 4: Đánh giá mô hình đã Fine-tune của Exercise 2
Đánh giá trọng số mô hình đã lưu tại `saved_model/` trên tập Test (không cần train lại):

```bash
python main.py --mode eval --model saved_model/
```

*(Tùy chọn: Nếu muốn huấn luyện lại từ đầu trên toàn bộ 50.000 mẫu)*:
```bash
python main.py --mode train
```

### Bước 5: Chạy trên Jupyter Notebook
Nếu bạn muốn theo dõi trực quan từng bước dưới dạng notebook:
- Mở `Ex1_SentimentAnalysis.ipynb` để xem bài thực hành Exercise 1.
- Mở `Ex2_Finetuning.ipynb` để xem chi tiết pipeline bài thực hành Exercise 2.
- Mở `Lab3_Practice3.ipynb` để xem báo cáo tổng hợp đầy đủ kèm biểu đồ và bảng phân tích.

### Bước 6: Tự động tạo file báo cáo PDF
Hệ thống tích hợp sẵn script sinh file PDF báo cáo đẹp mắt:

```bash
python scripts/generate_report.py
```
File PDF hoàn chỉnh sẽ được lưu tại `reports/YYYYMMDD_HHMMSS_lab3_report.pdf`.

---

## 👥 Tác Giả & Bản Quyền
- **Khóa học**: Deep Learning / Xử lý ngôn ngữ tự nhiên.
- **Dự án**: Lab 3 — Practice 3: Get Started with Hugging Face.
- **Frameworks**: PyTorch, Hugging Face Transformers, Hugging Face Datasets, Scikit-learn.
