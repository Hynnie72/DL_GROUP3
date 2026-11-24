# CourseWork: Medical Image Regression -- Dự đoán tuổi từ ảnh X-quang ngực

## 1. Giới thiệu đề tài

Đề tài xây dựng mô hình học sâu cho bài toán **hồi quy trên ảnh y tế
(Medical Image Regression)**, với mục tiêu dự đoán tuổi bệnh nhân từ ảnh
X-quang ngực thuộc bộ dữ liệu **NIH ChestX-ray14 (CXR8)**.

Khác với bài toán phân loại, đầu ra của mô hình là một giá trị liên tục
biểu diễn tuổi dự đoán. Ngoài thông tin hình ảnh, mô hình còn sử dụng
một số metadata của bệnh nhân nhằm bổ sung thông tin cho quá trình dự
đoán.

**Bài toán:** Age Regression\
**Đầu vào:** Ảnh X-quang ngực và metadata bệnh nhân\
**Đầu ra:** Một giá trị tuổi dự đoán\
**Kiến trúc chính:** EfficientNetV2-S + MLP + Gating Fusion + Regression
Head

## 2. Bộ dữ liệu

### NIH ChestX-ray14 (CXR8)

NIH ChestX-ray14 là bộ dữ liệu ảnh X-quang ngực quy mô lớn, được sử dụng
trong nghiên cứu học máy và học sâu cho ảnh y tế. Trong Coursework này,
dữ liệu được sử dụng cho mục tiêu **dự đoán tuổi bệnh nhân**, không phải
phân loại các bệnh lý trong ảnh.

Nguồn dữ liệu của dự án gồm:

-   Các ảnh X-quang ngực.
-   Tệp metadata `Data_Entry_2017_v2020.csv`.

Một số trường thông tin trong CSV:

  -----------------------------------------------------------------------
  Trường                  Ý nghĩa                 Vai trò trong dự án
  ----------------------- ----------------------- -----------------------
  `Image Index`           Tên tệp ảnh             Ghép metadata với ảnh

  `Patient ID`            Mã bệnh nhân            Nhận diện bệnh nhân

  `Patient Age`           Tuổi bệnh nhân          Nhãn mục tiêu (target)

  `Patient Sex`           Giới tính               Metadata đầu vào

  `View Position`         Tư thế chụp (ví dụ      Metadata đầu vào
                          AP/PA)                  

  `Finding Labels`        Nhãn phát hiện trong    Thông tin tham khảo,
                          ảnh                     không đưa vào đầu vào
                                                  mô hình
  -----------------------------------------------------------------------

Trong thiết kế hiện tại, mô hình sử dụng **Patient Age** làm giá trị cần
dự đoán; **Patient Sex** và **View Position** làm metadata đầu vào. Ảnh
X-quang được xử lý như đầu vào hình ảnh của mạng CNN.

> Lưu ý: `Patient Age` là nhãn mục tiêu, không được sử dụng như một đặc
> trưng đầu vào. `Finding Labels` sẽ được Multi-Hot Encoding thành 14 trường và tuyền vào nhánh MLP.

## 3. Ý tưởng và kiến trúc mô hình

Mô hình được xây dựng theo hướng **học đặc trưng đa phương thức
(multimodal feature learning)**: trích xuất đặc trưng từ ảnh X-quang và
metadata bệnh nhân bằng hai nhánh riêng, sau đó kết hợp chúng thông qua
cơ chế Gating để dự đoán tuổi.

### 3.1. Nhánh trích xuất đặc trưng ảnh -- EfficientNetV2-S

Ảnh X-quang được đưa vào backbone **EfficientNetV2-S** để học các đặc
trưng hình ảnh. Backbone được chia thành các giai đoạn đầu và các giai
đoạn còn lại:

-   Các stage đầu (0--2) trích xuất đặc trưng ảnh ban đầu; đặc trưng
    được đưa qua Adaptive Average Pooling để tạo vector đặc trưng.
-   Các stage còn lại tiếp tục xử lý đặc trưng hình ảnh đã được điều
    kiện hóa bởi thông tin hợp nhất.
-   Global Average Pooling được áp dụng ở cuối backbone để chuyển
    feature map thành vector đặc trưng ảnh.

Mô hình hỗ trợ đầu vào ảnh một kênh (grayscale), phù hợp với ảnh
X-quang.

### 3.2. Nhánh metadata -- MLP

Metadata gồm **Patient Sex** và **View Position** được mã hóa thành
vector số và đưa qua một MLP gồm các lớp tuyến tính, Batch
Normalization, ReLU và Dropout.

Nhánh này chuyển metadata đầu vào thành embedding có số chiều lớn hơn,
giúp mô hình học biểu diễn đặc trưng metadata thay vì sử dụng trực tiếp
các giá trị mã hóa ban đầu.

### 3.3. Kết hợp đặc trưng bằng Gating

Mô hình sử dụng **Mid-level Fusion với Gating Mechanism**.

Embedding metadata được đưa qua một phép chiếu tuyến tính và hàm Sigmoid
để tạo vector cổng (gate). Vector này điều chỉnh theo từng kênh của đặc
trưng ảnh ở các stage đầu. Đặc trưng ảnh sau điều chỉnh được nối với
embedding metadata và đưa qua MLP để tạo vector đặc trưng hợp nhất.

Vector hợp nhất được giữ lại, trong khi nhánh ảnh tiếp tục đi qua các
stage còn lại của EfficientNetV2-S. Ở cuối mạng, vector đặc trưng ảnh và
vector hợp nhất được nối với nhau.

### 3.4. Regression Head

Vector đặc trưng cuối cùng được đưa qua các lớp Fully Connected có
Dropout để tạo đầu ra một chiều. Đầu ra là **tuổi dự đoán dạng số
thực**, không qua hàm kích hoạt đầu ra.

## 4. Cấu trúc mã nguồn

Cấu trúc thư mục chính của dự án:

``` text
src/
├── __init__.py
├── dataset.py
├── model.py
├── scratch_check.py
├── scratch_labels.py
├── smoke_test.py
├── train.py
└── utils.py
```

  -----------------------------------------------------------------------
  Tệp                                 Chức năng
  ----------------------------------- -----------------------------------
  `__init__.py`                       Khai báo package Python

  `dataset.py`                        Đọc dữ liệu ảnh và metadata, xử lý
                                      và tạo Dataset/DataLoader

  `model.py`                          Định nghĩa kiến trúc
                                      MetadataAwareEfficientNetV2

  `scratch_check.py`                  Script kiểm tra dữ liệu hoặc thành
                                      phần trong quá trình phát triển

  `scratch_labels.py`                 Script kiểm tra/xử lý liên quan đến
                                      nhãn

  `smoke_test.py`                     Kiểm tra nhanh khả năng khởi tạo và
                                      chạy mô hình

  `train.py`                          Điều phối quá trình huấn luyện và
                                      đánh giá

  `utils.py`                          Các hàm tiện ích dùng chung
  -----------------------------------------------------------------------

Tên tệp và vai trò trong bảng được mô tả theo mục đích thể hiện qua tên
gọi; chi tiết triển khai có thể được cập nhật theo mã nguồn thực tế.

## 5. Phương pháp đánh giá

Mô hình là bài toán hồi quy, do đó kết quả được đánh giá bằng các chỉ số
sau:

  -----------------------------------------------------------------------
  Chỉ số                              Ý nghĩa
  ----------------------------------- -----------------------------------
  **MAE**                             Sai số tuyệt đối trung bình giữa
                                      tuổi dự đoán và tuổi thực tế

  **MSE**                             Trung bình bình phương sai số; phạt
                                      mạnh hơn các sai số lớn

  **RMSE**                            Căn bậc hai của MSE, cùng đơn vị
                                      với tuổi

  **MAPE**                            Sai số phần trăm tuyệt đối trung
                                      bình

  **R²**                              Hệ số xác định, phản ánh mức độ
                                      biến thiên của nhãn được mô hình
                                      giải thích
  -----------------------------------------------------------------------

Với (y_i) là tuổi thực tế, (`\hat{y}`{=tex}\_i) là tuổi dự đoán và (N)
là số mẫu đánh giá:

\[ MAE =
`\frac{1}{N}`{=tex}`\sum`{=tex}\_{i=1}\^{N}\|y_i-`\hat{y}`{=tex}\_i\| \]

\[ MSE =
`\frac{1}{N}`{=tex}`\sum`{=tex}\_{i=1}\^{N}(y_i-`\hat{y}`{=tex}\_i)\^2
\]

\[ RMSE = `\sqrt{MSE}`{=tex} \]

\[ MAPE = `\frac{100}{N}`{=tex}`\sum`{=tex}\_{i=1}\^{N}
`\left`{=tex}\|`\frac{y_i-\hat{y}_i}{y_i}`{=tex}`\right`{=tex}\| \]

\[ R\^2 = 1 - `\frac{\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}`{=tex}
{`\sum`{=tex}\_{i=1}^{N}(y_i-`\bar`{=tex}{y})^2} \]

Trong đó (`\bar`{=tex}{y}) là tuổi thực tế trung bình. Khi tính MAPE,
cần xử lý riêng các mẫu có (y_i=0) để tránh phép chia cho 0; cần ghi rõ
cách xử lý trong báo cáo.

### Bảng kết quả đánh giá

Điền các giá trị được script đánh giá trên tập test:

  Metric        Giá trị Đơn vị
  -------- ------------ --------
  MAE         `{{MAE}}` năm
  MSE         `{{MSE}}` năm²
  RMSE       `{{RMSE}}` năm
  MAPE       `{{MAPE}}` \%
  R²           `{{R2}}` ---

**Lưu ý:** RMSE là tên viết tắt chuẩn của Root Mean Squared Error. Nếu
script hiện tại xuất trường `RMAE`, cần kiểm tra định nghĩa trong mã
nguồn trước khi đưa vào báo cáo; không nên tự động xem `RMAE` là `RMSE`.

## 6. Hiệu suất và mức sử dụng phần cứng

Bên cạnh chất lượng dự đoán, báo cáo ghi nhận hiệu suất thực thi và mức
sử dụng tài nguyên phần cứng trong quá trình huấn luyện/đánh giá.

Các thông tin nên được script tự động thu thập:

  -----------------------------------------------------------------------
  Nhóm                                Thông tin cần ghi nhận
  ----------------------------------- -----------------------------------
  Thiết bị                            Tên CPU, GPU và loại thiết bị chạy
                                      mô hình

  Bộ nhớ GPU                          VRAM tổng, VRAM đã sử dụng và VRAM
                                      cực đại trong quá trình chạy

  Bộ nhớ hệ thống                     RAM tổng và RAM sử dụng/cực đại nếu
                                      hệ điều hành hoặc công cụ đo hỗ trợ

  Tải phần cứng                       Mức sử dụng GPU, CPU trong quá
                                      trình chạy

  Thời gian                           Tổng thời gian huấn luyện, thời
                                      gian mỗi epoch và thời gian đánh
                                      giá

  Tốc độ xử lý                        Số mẫu xử lý mỗi giây
                                      (samples/second), nếu có đo

  Cấu hình chạy                       Batch size, số epoch, kích thước
                                      ảnh và số workers
  -----------------------------------------------------------------------

### Bảng hiệu suất phần cứng

  Thông số                                         Giá trị
  -------------------------------- -----------------------
  CPU                                            `{{CPU}}`
  GPU                                            `{{GPU}}`
  VRAM cực đại                             `{{PEAK_VRAM}}`
  RAM cực đại                               `{{PEAK_RAM}}`
  GPU utilization                    `{{GPU_UTILIZATION}}`
  CPU utilization                    `{{CPU_UTILIZATION}}`
  Tổng thời gian huấn luyện            `{{TRAINING_TIME}}`
  Thời gian mỗi epoch trung bình          `{{EPOCH_TIME}}`
  Tốc độ xử lý                            `{{THROUGHPUT}}`

Các giá trị phần cứng cần được đo trong lúc chương trình chạy. Nếu không
có công cụ hoặc log tương ứng, ghi `N/A` thay vì ước lượng.

## 7. Tổng kết

Coursework triển khai bài toán dự đoán tuổi từ ảnh X-quang ngực bằng mô
hình hồi quy đa phương thức. EfficientNetV2-S đảm nhiệm việc trích xuất
đặc trưng hình ảnh, MLP mã hóa metadata bệnh nhân, còn Gating Mechanism
cho phép metadata điều chỉnh đặc trưng ảnh trước khi tạo biểu diễn hợp
nhất. Regression Head sử dụng biểu diễn cuối cùng để dự đoán tuổi.

Báo cáo đánh giá mô hình trên tập test bằng MAE, MSE, RMSE, MAPE và R²;
đồng thời ghi nhận thời gian thực thi, tốc độ xử lý và mức sử dụng CPU,
RAM, GPU, VRAM. Các kết quả định lượng được lấy trực tiếp từ log hoặc
script đánh giá, không điền bằng giá trị giả định.
