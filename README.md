# 🌤️ Weather Forecasting using Deep Learning

> **Tổng quan:** Dự án này nghiên cứu và triển khai giải pháp cho bài toán Dự báo chuỗi thời gian đa biến (**Multivariate Time Series Forecasting**). Hệ thống sử dụng dữ liệu khí hậu quá khứ để dự báo nhiệt độ trong tương lai, so sánh hiệu quả giữa các kiến trúc mạng nơ-ron: **LSTM Seq2Seq**, **LSTM Attention** và **Transformer**.

---

## 📑 Mục lục
1. [Giới thiệu bài toán](#1-giới-thiệu-bài-toán)
2. [Quy trình Xử lý Dữ liệu](#2-quy-trình-xử-lý-dữ-liệu)
3. [Kiến trúc Mô hình](#3-kiến-trúc-mô-hình)
4. [Cấu hình Hệ thống & Tham số](#4-cấu-hình-hệ-thống--tham-số)
5. [README.md](#weather-forecasting-with-deep-learning)

---

## 1. Giới thiệu bài toán
Dự án sử dụng bộ dữ liệu **Jena Climate Dataset**, ghi nhận các thông số khí tượng tại trạm thời tiết của Viện Max Planck (Jena, Đức). Mục tiêu là dự báo nhiệt độ trong 12 giờ tiếp theo dựa trên dữ liệu lịch sử của 96 giờ trước đó.

## 2. Quy trình Xử lý Dữ liệu
Bộ dữ liệu gốc bao gồm 14 đặc trưng khí tượng cơ bản được ghi nhận từ các cảm biến:

| Tên cột (Raw) | Đơn vị | Ý nghĩa vật lý |
| :--- | :--- | :--- |
| **Date Time** | String | Thời gian ghi nhận (định dạng `dd.mm.yyyy HH:MM:SS`). |
| **p (mbar)** | millibar | **Áp suất khí quyển**. Ảnh hưởng đến các hình thái thời tiết. |
| **T (degC)** | °C | **Nhiệt độ không khí**. Đây là biến mục tiêu (Target). |
| **Tpot (K)** | Kelvin | **Nhiệt độ thế vị**. |
| **Tdew (degC)** | °C | **Nhiệt độ điểm sương**. |
| **rh (%)** | % | **Độ ẩm tương đối**. |
| **VPmax (mbar)** | millibar | **Áp suất hơi nước bão hòa**. |
| **VPact (mbar)** | millibar | **Áp suất hơi nước thực tế**. |
| **VPdef (mbar)** | millibar | **Thâm hụt áp suất hơi nước**. |
| **sh (g/kg)** | g/kg | **Độ ẩm riêng**. |
| **H2OC (mmol/mol)**| mmol/mol | **Nồng độ hơi nước**. |
| **rho (g/m³)** | g/m³ | **Mật độ không khí**. |
| **wv (m/s)** | m/s | **Tốc độ gió**. |
| **max. wv (m/s)**| m/s | **Tốc độ gió tối đa**. |
| **wd (deg)** | độ (°) | **Hướng gió**. |

### Các đặc trưng đầu vào (Input Features)
Dữ liệu thô qua bước **Feature Engineering** để tạo ra 9 đặc trưng tối ưu:

* **Áp suất & Nhiệt độ & Mật độ:** `p`, `T`, `rho` (đã được chuẩn hóa).
* **Vector Gió ($W_x, W_y$):** Thay vì dùng tốc độ và hướng (độ), dữ liệu được chuyển sang hệ tọa độ vector để loại bỏ điểm gián đoạn $0^\circ/360^\circ$.
* **Tín hiệu Thời gian (Time Embeddings):** Mã hóa Sin/Cos cho chu kỳ ngày và năm để mô hình học được tính tuần hoàn (Sáng/Tối, Đông/Hè).

### Chuẩn hóa & Phân chia
* **StandardScaler:** Đưa toàn bộ dữ liệu về phân phối chuẩn ($\mu=0, \sigma=1$).
* **Data Split:** Train (80%) - Validation (10%) - Test (10%).

---

## 3. Kiến trúc Mô hình
Dự án triển khai và so sánh 3 kiến trúc Deep Learning:

### A. LSTM Seq2Seq (Encoder-Decoder)
* **Encoder:** Đọc chuỗi 96 bước quá khứ, nén thông tin vào Vector ngữ cảnh (Context Vector).
* **Decoder:** Giải mã vector ngữ cảnh để sinh ra dự báo cho 12 bước tương lai.
* **Kỹ thuật:** Sử dụng *Teacher Forcing* để hỗ trợ quá trình huấn luyện hội tụ nhanh hơn.

### B. LSTM với Attention Mechanism
* Khắc phục điểm yếu "nút thắt cổ chai" của Context Vector cố định.
* Tại mỗi bước dự báo, Decoder có khả năng "truy vấn" lại toàn bộ các trạng thái ẩn của Encoder để tập trung vào những mốc thời gian quan trọng nhất.

### C. Transformer (Self-Attention)
* **Self-Attention:** Cho phép mô hình đánh giá mối quan hệ giữa các bước thời gian song song.
* **Positional Encoding:** Bổ sung thông tin vị trí để bù đắp việc mất đi tính thứ tự của chuỗi.
* **Masking:** Sử dụng *Look-ahead mask* để đảm bảo tính nhân quả (không lộ thông tin tương lai).

---

## 4. Cấu hình Hệ thống & Tham số

| Tham số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **Input Window** | 96 | Độ dài chuỗi đầu vào (96 giờ) |
| **Forecast Horizon** | 12 | Độ dài chuỗi dự báo (12 giờ) |
| **Feature Dimension** | 9 | Số lượng đặc trưng đầu vào |
| **Hidden Size ($d_{model}$)** | 32 | Kích thước không gian vector ẩn |
| **Layers** | 2 | Số lớp mạng chồng lên nhau |
| **Dropout** | 0.3 | Tỉ lệ ngắt kết nối ngẫu nhiên |
| **Batch Size** | 64 | Kích thước lô dữ liệu |
| **Epochs** | 120 | Số vòng lặp huấn luyện tối đa |
| **Optimization** | Adam | Learning Rate = $1e-4$ |

---

##  Weather Forecasting with Deep Learning (English)

Multi-step temperature forecasting on the **Jena Climate** dataset using three deep-learning architectures implemented in PyTorch.

###  Task
Predict the next **12 hours** of temperature ($T$) given **96 hours** of historical weather data, using an hourly sub-sampled version of the dataset.

###  Project Structure
```text
projectDL/
├── config.py          # All hyperparameters & paths
├── dataset.py         # Download, feature engineering, DataLoaders
├── models.py          # LSTMSeq2Seq · LSTMAttention · TransformerModel
├── trainer.py         # Training loop, Early Stopping, Scheduled Sampling
├── evaluate.py        # Metrics, forecast & history plots
├── train.py           # Main entry-point (CLI)
├── requirements.txt
├── data/              # Raw CSV (auto-downloaded)
├── checkpoints/       # Saved model weights (best_*.pth)
└── logs/              # Training & performance plots

## Models

| Model | Architecture | Params |
|---|---|---|
| **LSTM** | Encoder-Decoder LSTM + Scheduled Sampling | ~27 K |
| **LSTM_Attn** | + Bahdanau Attention over encoder states | ~33 K |
| **Transformer** | Standard Encoder-Decoder Transformer | ~60 K |

All models share the same hyperparameters (configurable in config.py):

| Param | Value |
|---|---|
| d_model | 32 |
| 
um_layers | 2 |
| dropout | 0.3 |
| input_window | 96 h |
| Forecast_horizon | 12 h |
| epochs | 120 |
| learning_rate | 1e-4 |
| Batch_size | 64 |

---

## âœ¨ Features

- **Modular codebase** â€“ each concern in its own file
- **Scheduled Sampling** â€“ teacher-forcing ratio decays linearly to 0
- **Early Stopping** â€“ saves best checkpoint; restores it after training
- **LR Scheduler** â€“ ReduceLROnPlateau on val loss
- **Gradient Clipping** â€“ max_norm=1.0 for stability
- **Feature Engineering** â€“ wind vector decomposition + cyclic time encoding
- **CLI** â€“ flexible training & evaluation from the command line

---

##  Quick Start

`Bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all three models 
python train.py

# 3. Train a single model
python train.py --model LSTM_Attn

# 4. Evaluate saved checkpoints without retraining
python train.py --eval-only

# 5. Smoke-test (2 epochs)
python train.py --debug
`

---

## Results

After training, the script prints a comparison table and saves two plots:

- logs/forecast_comparison.png â€“ actual vs. predicted at 6 different horizons
- logs/training_history.png â€“ MSE loss & validation MAE curves


 Data

[Jena Climate 2009â€“2016](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip) (Max Planck Institute for Biogeochemistry).  
The raw CSV is automatically downloaded to data/ on first run.


