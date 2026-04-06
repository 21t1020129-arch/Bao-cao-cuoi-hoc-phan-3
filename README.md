📝 Dự Đoán Giá Trị Vòng Đời Khách Hàng (CLV Prediction)
📌 Tổng Quan Đề Tài
Dự án này tập trung vào việc phân tích dữ liệu giao dịch của một doanh nghiệp bán lẻ trực tuyến tại Anh để dự đoán Giá trị vòng đời khách hàng (Customer Lifetime Value - CLV). Mục tiêu cốt lõi là xác định những khách hàng nào sẽ mang lại lợi nhuận cao nhất trong tương lai, từ đó giúp doanh nghiệp tối ưu hóa chiến lược Marketing và chăm sóc khách hàng.

📊 Bộ Dữ Liệu
Nguồn: online_retail.csv (Giao dịch từ 01/12/2010 đến 09/12/2011).

Đặc điểm: Dữ liệu bán buôn (Wholesale) xuyên biên giới với các thông tin chính: InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID, và Country.

🛠 Quy Trình Thực Hiện
Dự án được triển khai qua 4 giai đoạn chính:

Tiền xử lý (Data Cleaning): Loại bỏ dữ liệu khuyết thiếu, xử lý đơn hàng hoàn trả (âm), và lọc nhiễu.

Kỹ thuật đặc trưng (Feature Engineering): * Trích xuất bộ chỉ số RFM (Recency, Frequency, Monetary).

Tính toán các biến nâng cao: Tần suất mua hàng trung bình, Tỷ lệ hoàn trả, và Vòng đời hiện tại.

Huấn luyện mô hình: Sử dụng thuật toán Random Forest Regressor để học các quy luật chi tiêu phức tạp.

Đánh giá: Kiểm chứng độ chính xác qua các chỉ số MAE, RMSE và biểu đồ Scatter Plot.

🚀 Kết Quả Đạt Được
Độ chính xác: Mô hình dự báo rất sát thực tế đối với nhóm khách hàng phổ thông (MAE ≈ 937).

Thấu hiểu khách hàng: Xác định được Monetary đóng góp tới 87.5% vào việc dự đoán giá trị tương lai.

Ứng dụng: Hệ thống hỗ trợ phân loại khách hàng VIP và khách hàng có nguy cơ rời bỏ (Churn) một cách tự động.

📁 Cấu Trúc Thư Mục
Plaintext
├── data/
│   └── online_retail.csv          # Dữ liệu thô
├── notebooks/
│   └── clv_analysis.ipynb         # File chạy thử nghiệm và trực quan hóa
├── src/
│   ├── preprocessing.py           # Code làm sạch dữ liệu
│   └── model_training.py          # Code huấn luyện Random Forest
├── README.md                      # Giới thiệu dự án
└── requirements.txt               # Các thư viện cần thiết (pandas, scikit-learn,...)
💻 Hướng Dẫn Cài Đặt
Cài đặt các thư viện cần thiết:

Bash
pip install -r requirements.txt
Chạy file phân tích chính:

Bash
python src/model_training.py
💡 Giá trị mang lại: Giúp doanh nghiệp chuyển đổi từ việc quản lý dựa trên cảm tính sang Quản trị dựa trên dữ liệu (Data-driven Management), tăng hiệu quả sử dụng ngân sách Marketing lên tới 20-30% thông qua việc tập trung vào đúng đối tượng khách hàng mục tiêu.
