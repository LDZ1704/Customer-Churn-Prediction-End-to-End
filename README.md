# Customer Churn Prediction – End-to-End

## Mục tiêu
Xây dựng quy trình hoàn chỉnh dự đoán khách hàng rời bỏ dịch vụ (churn) từ EDA → tiền xử lý → huấn luyện nhiều mô hình → đánh giá → phục vụ API.

## Yêu cầu môi trường
- Python 3.10+ (cài đặt trước trên Windows, kiểm tra với `python --version`)
- pip, venv (khuyến khích dùng môi trường ảo)

### Cài đặt nhanh
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Cấu trúc đề xuất
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — dữ liệu gốc Telco.
- `train.py` — huấn luyện, đánh giá, lưu mô hình tốt nhất.
- `predict.py` — suy luận từ mẫu JSON/CSV dùng mô hình đã huấn luyện.
- `app.py` — FastAPI phục vụ REST `/predict`.
- `models/` — nơi lưu mô hình (`best_model.joblib`) và metadata (`metadata.json`).
- `notebooks/` — sổ tay EDA (tùy chọn tự tạo thêm).

## Quy trình khuyến nghị
1) **EDA**: kiểm tra thiếu dữ liệu, phân bố biến, tương quan (tạo notebook riêng trong `notebooks/`).  
2) **Tiền xử lý** (đã code trong `train.py`):  
   - Loại bỏ `customerID`, mục tiêu `Churn`.  
   - Chuẩn hóa `TotalCharges` thành số, xử lý ô trống.  
   - One-Hot cho biến phân loại, giữ chuẩn hóa min-max cho số.  
3) **Huấn luyện**: Logistic Regression, Random Forest, XGBoost (nếu cài), chọn mô hình ROC-AUC cao nhất.  
4) **Đánh giá**: accuracy, F1, ROC-AUC; xem classification report trên tập validation.  
5) **Triển khai**: chạy `app.py` (FastAPI) hoặc tự xây Streamlit dashboard.

## Chạy huấn luyện
```bash
python train.py --data WA_Fn-UseC_-Telco-Customer-Churn.csv --test-size 0.2 --random-state 42
```
Đầu ra:
- `models/best_model.joblib`
- `models/metadata.json` (cột đầu vào, loại cột, tham số mô hình)
- In ra bảng metric mỗi mô hình.

## Dự đoán nhanh
```bash
python predict.py --json sample.json
```
`sample.json` chứa 1 mẫu theo đúng tên cột gốc (trừ `customerID`, `Churn`). Ví dụ được ghi chú trong `predict.py`.

## Phục vụ API
```bash
uvicorn app:app --reload --port 8000
```
Gửi POST:
```json
{
  "inputs": [
    {
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 1,
      "PhoneService": "No",
      "MultipleLines": "No phone service",
      "InternetService": "DSL",
      "OnlineSecurity": "No",
      "OnlineBackup": "Yes",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 29.85,
      "TotalCharges": 29.85
    }
  ]
}
```
Đáp ứng: xác suất churn và nhãn dự đoán.