import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

print("1. Đang tải và làm sạch dữ liệu...")
df = pd.read_csv('online_retail.csv', encoding='latin1')
df.to_parquet('online_retail.parquet')
df = pd.read_parquet('online_retail.parquet')
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalSum'] = df['Quantity'] * df['UnitPrice']

print("2. Đang tạo đặc trưng RFM và Nhãn (Target)...")
cutoff_date = pd.to_datetime('2011-09-01')
feature_df = df[df['InvoiceDate'] < cutoff_date]
target_df = df[df['InvoiceDate'] >= cutoff_date]

rfm = feature_df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (cutoff_date - x.max()).days, # Recency
    'InvoiceNo': 'nunique',                                # Frequency
    'TotalSum': 'sum'                                      # Monetary
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

target = target_df.groupby('CustomerID').agg({'TotalSum': 'sum'}).reset_index()
target.columns = ['CustomerID', 'Future_CLV']

# Gộp dữ liệu
data = pd.merge(rfm, target, on='CustomerID', how='left')
data['Future_CLV'] = data['Future_CLV'].fillna(0)

print("3. Đang xuất file clv_processed.csv cho Streamlit...")
data.to_csv('clv_processed.csv', index=False)

print("4. Đang huấn luyện mô hình Random Forest...")
X = data[['Recency', 'Frequency', 'Monetary']]
y = data['Future_CLV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("5. Đang lưu mô hình (model.pkl)...")
# Tạo thư mục 'models' nếu chưa có
if not os.path.exists('models'):
    os.makedirs('models')

# Lưu mô hình
with open('models/model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("✅ HOÀN TẤT! Bây giờ bạn có thể chạy app.py bằng Streamlit.")