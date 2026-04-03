"""
Generates E-commerce churn model using a synthetic dataset.
This avoids Kaggle download issues and produces 100% compatible .pkl files.
"""
import os
import pickle
import json
import sys

print(f"Python: {sys.executable}")

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("✅ All packages loaded!")

np.random.seed(42)
N = 5000

# Match typical ecommerce churn features
df = pd.DataFrame({
    'Tenure': np.random.randint(0, 72, N),
    'CityTier': np.random.choice([1, 2, 3], N),
    'WarehouseToHome': np.random.randint(5, 35, N),
    'HourSpendOnApp': np.random.uniform(0, 5, N).round(1),
    'NumberOfDeviceRegistered': np.random.randint(1, 6, N),
    'SatisfactionScore': np.random.randint(1, 6, N),
    'NumberOfAddress': np.random.randint(1, 10, N),
    'Complain': np.random.randint(0, 2, N),
    'OrderAmountHikeFromlastYear': np.random.uniform(10, 30, N).round(1),
    'CouponUsed': np.random.randint(0, 15, N),
    'OrderCount': np.random.randint(1, 16, N),
    'DaySinceLastOrder': np.random.randint(0, 30, N),
    'CashbackAmount': np.random.uniform(0, 300, N).round(2),
    'PreferredLoginDevice': np.random.choice(['Mobile Phone', 'Computer', 'Phone'], N),
    'PreferredPaymentMode': np.random.choice(['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E wallet'], N),
    'Gender': np.random.choice(['Male', 'Female'], N),
    'PreferedOrderCat': np.random.choice(['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'], N),
    'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], N),
})

# Synthetic churn based on features
churn_prob = (
    (df['SatisfactionScore'] <= 2).astype(int) * 0.4 +
    (df['Complain'] == 1).astype(int) * 0.3 +
    (df['Tenure'] < 6).astype(int) * 0.3
)
df['Churn'] = (churn_prob + np.random.uniform(0, 0.3, N) > 0.6).astype(int)

target = 'Churn'
X = df.drop(target, axis=1)
y = df[target]

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print(f"Features: {X.shape[1]}, Cat: {cat_cols}")

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

print("Training model...")
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print(f"✅ Trained! Accuracy estimate: {model.score(X, y):.2%}")

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'e-commerse')
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, 'best_model_gradient_boosting.pkl'), 'wb') as f:
    pickle.dump(model, f)
with open(os.path.join(out_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(out_dir, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump(encoders, f)

info = {"features": X.columns.tolist(), "categorical_columns": cat_cols}
with open(os.path.join(out_dir, 'model_info.json'), 'w') as f:
    json.dump(info, f, indent=2)

print(f"✅ All model files saved to: {out_dir}")
print("Done! Restart Flask to use the fresh model.")
