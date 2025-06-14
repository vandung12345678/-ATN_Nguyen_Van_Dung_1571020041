import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, classification_report

# 1. Load và xử lý dữ liệu
df = pd.read_csv("data_traning.1.2.csv")
df['Hour']      = ((df['Time'] // 3600) % 24).astype(int)
df['DayOfWeek'] = ((df['Time'] // (3600 * 24)) % 7).astype(int)
df = df.drop(columns=['Time'])
df = df.iloc[:-100]
# 2. Chọn đặc trưng bao gồm 31 cột
features = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount'
]
X = df[features]
y = df['Class']

# 3. Chuẩn hóa dữ liệu và chia tập
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Tạo thư mục lưu mô hình
os.makedirs("models", exist_ok=True)

# 5. Hàm đánh giá và lưu mô hình
def evaluate_and_save(model, X_test, y_test, name, is_keras=False):
    if is_keras:
        X_eval = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        y_pred = (model.predict(X_eval).ravel() > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name.upper()} ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, digits=4))

    # Lưu model
    if is_keras:
        model.save(f"models/{name}.h5")
    else:
        joblib.dump(model, f"models/{name}.pkl")
    print(f"💾 Saved model: {name}")

# 6. Huấn luyện các mô hình

# Random Forest
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
evaluate_and_save(rf, X_test, y_test, "random_forest")

# Neural Network (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
evaluate_and_save(mlp, X_test, y_test, "neural_network")

# SVM
svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
svm.fit(X_train, y_train)
evaluate_and_save(svm, X_test, y_test, "svm")

# Decision Tree
dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
evaluate_and_save(dt, X_test, y_test, "decision_tree")

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
evaluate_and_save(xgb, X_test, y_test, "xgboost")

# LSTM
X_train_l = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_l  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_l.shape[1], 1)),
    LSTM(50),
    Dense(1, activation='sigmoid')
])
lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm.fit(X_train_l, y_train, epochs=10, batch_size=64, validation_data=(X_test_l, y_test), verbose=2)
evaluate_and_save(lstm, X_test, y_test, "lstm", is_keras=True)
