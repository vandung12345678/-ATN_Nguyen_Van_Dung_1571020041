# model.py
from pydantic import BaseModel
import joblib
import numpy as np
import os
import tensorflow as tf

# 1. Định nghĩa input model gồm 31 đặc trưng
class TransactionData(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float


# 2. Load model
def load_model(model_name: str):
    model_dir = "models"
    pkl_path = os.path.join(model_dir, f"{model_name}.pkl")
    h5_path = os.path.join(model_dir, f"{model_name}.h5")

    if os.path.exists(pkl_path):
        print(f"✅ Loaded sklearn model: {pkl_path}")
        return joblib.load(pkl_path), "sklearn"
    elif os.path.exists(h5_path):
        print(f"✅ Loaded keras model: {h5_path}")
        model = tf.keras.models.load_model(h5_path)
        return model, "keras"
    else:
        raise ValueError(f"Model '{model_name}' không tồn tại trong thư mục models.")

# 3. Xử lý đầu vào
def preprocess_input(data: TransactionData):
    features = [
        data.V1, data.V2, data.V3, data.V4, data.V5,
        data.V6, data.V7, data.V8, data.V9, data.V10,
        data.V11, data.V12, data.V13, data.V14, data.V15,
        data.V16, data.V17, data.V18, data.V19, data.V20,
        data.V21, data.V22, data.V23, data.V24, data.V25,
        data.V26, data.V27, data.V28,
        data.Amount
    ]
    return np.array([features], dtype=float)

# 4. Dự đoán
def predict_transaction(input_data: np.ndarray, model_name: str):
    model, mtype = load_model(model_name)

    if mtype == "sklearn":
        prob = model.predict_proba(input_data)[0][1]
        cls = int(model.predict(input_data)[0])
    elif mtype == "keras":
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
        prob = float(model.predict(input_data)[0][0])
        cls = int(prob > 0.5)
    else:
        raise ValueError("Unsupported model type")

    return {"class": cls, "probability": prob}
