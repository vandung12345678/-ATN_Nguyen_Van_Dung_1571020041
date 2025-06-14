
from model import TransactionData, preprocess_input, predict_transaction

def make_prediction_for_model(
    data: TransactionData,
    model_name: str
) -> dict:
    """
    Nhận Pydantic model TransactionData và tên model_name,
    trả về {'class': int, 'probability': float}.
    """
    # 1. Tiền xử lý input
    input_arr = preprocess_input(data)
    # 2. Dự đoán
    try:
        result = predict_transaction(input_arr, model_name)
    except Exception as e:
        # Nếu có lỗi trong load/predict, bọc lại để router bắt
        raise ValueError(f"Prediction error: {e}")
    return result
