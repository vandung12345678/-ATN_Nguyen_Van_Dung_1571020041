
from fastapi import APIRouter, HTTPException
from controller import make_prediction_for_model
from model import TransactionData

router = APIRouter(prefix="/api")

# Thống kê độ chính xác từng model
model_stats = {
    "random_forest": "95.3%",
    "xgboost":        "96.1%",
    "svm":            "93.8%",
    "neural_network": "94.2%",
    "decision_tree":  "90.5%",
    "lstm":           "92.7%"
}

@router.post("/predict/{model_name}")
async def predict_specific_model_endpoint(
    data: TransactionData,
    model_name: str
):
    # 1. Kiểm tra model có hỗ trợ không
    if model_name not in model_stats:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' không tồn tại."
        )
    # 2. Gọi controller để dự đoán
    try:
        result = make_prediction_for_model(data, model_name)
    except ValueError as e:
        # Lỗi nội bộ khi predict
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    # 3. Trả về kết quả
    return {
        "model":       model_name,
        "prediction":  result["class"],
        "probability": result["probability"],
        "accuracy":    model_stats[model_name]
    }
