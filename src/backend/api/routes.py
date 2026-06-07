from fastapi import APIRouter

from src.backend.schemas.prediction_schema import PredictionRequests, PredictionResponse

from src.backend.services.predictor import predict_disease


router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}


@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(input_data: PredictionRequests):
    disease = input_data.disease
    features = input_data.features
    result = predict_disease(disease=disease, input_data=features)

    return PredictionResponse(**result)
