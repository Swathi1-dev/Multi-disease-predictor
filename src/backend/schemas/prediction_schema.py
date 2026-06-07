from typing import Dict
from pydantic import BaseModel


class PredictionRequests(BaseModel):
    disease: str
    features: Dict[
        str, int | float
    ]  # keys shloud be str and values should be int or float


class PredictionResponse(BaseModel):
    disease: str
    prediction: int
    probability: float
