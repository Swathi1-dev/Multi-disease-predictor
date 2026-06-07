import logging
from pathlib import Path

import pandas as pd
from joblib import load

from src.backend.config.settings import Settings
from src.common.preprocessing_util import replace_zero_with_nan

settings = Settings()

DIABETES_MODEL_PATH = Path(settings.diabetes_model_path)
HEART_DISEASE_MODEL_PATH = Path(settings.heart_disease_model_path)
LOG_PATH = Path(settings.log_path)

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH / "app.log"), logging.StreamHandler()],
)

# load models once

logging.info("Loading trained models...")
diabets_model = load(DIABETES_MODEL_PATH / "diabetes_prediction_pipeline.joblib")
heart_disease_model = load(
    HEART_DISEASE_MODEL_PATH / "heart_disease_prediction_pipeline.joblib"
)
logging.info("Models loaded successfully.")


# common predictor function
def predict_disease(disease: str, input_data: dict):
    if disease == "diabetes":
        model = diabets_model
    elif disease == "heart_disease":
        model = heart_disease_model
    else:
        raise ValueError("Invalid disease type. Use diabetes or heart_disease")

    x_df = pd.DataFrame(input_data, index=[0])

    prediction = int(model.predict(x_df)[0])

    # probability for positiveclass
    probability = float(model.predict_proba(x_df)[0][1])

    logging.info(f"{disease} prediction={prediction},probability={probability}")

    return {"disease": disease, "prediction": prediction, "probability": probability}


# # example
# features = {
#     "age": 52,
#     "sex": 1,
#     "cp": 0,
#     "trestbps": 125,
#     "chol": 212,
#     "fbs": 0,
#     "restecg": 1,
#     "thalach": 168,
#     "exang": 0,
#     "oldpeak": 1,
#     "slope": 2,
#     "ca": 2,
#     "thal": 3,
# }

# predict_disease(disease="heart_disease", input_data=features)

featuress = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}

predict_disease(disease="diabetes", input_data=featuress)
