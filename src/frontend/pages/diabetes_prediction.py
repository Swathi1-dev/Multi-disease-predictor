import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import requests

from src.frontend.config.settings import Settings

settings = Settings()
api_url = settings.api_url

st.set_page_config(
    page_title="MedBuddy - Diabetes  Predictor", page_icon="❤️", layout="centered"
)
st.title("MedBuddy - Diabetes Predictor")
st.markdown(
    """
    This application predicts the likelihood of Diabetes based on user input.
    Please fill in the details below and click the 'Predict' button to see your results.
    """
)

# "Pregnancies": 1,
# "Glucose": 85,
# "BloodPressure": 66,
# "SkinThickness": 29,
# "Insulin": 0,
# "BMI": 26.6,
# "DiabetesPedigreeFunction": 0.351,
# "Age": 31,

col1, col2 = st.columns(2)
with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    Glucose = st.number_input("Glucose", max_value=100, value=40, min_value=1)
    BloodPressure = st.number_input(
        "BloodPressure", max_value=100, value=50, min_value=1
    )
    SkinThickness = st.number_input(
        "SkinThickness", max_value=100, value=1, min_value=1
    )
with col2:
    Insulin = st.number_input("Insulin", max_value=100, value=40, min_value=1)
    BMI = st.number_input("BMI", max_value=100, value=40, min_value=1)
    DiabetesPedigreeFunction = st.number_input(
        "DiabetesPedigreeFunction", max_value=100, value=10, min_value=1
    )
    Age = st.number_input("Age", max_value=100, value=18, min_value=1)

if st.button("Predict"):
    input_data = {
        "disease": "diabetes",
        "features": {
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "BloodPressure": BloodPressure,
            "SkinThickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
            "Age": Age,
        },
    }

    response = requests.post(api_url, json=input_data)

    if response.status_code != 200:
        st.error("Prediction failed. Please try again.")
    else:
        result = response.json()
        prediction = result["prediction"]
        probability = result["probability"]
        disease = result["disease"]

        st.divider()

        st.metric(
            label="Heart Disease prediction",
            value="probability: {:.2f}%".format(probability * 100),
            delta=disease,
        )

        if prediction == 1:
            st.error("Diabetes Detected. Please consult a doctor.")
        else:
            st.success("No Diabetes Detected. Keep up the healthy lifestyle!")
