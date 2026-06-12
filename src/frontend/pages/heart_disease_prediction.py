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
    page_title="MedBuddy - Heart Disease Predictor", page_icon="❤️", layout="centered"
)
st.title("MedBuddy - Heart Disease Predictor")
st.markdown(
    """
    This application predicts the likelihood of heart disease based on user input.
    Please fill in the details below and click the 'Predict' button to see your results.
    """
)


col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox(
        "Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male"
    )
    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic",
        ][x],
    )
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120
    )
    chol = st.number_input(
        "Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200
    )
with col2:
    fbs = st.number_input(
        "Fasting Blood Sugar > 120 mg/dl", min_value=0, max_value=1, value=0
    )
    restecg = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy",
        ][x],
    )
    thalach = st.number_input(
        "Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
    )
    with col3:
        oldpeak = st.number_input(
            "ST Depression Induced by Exercise",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )
        slope = st.selectbox(
            "Slope of the Peak Exercise ST Segment",
            options=[0, 1, 2],
            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
        )
        ca = st.selectbox(
            "Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3]
        )
        thal = st.number_input(
            "Thalassemia (1 = Normal; 2 = Reversible Defect; 3 = Fixed Defect)",
            min_value=1,
            max_value=3,
            value=1,
        )

if st.button("Predict"):
    input_data = {
        "disease": "heart_disease",
        "features": {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
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
            st.error("Heart Disease Detected. Please consult a doctor.")
        else:
            st.success("No Heart Disease Detected. Keep up the healthy lifestyle!")
