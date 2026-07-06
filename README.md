# Multi-Disease Predictor

A machine learning system that predicts the risk of **diabetes** and **heart disease** from patient health data, served through a FastAPI backend. Each disease has its own trained scikit-learn pipeline, covering the full workflow from EDA and preprocessing to hyperparameter tuning and deployment.

## Overview

- **Diabetes prediction** — trained on the Pima Indians Diabetes dataset (768 records, 8 features)
- **Heart disease prediction** — trained on a heart disease dataset (1,025 records, 13 features)
- Both models are exposed via a single FastAPI application with disease-specific prediction endpoints

## Features

- End-to-end EDA: missing value analysis, class distribution, correlation heatmaps, outlier detection
- Custom preprocessing: zero-as-missing value imputation for physiologically invalid zero entries (e.g., zero blood pressure), median imputation, feature scaling
- Model benchmarking across Logistic Regression, SVM, Random Forest, and XGBoost using stratified k-fold cross-validation
- Hyperparameter tuning via GridSearchCV (diabetes) and RandomizedSearchCV (heart disease)
- Group-aware train/test splitting (GroupShuffleSplit, StratifiedGroupKFold) to prevent data leakage from duplicate records in the heart disease dataset
- Trained pipelines serialized with joblib for reproducible inference
- REST API built with FastAPI and Pydantic for request/response validation
- Interactive Streamlit frontend with sidebar navigation for diabetes and heart disease risk prediction, calling the FastAPI backend for live inference
- Environment-based configuration (dataset paths, model paths, hyperparameter config) via `.env`

## Results

| Disease | Best Model | Test Accuracy | Notes |
|---|---|---|---|
| Heart Disease | Random Forest (tuned) | **83.97%** | F1 (macro, CV): 0.85 |
| Diabetes | SVM (tuned) | **72.73%** | F1 (CV): 0.69 |

Both models were tuned against baseline Logistic Regression models and evaluated using accuracy, precision, recall, and F1-score on held-out test sets.

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn, XGBoost
- **API:** FastAPI, Pydantic, Uvicorn
- **Frontend:** Streamlit
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Model persistence:** joblib
- **Config:** python-dotenv / pydantic-settings, YAML

## Project Structure

```
Multi-disease-predictor/
├── dataset/                # diabetes.csv, heart.csv
├── model_dir/               # serialized model pipelines (.joblib)
├── notebook_dir/             # EDA, training & evaluation notebooks
│   ├── diabetes_prediction.ipynb
│   └── heart_disease_prediction.ipynb
├── src/
│   ├── backend/
│   │   └── api/
│   │       └── routes.py    # FastAPI route definitions
│   ├── frontend/
│   │   └── app.py            # Streamlit app entry point
│   └── training/
│       └── config/
│           └── best_hyperparameters.yaml
├── main.py                  # FastAPI app entry point
├── requirements.txt
└── env_template.txt         # environment variable template
```

## Setup

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `env_template.txt` to `.env` and set your project paths:
   ```bash
   cp env_template.txt .env
   ```

3. Run the API:
   ```bash
   uvicorn main:app --reload
   ```

4. Access the interactive API docs at `http://localhost:8000/docs`

5. In a separate terminal, launch the Streamlit frontend:
   ```bash
   streamlit run src/frontend/app.py
   ```

## Future Improvements

- Expand diabetes dataset or apply resampling techniques to address class imbalance
- Add SHAP-based explainability for model predictions
- Containerize with Docker and add CI/CD via GitHub Actions
