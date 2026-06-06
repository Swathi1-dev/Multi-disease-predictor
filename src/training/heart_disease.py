import logging
import yaml
import os
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.common.preprocessing_util import replace_zero_with_nan
from src.training.config.settings import Settings


def train_heart_disease_model():
    try:
        settings = Settings()
        DATASET_PATH = Path(settings.heart_disease_dataset_path)
        MODEL_PATH = Path(settings.heart_disease_model_path)
        LOG_PATH = Path(settings.log_path)
        HYPERPARAMS = Path(settings.hyper_params_yaml_path)

        TARGET_COL = settings.heart_disease_target_col
        TEST_SIZE = settings.test_size
        RANDOM_STATE = settings.random_state

        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        LOG_PATH.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(LOG_PATH / "training.log"),
                logging.StreamHandler(),
            ],
        )

        logging.info("Starting heart disease model training")

        print(DATASET_PATH)
        print(DATASET_PATH.exists())
        df = pd.DataFrame(pd.read_csv(DATASET_PATH))
        logging.info(f"Dataset loaded with shape: {df.shape}")

        x = df.drop(TARGET_COL, axis=1)
        y = df[TARGET_COL]

        row_signature = pd.util.hash_pandas_object(x, index=False)

        gss = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        train_idx, test_idx = next(gss.split(x, y, groups=row_signature))

        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        logging.info(f"Train shape: {x_train.shape}, Test shape:{x_test.shape}")

        with open(HYPERPARAMS, "r") as f:
            hyperparams = yaml.safe_load(f)

        model_params = hyperparams["heart_disease"]["params"]

        best_rf = RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, **model_params
        )

        # Keep scaler in pipeline to match notebook structure

        pipeline = Pipeline([("sclaer", StandardScaler()), ("model", best_rf)])

        pipeline.fit(x_train, y_train)
        logging.info("Model training completed")

        y_train_pred = pipeline.predict(x_train)
        y_test_pred = pipeline.predict(x_test)

        logging.info(
            f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}, "
            f"F1 Score: {f1_score(y_train, y_train_pred):.4f}, "
            f"Precision: {precision_score(y_train, y_train_pred):.4f}, "
            f"Recall: {recall_score(y_train, y_train_pred):.4f}"
        )

        logging.info(
            f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}, "
            f"F1 Score: {f1_score(y_test, y_test_pred):.4f}, "
            f"Precision: {precision_score(y_test, y_test_pred):.4f}, "
            f"Recall: {recall_score(y_test, y_test_pred):.4f}"
        )

        dump(pipeline, MODEL_PATH / "heart_disease_prediction_pipeline.joblib")
        logging.info(f"Model saved to {MODEL_PATH}")

    except Exception as e:
        logging.error("Training failed", exc_info=True)
        raise


if __name__ == "__main__":
    train_heart_disease_model()
