import logging
import yaml
import os
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
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


def train_diabets_model():
    try:
        settings = Settings()
        DATASET_PATH = Path(settings.diabetes_dataset_path)
        MODEL_PATH = Path(settings.diabetes_model_path)
        LOG_PATH = Path(settings.log_path)
        HYPERPARAMS = Path(settings.hyper_params_yaml_path)

        TARGET_COL = settings.diabetes_target_col
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

        logging.info("Starting diabetes model training")

        print(DATASET_PATH)
        print(DATASET_PATH.exists())
        df = pd.DataFrame(pd.read_csv(DATASET_PATH))
        logging.info(f"Dataset loaded with shape: {df.shape}")

        x = df.drop(TARGET_COL, axis=1)
        y = df[TARGET_COL]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        logging.info(f"Data split into train and test sets with test size {TEST_SIZE}")

        numeric_features = x_train.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            (
                                "replace_zero",
                                FunctionTransformer(
                                    replace_zero_with_nan, validate=False
                                ),
                            ),
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                )
            ],
            remainder="drop",
        )
        print(HYPERPARAMS.exists())
        with open(HYPERPARAMS, "r") as f:
            hyperparams = yaml.safe_load(f)

        model_params = hyperparams["diabetes"]["params"]

        model = SVC(random_state=RANDOM_STATE, **model_params)

        pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

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

        dump(pipeline, MODEL_PATH / "diabetes_prediction_pipeline.joblib")
        logging.info(f"Model saved to {MODEL_PATH}")

    except Exception as e:
        logging.error("Training failed", exc_info=True)
        raise


if __name__ == "__main__":
    train_diabets_model()
