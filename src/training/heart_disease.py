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

    except Exception as e:
        logging.error("Training failed", exc_info=True)
        raise


if __name__ == "__main__":
    train_heart_disease_model()
