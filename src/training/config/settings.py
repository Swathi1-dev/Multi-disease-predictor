from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_path: str
    diabetes_dataset_path: str
    heart_disease_dataset_path: str
    diabetes_model_path: str
    heart_disease_model_path: str
    diabetes_target_col: str
    heart_disease_target_col: str
    hyper_params_yaml_path: str
    test_size: float
    random_state: int

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
