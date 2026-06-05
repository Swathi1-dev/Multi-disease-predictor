import numpy as np


def replace_zero_with_nan(X):
    x = X.copy()
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols:
        if col in x.columns:
            x[col] = x[col].replace(0, np.nan)
    return x
