"""
ead_model.py
------------
Exposure at Default (EAD) model.

EAD = expected outstanding balance at time of default.
Expressed as a fraction of the original loan amount (Credit Conversion Factor).

For term loans (LendingClub), EAD is modelled as:
    EAD = loan_amnt * CCF
where CCF (Credit Conversion Factor) is predicted by a linear model.

Basel III context:
    For revolving facilities, EAD is more complex (depends on undrawn limits).
    For term loans, CCF is typically close to 1 but varies by time-to-default.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

MODELS_DIR = Path("models")
FEATURES   = ["loan_amnt", "int_rate", "term", "dti", "fico_score"]


def fit_ead_model(df_train: pd.DataFrame) -> tuple:
    """
    Fit Ridge regression to predict Credit Conversion Factor (CCF).

    CCF = EAD / loan_amnt (bounded between 0.5 and 1.0 in our data).
    Ridge regression handles multicollinearity between loan features.
    """
    X = df_train[FEATURES].values
    y = df_train["ead_util"].values   # CCF = EAD / loan_amnt

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled).clip(0.5, 1.0)
    mae    = mean_absolute_error(y, y_pred)
    r2     = r2_score(y, y_pred)
    print(f"[ead_model] Train MAE: {mae:.4f}  R²: {r2:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model,  "models/ead_model.pkl")
    joblib.dump(scaler, "models/ead_scaler.pkl")
    return model, scaler


def predict_ead(model, scaler, df: pd.DataFrame) -> np.ndarray:
    """
    Return predicted EAD (in currency) for each loan.
    EAD = loan_amnt * predicted_CCF
    """
    X   = scaler.transform(df[FEATURES])
    ccf = model.predict(X).clip(0.5, 1.0)
    return df["loan_amnt"].values * ccf