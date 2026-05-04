"""
ead_model.py
------------
Exposure at Default (EAD) model.

EAD = expected outstanding balance at time of default.
Expressed as a fraction of the original loan amount (Credit Conversion Factor).

For term loans (LendingClub), EAD is modelled as:
    EAD = loan_amnt * CCF
where CCF is predicted from loan and borrower characteristics.

Basel III context:
    CCF is driven primarily by time-to-default: riskier borrowers default
    earlier in the loan lifecycle, leaving more balance outstanding.
    payment_to_income captures affordability stress — a leading indicator
    of how quickly a borrower will exhaust their capacity to service the debt.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

MODELS_DIR = Path("models")
BASE_FEATURES = ["loan_amnt", "int_rate", "term", "dti", "fico_score", "grade_idx"]
FEATURES      = BASE_FEATURES + ["payment_to_income"]


def _add_payment_to_income(df: pd.DataFrame) -> pd.DataFrame:
    """Derive monthly payment-to-income ratio from loan terms."""
    df = df.copy()
    monthly_rate = df["int_rate"] / 12
    df["monthly_payment"] = (
        df["loan_amnt"] * monthly_rate / (1 - (1 + monthly_rate) ** (-df["term"]))
    )
    df["payment_to_income"] = df["monthly_payment"] / (df["annual_inc"] / 12)
    return df


def fit_ead_model(df_train: pd.DataFrame, df_test: pd.DataFrame | None = None) -> tuple:
    """
    Fit GradientBoostingRegressor to predict Credit Conversion Factor (CCF).

    CCF = EAD / loan_amnt — the fraction of principal outstanding at default.
    Gradient boosting captures the non-linear interaction between borrower risk
    (grade, FICO, DTI) and how early in the loan lifecycle default occurs.
    """
    df_train = _add_payment_to_income(df_train)

    X_train = df_train[FEATURES].values
    y_train = df_train["ead_util"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=50,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train).clip(0.10, 1.0)
    train_mae  = mean_absolute_error(y_train, train_pred)
    train_r2   = r2_score(y_train, train_pred)
    print(f"[ead_model] Train  MAE: {train_mae:.4f}  R²: {train_r2:.4f}")

    if df_test is not None:
        df_test  = _add_payment_to_income(df_test)
        X_test   = scaler.transform(df_test[FEATURES].values)
        test_pred = model.predict(X_test).clip(0.10, 1.0)
        test_mae  = mean_absolute_error(df_test["ead_util"].values, test_pred)
        test_r2   = r2_score(df_test["ead_util"].values, test_pred)
        print(f"[ead_model] Test   MAE: {test_mae:.4f}  R²: {test_r2:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model,  "models/ead_model.pkl")
    joblib.dump(scaler, "models/ead_scaler.pkl")
    return model, scaler


def predict_ead(model, scaler, df: pd.DataFrame) -> np.ndarray:
    """Return predicted EAD (in currency) for each loan."""
    df  = _add_payment_to_income(df)
    X   = scaler.transform(df[FEATURES].values)
    ccf = model.predict(X).clip(0.10, 1.0)
    return df["loan_amnt"].values * ccf


def evaluate_ead(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute EAD model metrics on a held-out set."""
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"[ead_model] Eval   MAE: {mae:.4f}  R²: {r2:.4f}")
    return {"mae": round(mae, 4), "r2": round(r2, 4)}
