"""
pd_model.py
-----------
Probability of Default (PD) model.

Uses logistic regression on borrower and loan characteristics.
Output: P(default) per loan — first component of Expected Loss.

Basel III context:
    PD is the probability that a borrower will default within 12 months.
    Must be calibrated to a long-run average default rate (Through-the-Cycle).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pathlib import Path

MODELS_DIR = Path("models")
FEATURES   = ["fico_score", "dti", "int_rate", "annual_inc", "grade_idx", "term"]


def fit_pd_model(df_train: pd.DataFrame) -> tuple:
    """
    Fit logistic regression for PD estimation.

    Features:
        fico_score  : credit score at origination
        dti         : debt-to-income ratio
        int_rate    : loan interest rate (proxy for lender risk assessment)
        annual_inc  : borrower income
        grade_idx   : ordinal loan grade (A=0 ... G=6)
        term        : loan term in months
    """
    X = df_train[FEATURES].copy()
    y = df_train["default"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    train_auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
    print(f"[pd_model] Train AUC: {train_auc:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model,  "models/pd_model.pkl")
    joblib.dump(scaler, "models/pd_scaler.pkl")
    return model, scaler


def predict_pd(
    model,
    scaler,
    df: pd.DataFrame,
) -> np.ndarray:
    """Return P(default) for each loan."""
    X = scaler.transform(df[FEATURES])
    return model.predict_proba(X)[:, 1]


def evaluate_pd(y_true, y_pred_prob) -> dict:
    """Compute PD model metrics."""
    from sklearn.metrics import roc_auc_score, brier_score_loss
    auc    = roc_auc_score(y_true, y_pred_prob)
    brier  = brier_score_loss(y_true, y_pred_prob)
    print(f"[pd_model] Test AUC:    {auc:.4f}")
    print(f"[pd_model] Brier score: {brier:.4f}")
    return {"auc": round(auc, 4), "brier": round(brier, 4)}