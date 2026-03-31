"""
lgd_model.py
------------
Loss Given Default (LGD) model.

LGD = fraction of exposure lost when a loan defaults.
Range: 0 (full recovery) to 1 (zero recovery).

Why beta regression?
    LGD is bounded between 0 and 1, with mass concentrated near the
    boundaries. Standard linear regression violates these bounds and
    produces biased estimates. Beta regression is the canonical approach
    for modelling rates and proportions in (0, 1).

Basel III context:
    LGD must reflect downturn conditions (Downturn LGD).
    Typically estimated on defaulted loan observations only.
"""

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from scipy.special import gammaln
from pathlib import Path

MODELS_DIR = Path("models")
FEATURES   = ["fico_score", "dti", "grade_idx", "int_rate", "loan_amnt"]


# ---------------------------------------------------------------------------
# Beta regression implementation
# ---------------------------------------------------------------------------

def _beta_log_likelihood(params, X, y):
    """
    Negative log-likelihood for beta regression.
    params = [intercept, coef1, ..., coefN, log_phi]
    """
    n_features = X.shape[1]
    beta  = params[:n_features + 1]
    phi   = np.exp(params[n_features + 1])   # precision parameter

    Xb    = np.column_stack([np.ones(len(X)), X]) @ beta
    mu    = 1 / (1 + np.exp(-Xb))            # logistic link
    mu    = np.clip(mu, 1e-6, 1 - 1e-6)

    alpha = mu * phi
    beta_ = (1 - mu) * phi
    y     = np.clip(y, 1e-6, 1 - 1e-6)

    ll = (
        gammaln(phi)
        - gammaln(alpha)
        - gammaln(beta_)
        + (alpha - 1) * np.log(y)
        + (beta_ - 1) * np.log(1 - y)
    )
    return -ll.sum()


class BetaRegression:
    """
    Simple beta regression fitted via maximum likelihood.
    Suitable for modelling LGD as a proportion in (0, 1).
    """

    def __init__(self):
        self.params_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_features_ = X.shape[1]
        n_params = self.n_features_ + 2   # intercept + coefs + log_phi
        x0 = np.zeros(n_params)

        result = minimize(
            _beta_log_likelihood,
            x0,
            args=(X, y),
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        self.params_ = result.x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        beta = self.params_[:self.n_features_ + 1]
        Xb   = np.column_stack([np.ones(len(X)), X]) @ beta
        return 1 / (1 + np.exp(-Xb))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def fit_lgd_model(df_train: pd.DataFrame) -> tuple:
    """
    Fit beta regression on defaulted loans only.

    Basel III note: LGD is estimated on the defaulted sub-population.
    Applying it to the full portfolio during EL calculation is correct —
    it represents the expected loss conditional on default.
    """
    from sklearn.preprocessing import StandardScaler

    defaulted = df_train[df_train["default"] == 1].copy()
    print(f"[lgd_model] Training on {len(defaulted):,} defaulted loans")

    X = defaulted[FEATURES].values
    y = defaulted["lgd"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = BetaRegression()
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    mae    = np.abs(y - y_pred).mean()
    print(f"[lgd_model] Train MAE: {mae:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model,  "models/lgd_model.pkl")
    joblib.dump(scaler, "models/lgd_scaler.pkl")
    return model, scaler


def predict_lgd(model, scaler, df: pd.DataFrame) -> np.ndarray:
    """Return predicted LGD for each loan."""
    X = scaler.transform(df[FEATURES])
    return model.predict(X)