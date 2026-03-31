"""
data_prep.py
------------
Data preparation for the Expected Loss framework.

Generates a synthetic LendingClub-style loan portfolio since the full
LendingClub dataset requires registration. The synthetic data mirrors
real distributions documented in LendingClub's public statistics.

In production, replace generate_synthetic_portfolio() with a loader
for the real CSV from:
https://www.kaggle.com/datasets/wordsforthewise/lending-club
"""

import numpy as np
import pandas as pd


def generate_synthetic_portfolio(
    n: int = 50000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic loan portfolio with realistic distributions.

    Columns mirror LendingClub schema:
        loan_amnt       : original loan amount
        int_rate        : annual interest rate
        annual_inc      : borrower annual income
        dti             : debt-to-income ratio
        fico_score      : credit score at origination
        loan_status     : 0 = performing, 1 = defaulted
        recovery_rate   : fraction recovered after default (LGD input)
        utilization     : credit utilisation at default (EAD input)
        grade           : loan grade A-G
    """
    rng = np.random.default_rng(random_state)
    n   = int(n)

    # Loan characteristics
    loan_amnt  = rng.lognormal(9.5, 0.6, n).clip(1000, 40000).round(-2)
    int_rate   = rng.beta(2, 8, n) * 0.25 + 0.05
    annual_inc = rng.lognormal(11.0, 0.5, n).clip(20000, 500000)
    dti        = rng.beta(3, 8, n) * 0.6
    fico_score = rng.normal(700, 50, n).clip(580, 850).round().astype(int)
    term       = rng.choice([36, 60], n, p=[0.7, 0.3])

    # Loan grade — correlated with FICO
    grade_probs = np.column_stack([
        np.where(fico_score >= 750, 0.40, 0.05),  # A
        np.where(fico_score >= 720, 0.30, 0.10),  # B
        np.ones(n) * 0.20,                         # C
        np.where(fico_score < 680, 0.25, 0.10),   # D
        np.where(fico_score < 660, 0.20, 0.05),   # E
        np.where(fico_score < 640, 0.15, 0.03),   # F
        np.where(fico_score < 620, 0.10, 0.02),   # G
    ])
    grade_probs = grade_probs / grade_probs.sum(axis=1, keepdims=True)
    grade_idx   = np.array([rng.choice(7, p=p) for p in grade_probs])
    grade       = np.array(list("ABCDEFG"))[grade_idx]

    # PD — driven by FICO, DTI, and grade
    log_odds = (
        -3.0
        - 0.008 * (fico_score - 660)
        + 2.5   * dti
        + 0.4   * grade_idx
        + rng.normal(0, 0.3, n)
    )
    pd_true = 1 / (1 + np.exp(-log_odds))
    default = rng.binomial(1, pd_true).astype(int)

    # LGD — beta distributed, higher for lower grades
    lgd_mean  = 0.30 + 0.08 * grade_idx
    lgd_alpha = lgd_mean * 5
    lgd_beta  = (1 - lgd_mean) * 5
    recovery  = rng.beta(lgd_alpha, lgd_beta, n).clip(0.01, 0.99)
    lgd       = 1 - recovery

    # EAD — utilisation fraction of loan amount
    ead_util  = rng.beta(5, 2, n).clip(0.5, 1.0)
    ead       = loan_amnt * ead_util

    df = pd.DataFrame({
        "loan_id":      range(n),
        "loan_amnt":    loan_amnt,
        "int_rate":     int_rate.round(4),
        "annual_inc":   annual_inc.round(0),
        "dti":          dti.round(4),
        "fico_score":   fico_score,
        "term":         term,
        "grade":        grade,
        "grade_idx":    grade_idx,
        "default":      default,
        "lgd":          lgd.round(4),
        "ead":          ead.round(2),
        "ead_util":     ead_util.round(4),
        "pd_true":      pd_true.round(4),
    })

    print(f"[data_prep] Portfolio: {n:,} loans")
    print(f"[data_prep] Default rate: {default.mean():.2%}")
    print(f"[data_prep] Mean LGD (defaulted): {lgd[default==1].mean():.2%}")
    print(f"[data_prep] Mean EAD: ${ead.mean():,.0f}")
    return df


def split_portfolio(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple:
    """
    Split into train/test preserving default rate.
    Returns: df_train, df_test
    """
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        df, test_size=test_size,
        stratify=df["default"],
        random_state=random_state,
    )
    print(f"[split] Train: {len(train):,} | Test: {len(test):,}")
    return train.reset_index(drop=True), test.reset_index(drop=True)