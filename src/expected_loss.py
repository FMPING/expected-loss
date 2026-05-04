"""
expected_loss.py
----------------
Expected Loss (EL) calculation and portfolio provisioning report.

Formula (Basel III):
    EL = PD × LGD × EAD

Where:
    PD  = Probability of Default (model output)
    LGD = Loss Given Default     (model output)
    EAD = Exposure at Default    (model output)

EL represents the average loss a lender expects from a loan over a
given horizon. Summing EL across the portfolio gives the required
provision (loan loss reserve).
"""

import numpy as np
import pandas as pd


def compute_expected_loss(
    pd_values:  np.ndarray,
    lgd_values: np.ndarray,
    ead_values: np.ndarray,
) -> np.ndarray:
    """
    Compute Expected Loss per loan.
    EL = PD × LGD × EAD
    """
    return pd_values * lgd_values * ead_values


def provisioning_report(
    df: pd.DataFrame,
    el: np.ndarray,
    pd_values: np.ndarray,
    lgd_values: np.ndarray,
    ead_values: np.ndarray,
) -> pd.DataFrame:
    """
    Generate a portfolio provisioning report by loan grade.

    Shows expected loss, provision amount, and coverage ratio
    broken down by risk segment — mirrors standard credit risk
    reporting in banks and fintechs.
    """
    results = df[["loan_id", "grade", "loan_amnt", "default"]].copy()
    results["pd"]            = pd_values.round(4)
    results["lgd"]           = lgd_values.round(4)
    results["ead"]           = ead_values.round(2)
    results["expected_loss"] = el.round(2)
    results["el_rate"]       = (el / ead_values).round(4)

    # Portfolio summary
    print("\n" + "=" * 60)
    print("EXPECTED LOSS — PORTFOLIO PROVISIONING REPORT")
    print("=" * 60)
    print(f"Total loans:          {len(results):,}")
    print(f"Total EAD:            ${ead_values.sum():>15,.0f}")
    print(f"Total Expected Loss:  ${el.sum():>15,.0f}")
    print(f"EL Rate (EL/EAD):     {el.sum()/ead_values.sum():>14.2%}")
    print(f"Actual default rate:  {df['default'].mean():>14.2%}")

    # By grade
    print("\nBreakdown by loan grade:")
    print("-" * 60)
    grade_summary = results.groupby("grade").agg(
        n_loans      = ("loan_id",       "count"),
        total_ead    = ("ead",           "sum"),
        total_el     = ("expected_loss", "sum"),
        mean_pd      = ("pd",            "mean"),
        mean_lgd     = ("lgd",           "mean"),
        actual_dr    = ("default",       "mean"),
    ).round(4)
    grade_summary["el_rate"]   = (grade_summary["total_el"] / grade_summary["total_ead"]).round(4)
    grade_summary["provision"] = grade_summary["total_el"].map("${:,.0f}".format)
    print(grade_summary[["n_loans","mean_pd","mean_lgd",
                          "el_rate","provision","actual_dr"]].to_string())
    print("=" * 60)

    return results, grade_summary


def run_el_pipeline(df_train, df_test) -> dict:
    """
    Full Expected Loss pipeline:
        1. Fit PD, LGD, EAD models on training data
        2. Predict on test set
        3. Compute EL = PD × LGD × EAD
        4. Generate provisioning report
    """
    from src.pd_model  import fit_pd_model,  predict_pd,  evaluate_pd
    from src.lgd_model import fit_lgd_model, predict_lgd
    from src.ead_model import fit_ead_model, predict_ead

    print("\n--- Fitting PD model ---")
    pd_model,  pd_scaler  = fit_pd_model(df_train)

    print("\n--- Fitting LGD model ---")
    lgd_model, lgd_scaler = fit_lgd_model(df_train)

    print("\n--- Fitting EAD model ---")
    ead_model, ead_scaler = fit_ead_model(df_train, df_test)

    print("\n--- Predicting on test set ---")
    pd_pred  = predict_pd(pd_model,   pd_scaler,  df_test)
    lgd_pred = predict_lgd(lgd_model, lgd_scaler, df_test)
    ead_pred = predict_ead(ead_model,  ead_scaler, df_test)

    print("\n--- PD model evaluation ---")
    evaluate_pd(df_test["default"].values, pd_pred)

    el = compute_expected_loss(pd_pred, lgd_pred, ead_pred)

    results, grade_summary = provisioning_report(
        df_test, el, pd_pred, lgd_pred, ead_pred
    )

    return {
        "results":       results,
        "grade_summary": grade_summary,
        "total_el":      el.sum(),
        "el_rate":       el.sum() / ead_pred.sum(),
        "pd_model":      pd_model,
        "lgd_model":     lgd_model,
        "ead_model":     ead_model,
    }