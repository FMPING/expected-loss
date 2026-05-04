# Expected Loss Framework — Basel III Credit Risk Provisioning

A Basel III-compliant Expected Loss (EL) framework for a term-loan portfolio.
Implements the full PD → LGD → EAD pipeline and generates a loan-loss
provisioning report broken down by risk grade.

**Core formula:** `EL = PD × LGD × EAD`

---

## Model performance (test set — 15,000 loans)

### PD — Probability of Default (Logistic Regression)

| Metric | Value | Benchmark |
|--------|-------|-----------|
| AUC-ROC | **0.7040** | > 0.65 = acceptable |
| Brier score | **0.1166** | < 0.15 = good calibration |

### LGD — Loss Given Default (Beta Regression)

| Metric | Value |
|--------|-------|
| Train MAE | **0.1623** |
| Direction | Grade A ≈ 30% loss · Grade G ≈ 78% loss |

Beta regression is the canonical approach for proportions bounded in (0, 1).
Fitted on defaulted loans only, as required under Basel III IRB.

### EAD — Exposure at Default (Gradient Boosting)

| Metric | Value |
|--------|-------|
| Test R² | **0.2509** |
| Test MAE | **0.0959** (9.6pp error on CCF) |

CCF (Credit Conversion Factor) is driven by time-to-default: riskier borrowers
default earlier, leaving more principal outstanding. R² of 0.25 is typical for
term-loan EAD without behavioural payment data.

---

## Portfolio provisioning output

Results on a synthetic 50,000-loan portfolio (LendingClub-style):

```
Total EAD:            $  66,311,283
Total Expected Loss:  $   6,453,125
EL Rate (EL/EAD):           9.73%
Actual default rate:         14.74%
```

| Grade | Mean PD | Mean LGD | EL Rate | Provision |
|-------|---------|----------|---------|-----------|
| A | 4.7% | 29.1% | 1.46% | $84,360 |
| B | 7.4% | 36.8% | 2.83% | $340,616 |
| C | 11.7% | 45.2% | 5.51% | $944,121 |
| D | 18.1% | 53.9% | 10.13% | $1,512,493 |
| E | 25.6% | 62.3% | 16.59% | $1,419,390 |
| F | 33.3% | 70.0% | 24.34% | $1,204,917 |
| G | 40.4% | 76.6% | 32.00% | $947,228 |

Both PD and LGD increase monotonically with grade risk — EL rate cascade
from 1.46% (A) to 32.00% (G) is consistent with real bank provisioning reports.

---

## Project structure

```
expected-loss/
├── run_pipeline.py          # entry point — runs full EL pipeline
├── src/
│   ├── data_prep.py         # synthetic portfolio generation
│   ├── pd_model.py          # Probability of Default (logistic regression)
│   ├── lgd_model.py         # Loss Given Default (beta regression, MLE)
│   ├── ead_model.py         # Exposure at Default (gradient boosting)
│   └── expected_loss.py     # EL calculation + provisioning report
├── models/                  # serialised .pkl artifacts (not committed)
├── notebooks/               # exploratory notebooks
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
git clone https://github.com/FMPING/expected-loss.git
cd expected-loss
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the pipeline

```bash
python run_pipeline.py
```

This generates the portfolio, fits all three models, and prints the
provisioning report to stdout. Serialised models are saved to `models/`.

---

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Logistic regression for PD | Linear in log-odds, interpretable coefficients, standard in Basel IRB models. |
| Beta regression for LGD | LGD is bounded in (0, 1) with mass near the boundaries. OLS violates these bounds and produces biased estimates. |
| Gradient Boosting for EAD | CCF has a non-linear relationship with borrower risk. GBM captures interactions between FICO, DTI, grade, and term that Ridge cannot. |
| EAD via amortisation formula | CCF is derived from when in the loan lifecycle a borrower defaults. Risk score (FICO + DTI + grade) drives time-to-default, which determines remaining balance. |
| LGD fitted on defaulted loans only | Basel III IRB Advanced requires LGD estimation on the defaulted sub-population. Applied to the full portfolio during EL calculation — represents loss conditional on default. |
| `payment_to_income` as EAD feature | Monthly payment ÷ monthly income captures affordability stress — the primary driver of how quickly a borrower exhausts capacity to service the debt. |

---

## Changelog

### v1.1 — EAD and LGD bug fixes
- **EAD data generation fixed:** `ead_util` was previously a random beta draw
  with no relationship to loan features, making R² ≈ 0.0001 mathematically
  guaranteed. Now derived via the standard amortisation formula — riskier
  borrowers default earlier, leaving more balance outstanding.
- **EAD model upgraded:** replaced Ridge regression with GradientBoostingRegressor;
  added `payment_to_income` engineered feature. R² improved from ~0.0001 to 0.25.
  Added held-out test-set evaluation (previously only reported train metrics).
- **LGD data generation fixed:** `lgd_mean` was used as the beta distribution
  mean for *recovery* then inverted, reversing the grade relationship (Grade A
  had 70% LGD, Grade G 22%). Now samples LGD directly — Grade A ≈ 30%,
  Grade G ≈ 78%, consistent with Basel III expectations.
- **requirements.txt:** replaced UTF-16 bloated file with a clean pip freeze
  from the dedicated project venv.

### v1.0 — Initial framework
- PD, LGD, EAD models with end-to-end pipeline
- Provisioning report by loan grade

---

## Tech stack

`Python 3.x` · `pandas` · `numpy` · `scikit-learn` · `scipy` · `joblib`
