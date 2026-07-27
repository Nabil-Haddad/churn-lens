# Customer Churn Prediction — Project Summary

Predicting customer churn on the Telco Customer Churn dataset (7,043 customers, 21 columns) using a staged, leakage-aware ML pipeline: EDA → preprocessing → model selection → evaluation/explainability.

> **Status:** Complete and reproducible. The pipeline has been run end-to-end (`random_state=42` fixed throughout); final numbers are in [Results](#results-summary) and in the standalone `RESULTS.md`. For a quick evaluation, start with the top-level `README.md`.

## Pipeline Overview

```
ml_pipline/
├── data/
│   ├── raw/Telco_Data.csv          # source dataset (7043 rows × 21 cols)
│   └── processed/                  # df_clean.csv + X/y train-test splits (generated)
├── notebooks/
│   └── EDA.ipynb                   # Stage 1–6 exploratory analysis + cleaning
├── src/
│   ├── 01_preprocessing.py         # feature engineering + train/test split + encoding
│   ├── 02_train_crossval.py        # model comparison + hyperparameter tuning
│   └── 03_evaluate.py              # test-set evaluation, curves, SHAP explainability
├── models/                         # fitted pipeline, best model, feature names (generated)
└── reports/                        # all plots + JSON/CSV result summaries (generated)
```

## 1. Exploratory Data Analysis (`EDA.ipynb`)

Conducted in six staged sections, each ending in a written findings summary.

### Stage 1 — Data loading & first inspection
- Shape: 7,043 rows × 21 columns.
- `TotalCharges` loaded as `object` — should be numeric.
- `SeniorCitizen` is `int64` but logically a binary category.
- `customerID` is an identifier, not a feature.

### Stage 2 — Target variable analysis
- Churn rate: **26.5%** (≈2.7:1 imbalance).
- Accuracy would be misleading → chose **AUC-ROC** and **precision-recall** as primary metrics.
- Imbalance judged not severe enough to require SMOTE.

### Stage 3 — Univariate EDA
- `tenure` is bimodal (many brand-new customers, many long-tenured) — flagged as a candidate for binning.
- `MonthlyCharges` roughly uniform; `TotalCharges` right-skewed (≈ tenure × MonthlyCharges).
- No numerical feature is normally distributed (confirmed via Q-Q plots) — noted for downstream statistical test validity.
- Six service columns carry a redundant `"No internet service"` category duplicating `InternetService` — to be collapsed into `"No"`.

### Stage 4 — Bivariate EDA
**Strongest churn predictors identified:**
- `Contract`: month-to-month churns at ~43% vs. ~3% for two-year contracts.
- `InternetService`: Fiber optic churns far more than DSL (~42% vs. ~19%).
- `TechSupport` / `OnlineSecurity`: absence roughly doubles churn.
- `PaymentMethod`: electronic-check payers churn more than automatic-payment customers.

**Numerical features:** churned customers have lower `tenure` and slightly higher `MonthlyCharges` (Mann-Whitney p << 0.05 for all three numerical features).

**Multicollinearity:** `TotalCharges` correlates strongly with `tenure` (r ≈ 0.83) and `MonthlyCharges` (r ≈ 0.65) — expected, since it's roughly their product. Flagged as a concern for linear models (tree models unaffected).

### Stage 5 — Data quality audit
- `TotalCharges` had 11 hidden missing values stored as whitespace strings (not `NaN`) — all belong to customers with `tenure = 0`.
- No duplicate rows or duplicate `customerID`s.
- No data leakage: all features are observable before the churn event.

### Stage 6 — Data cleaning (`df_clean`)
| Fix | Detail |
|---|---|
| `TotalCharges` missing values | Imputed to `0` (new customers, no charges yet) |
| `"No internet/phone service"` categories | Collapsed to `"No"` across 7 service columns |
| `customerID` | Dropped (identifier, not predictive) |
| Post-clean assertions | No nulls, `TotalCharges` is float, `Churn` is binary |
| Persistence | `df_clean` written to `data/processed/df_clean.csv` for the preprocessing script |

## 2. Preprocessing (`01_preprocessing.py`)

- Loads `df_clean.csv`, asserts target column present and no residual nulls.
- **Feature engineering:**
  - `charges_per_tenure` = `MonthlyCharges / (tenure + 1)` — cost intensity signal.
  - `tenure_group` — bins tenure into `new` / `mid` / `loyal` (captures the bimodal pattern from EDA Stage 3).
  - `total_services` — count of subscribed "Yes" services, as a stickiness proxy.
- Stratified 80/20 train/test split (preserves the 26.5% churn ratio in both splits).
- `ColumnTransformer`: `StandardScaler` on numerical/binary features, `OneHotEncoder(drop="first", handle_unknown="ignore")` on categoricals — fit on train only to avoid leakage.
- Persists `X_train/X_test/y_train/y_test` (pickle), the fitted pipeline, and feature names for reuse downstream.

## 3. Model Training & Selection (`02_train_crossval.py`)

- **Candidates:** Dummy (majority-class baseline), Logistic Regression, Random Forest, Gradient Boosting, KNN — most with `class_weight="balanced"` to counter imbalance.
- **Evaluation:** 5-fold `StratifiedKFold` cross-validation scored on AUC-ROC, average precision, and F1, with train scores also captured to detect overfitting.
- Produces a leaderboard plus two diagnostic plots: metric comparison across models, and train-vs-validation AUC-ROC (overfitting check).
- **Hyperparameter tuning:** `GridSearchCV` (nested within the same CV strategy, so the test set stays untouched) on the top candidate, with per-model parameter grids.
- Saves the tuned best model and a full CV report (JSON + CSV).

## 4. Evaluation (`03_evaluate.py`)

Runs exclusively against the held-out test set (train data is never loaded here, by design).

- Classification report, confusion matrix (counts + row-normalized %).
- ROC curve (AUC) and Precision-Recall curve (average precision) — PR curve emphasized as more informative given the minority-class churn rate.
- **Threshold analysis:** sweeps precision/recall/F1 across all thresholds and reports the F1-optimal cutoff, since missing a churner (FN) is framed as costlier than a false alarm (FP).
- **SHAP explainability:** `TreeExplainer` for tree models / `KernelExplainer` otherwise, producing a global importance bar chart and a summary beeswarm plot (direction of effect per feature).
- Writes `eval_summary.json` — intended to feed a downstream API/dashboard for model metadata.

## Results Summary

Full breakdown in `RESULTS.md`. Headlines:

- **Winner: Logistic Regression**, AUC-ROC **0.8449** on the held-out test set — chosen because it was the only candidate that generalized cleanly (train/validation gap of 0.0045 vs Random Forest's 0.1753 overfitting gap).
- At the F1-optimal threshold (0.577): churn recall **0.786**, churn precision 0.505 — the model catches ~79% of actual churners, a deliberate trade-off driven by `class_weight="balanced"`.
- **Top SHAP drivers:** MonthlyCharges, Fiber-optic internet, tenure, then the engineered `total_services` feature (4th), confirming it as a genuine stickiness proxy rather than just an EDA hunch.

## Key Findings (from EDA)

1. **Contract type** is the single strongest churn driver — month-to-month customers churn at over 10× the rate of two-year contract customers.
2. **Fiber-optic internet** and **lack of tech support/online security** are secondary strong signals.
3. New customers (low tenure) are disproportionately likely to churn — motivating the `tenure_group` and `charges_per_tenure` engineered features.
4. Class imbalance (26.5% positive) shapes every later design choice: stratified splits, `class_weight="balanced"`, AUC-ROC/PR as primary metrics instead of accuracy, and F1-based threshold tuning instead of a fixed 0.5 cutoff.


## Resolved Setup Notes

The following were fixed to get the pipeline running end-to-end, and are recorded here for reproducibility:

- `EDA.ipynb` now writes `df_clean` to `data/processed/df_clean.csv` (previously it was built but never persisted, so `01_preprocessing.py` had nothing to load).
- A notebook cell that wrote to a non-existent `try/` folder was corrected to `data/processed/`.
- Hardcoded relative paths in the scripts (which only worked from one specific working directory) were fixed.
- A `scikit-learn` API change (`needs_proba` removed from `make_scorer`) was silently swallowing the average-precision score into `NaN` on every run — now fixed.
- `shap` added to dependencies so the explainability plots run.
- `02_train crossval.py` renamed to `02_train_crossval.py` (removing the space) to match its own usage docstring.
- `requirements.txt` added.

**Outstanding cosmetic item:** the project folder is still named `ml_pipline/` (typo) rather than `ml_pipeline/`. Scripts run correctly against the current name; renaming is optional but would align with the docstrings.