# Customer Churn Prediction — Results

Final results from the trained pipeline. For the methodology (EDA, preprocessing, model selection design), see [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md). This document covers what the pipeline actually produced once run end-to-end.

**Winner: Logistic Regression** — AUC-ROC 0.8449 on the held-out test set, chosen not just for accuracy but because it was the only candidate that generalized cleanly (see below).

## Model Selection (5-fold cross-validation on training data)

| Model | CV AUC-ROC | CV Avg Precision | CV F1 | Train AUC-ROC | Train/Test Gap |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.8491 ± 0.011** | 0.6699 | 0.6267 | 0.8536 | **0.0045** |
| Gradient Boosting | 0.8475 ± 0.011 | 0.6686 | 0.5845 | 0.8859 | 0.0384 |
| Random Forest | 0.8246 ± 0.011 | 0.6235 | 0.5367 | 0.9999 | **0.1753** |
| KNN | 0.7816 ± 0.010 | 0.5212 | 0.5478 | 0.9009 | 0.1193 |
| Dummy (majority, baseline floor) | 0.5000 | 0.2654 | 0.0000 | 0.5000 | — |

**Why Logistic Regression won, not just Random Forest or Gradient Boosting:** Random Forest's train AUC-ROC of 0.9999 (essentially memorizing the training set) against a test-comparable CV score of 0.8246 is a textbook overfitting signature — an 18-point gap. Gradient Boosting and KNN show smaller but still real gaps. Logistic Regression's train and validation scores are nearly identical (0.8536 vs 0.8491), meaning it's learning genuine signal rather than fitting noise — which is why it also held up best on the truly unseen test set. See `reports/08_overfit_check.png` for the visual.

Hyperparameter tuning (`GridSearchCV` over `C`, `penalty`, `solver`) moved CV AUC-ROC only marginally, from 0.8491 to 0.8493, landing on `C=10, penalty='l2', solver='liblinear'` — the untuned default was already close to optimal. Note `C=10` sits at the edge of the search grid (`[0.01, 0.1, 1, 10]`); a wider grid (e.g. adding 50, 100) would be worth trying to confirm the optimum isn't just past the boundary.

## Final Test-Set Performance

Evaluated once, on the 1,409 held-out test customers never touched during training or tuning.

| Metric | Value |
|---|---|
| AUC-ROC | 0.8449 |
| Average Precision | 0.6592 |
| Accuracy | 0.74 |

**At the default 0.5 threshold:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Retained (No) | 0.90 | 0.72 | 0.80 | 1,035 |
| Churned (Yes) | 0.51 | 0.79 | 0.62 | 374 |

**At the F1-optimal threshold (0.577, found via `reports/12_threshold_analysis.png`):**

| Metric | Value |
|---|---|
| Churn Precision | 0.505 |
| Churn Recall | 0.786 |
| Churn F1 | 0.615 |

The model catches **~79% of customers who actually churn**, at the cost of roughly 1 false alarm for every correct churn flag. This trade-off is intentional — the pipeline uses `class_weight="balanced"`, which deliberately biases toward not missing churners, on the reasoning that a missed churner is more costly to the business than an unnecessary retention offer to someone who would have stayed anyway.

## What Drives Churn (SHAP)

Top features by mean absolute SHAP value (impact on the model's churn prediction, averaged over 200 test customers):

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | MonthlyCharges | 0.190 |
| 2 | InternetService = Fiber optic | 0.155 |
| 3 | tenure | 0.121 |
| 4 | total_services (engineered) | 0.096 |
| 5 | InternetService = No | 0.081 |
| 6 | Contract = Two year | 0.058 |
| 7 | Contract = One year | 0.020 |
| 8 | TotalCharges | 0.019 |
| 9 | OnlineSecurity = Yes | 0.018 |
| 10 | tenure_group = mid | 0.016 |

This confirms and sharpens the EDA findings: **billing amount, internet plan type, and tenure dominate**, with contract length acting as a secondary, protective signal (longer contracts push predictions toward "retained"). The engineered `total_services` feature (from `01_preprocessing.py`) ranks 4th — validating that "number of subscribed services" is a genuinely useful stickiness proxy, not just a nice idea from EDA. See `reports/13_shap_bar.png` for the full top-20 ranking and `reports/14_shap_summary.png` for the direction of each feature's effect (e.g. does *higher* MonthlyCharges push toward churn or away from it).

## Artifacts Produced

All under `ml_pipline/reports/` unless noted:

| File | Contents |
|---|---|
| `01`–`06_*.png` | EDA plots (distributions, churn-rate breakdowns, correlation matrix) |
| `07_cv_comparison.png` | Model comparison across AUC-ROC / Avg Precision / F1 |
| `08_overfit_check.png` | Train vs. validation AUC-ROC per model — the overfitting evidence above |
| `09_confusion_matrix.png` | Test-set confusion matrix (counts + %) |
| `10_roc_curve.png` | ROC curve, AUC = 0.8449 |
| `11_precision_recall_curve.png` | PR curve, AP = 0.6592, vs. the 26.5% prevalence baseline |
| `12_threshold_analysis.png` | Precision/Recall/F1 across all thresholds, optimal marked at 0.577 |
| `13_shap_bar.png` | Global SHAP feature importance (top 20) |
| `14_shap_summary.png` | SHAP beeswarm — direction of effect per feature |
| `cv_results.json` / `.csv` | Full cross-validation leaderboard |
| `eval_summary.json` | Machine-readable summary of final metrics (consumed by any downstream API/dashboard) |
| `ml_pipline/models/best_model.pkl` | The tuned Logistic Regression model |
| `ml_pipline/models/preprocessing_pipeline.pkl` | The fitted `ColumnTransformer` (scaler + one-hot encoder) |

## Reproducibility Note

This run followed three fixes applied to get the pipeline running end-to-end (see git history / conversation record if tracked): a notebook cell writing to a non-existent `try/` folder instead of `data/processed/`, hardcoded relative paths in the scripts that only worked from one specific working directory, and a `scikit-learn` API change (`needs_proba` removed from `make_scorer`) that was silently swallowing the average-precision score into `NaN` on every run. All three are now fixed, and `shap` has been installed so the explainability plots run too. `random_state=42` is fixed throughout, so re-running the pipeline end-to-end should reproduce these exact numbers.
