import json
import logging
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV,StratifiedKFold,cross_validate,)
from sklearn.metrics import ( make_scorer, roc_auc_score, average_precision_score, f1_score,)
from sklearn.neighbors import KNeighborsClassifier




warnings.filterwarnings("ignore")

 
# Logging
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

 
# Config
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG = {
    "processed_dir": os.getenv("PROCESSED_DIR", str(BASE_DIR / "data" / "processed")),
    "models_dir": os.getenv("MODELS_DIR", str(BASE_DIR / "models")),
    "reports_dir": os.getenv("REPORTS_DIR",   str(BASE_DIR / "reports")),

    # Cross-validation strategy
    "cv_folds": 5,
    "random_state":42,

    # Scoring metrics — we track multiple, select on AUC-ROC
    # Rationale: accuracy is misleading on imbalanced data (26.5% churn).
    # AUC-ROC measures discrimination ability regardless of threshold.
    # Average precision summarises the precision-recall curve.
    # F1 balances precision and recall at the default 0.5 threshold.
    "primary_metric": "roc_auc",
}

 
# Scoring suite
 
SCORING = {
    "roc_auc": "roc_auc",
    "avg_precision": make_scorer(average_precision_score, response_method="predict_proba"),
    "f1":  make_scorer(f1_score),
}


 
# Step 1 — Load artefacts
def load_artefacts() -> tuple:
    """Load preprocessed splits and feature names produced by 01_preprocessing.py."""
    processed_dir = Path(CONFIG["processed_dir"])
    models_dir = Path(CONFIG["models_dir"])

    def _load(path):
        assert path.exists(), f"File not found: {path}. Run 01_preprocessing.py first."
        with open(path, "rb") as f:
            return pickle.load(f)

    X_train = _load(processed_dir / "X_train.pkl")
    X_test  = _load(processed_dir / "X_test.pkl")
    y_train = _load(processed_dir / "y_train.pkl")
    y_test  = _load(processed_dir / "y_test.pkl")

    with open(models_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    log.info("Loaded — X_train: %s  X_test: %s", X_train.shape, X_test.shape)
    log.info("Churn rate — train: %.1f%%  test: %.1f%%",
             np.mean(y_train) * 100, np.mean(y_test) * 100)
    return X_train, X_test, y_train, y_test, feature_names


 
# Step 2 — Define candidate models
 
def get_candidate_models() -> dict:

    rs = CONFIG["random_state"]

    return {
        "Dummy (majority)": DummyClassifier(
            strategy="most_frequent",
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=rs,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=rs,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=rs,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
    }


 
# Step 3 — Cross-validate all candidates
 
def run_cross_validation( models: dict, X_train: np.ndarray, y_train: pd.Series,) -> pd.DataFrame:
    cv = StratifiedKFold(
        n_splits=CONFIG["cv_folds"],
        shuffle=True,
        random_state=CONFIG["random_state"],
    )

    results = []
    for name, model in models.items():
        log.info("Cross-validating: %-25s", name)
        start = time.time()

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            return_train_score=True,  # detect overfitting
            n_jobs=-1,
        )

        elapsed = time.time() - start

        row = {"model": name, "fit_time_s": round(elapsed, 2)}
        for metric in SCORING:
            test_scores  = scores[f"test_{metric}"]
            train_scores = scores[f"train_{metric}"]
            row[f"{metric}_mean"]       = round(test_scores.mean(), 4)
            row[f"{metric}_std"]        = round(test_scores.std(), 4)
            row[f"{metric}_train_mean"] = round(train_scores.mean(), 4)

        results.append(row)
        log.info(
            "  AUC-ROC: %.4f ± %.4f  |  AvgPrecision: %.4f  |  F1: %.4f  |  %.1fs",
            row["roc_auc_mean"], row["roc_auc_std"],
            row["avg_precision_mean"],
            row["f1_mean"],
            elapsed,
        )

    df_results = pd.DataFrame(results).sort_values(
        "roc_auc_mean", ascending=False
    ).reset_index(drop=True)

    return df_results


 
# Step 4 — Plot CV results
 
def plot_cv_results(df_results: pd.DataFrame, reports_dir: Path) -> None:

    metrics = ["roc_auc_mean", "avg_precision_mean", "f1_mean"]
    labels  = ["AUC-ROC", "Avg Precision", "F1"]
    colors  = ["#3498db", "#2ecc71", "#e67e22"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Cross-validation results ({CONFIG['cv_folds']}-fold StratifiedKFold)",
        fontsize=13,
    )

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        std_col = metric.replace("_mean", "_std") if metric != "avg_precision_mean" else None
        yer = df_results[std_col].values if std_col in df_results else None

        bars = ax.barh(
            df_results["model"],
            df_results[metric],
            xerr=yerr,
            color=color,
            alpha=0.85,
            capsize=4,
        )
        ax.set_title(label)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Score")
        ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8)

        for bar, val in zip(bars, df_results[metric]):
            ax.text(
                val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9,
            )

    plt.tight_layout()
    path = reports_dir / "07_cv_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved plot: %s", path)


 
# Step 5 — Overfitting check
 
def plot_overfit_check(df_results: pd.DataFrame, reports_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    x      = np.arange(len(df_results))
    width  = 0.35

    ax.bar(x - width/2, df_results["roc_auc_train_mean"],
           width, label="Train AUC-ROC", color="#3498db", alpha=0.8)
    ax.bar(x + width/2, df_results["roc_auc_mean"],
           width, label="Val AUC-ROC",   color="#e74c3c", alpha=0.8,
           yerr=df_results["roc_auc_std"], capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(df_results["model"], rotation=15, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Train vs Validation AUC-ROC — overfitting check")
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    path = reports_dir / "08_overfit_check.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved plot: %s", path)


 
# Step 6 — Hyperparameter tuning on best model
 
def tune_best_model(
    best_name: str,
    X_train: np.ndarray,
    y_train: pd.Series,
) -> tuple:
    cv = StratifiedKFold(
        n_splits=CONFIG["cv_folds"],
        shuffle=True,
        random_state=CONFIG["random_state"],
    )

    # Parameter grids per model
    param_grids = {
        "Logistic Regression": {
            "C":       [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver":  ["liblinear"],
        },
        "Random Forest": {
            "n_estimators":      [100, 200],
            "max_depth":         [None, 10, 20],
            "min_samples_split": [2, 5],
            "max_features":      ["sqrt", "log2"],
        },
        "Gradient Boosting": {
            "n_estimators":  [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth":     [3, 5],
            "subsample":     [0.8, 1.0],
        },
        "KNN": {
            "n_neighbors": [3, 5, 9, 15],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan"],
        },
    }

    rs = CONFIG["random_state"]

    # Base models for tuning (fresh instances)
    base_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=rs
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=rs, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
        "KNN": KNeighborsClassifier(n_jobs=-1),
    }

    if best_name not in param_grids:
        log.warning("No param grid for '%s'. Skipping tuning.", best_name)
        return base_models.get(best_name), {}

    log.info("Tuning: %s", best_name)
    log.info("Parameter grid: %s", param_grids[best_name])

    grid_search = GridSearchCV(
        estimator=base_models[best_name],
        param_grid=param_grids[best_name],
        cv=cv,
        scoring=CONFIG["primary_metric"],
        n_jobs=-1,
        verbose=1,
        refit=True,  # refit best params on full X_train after search
    )

    grid_search.fit(X_train, y_train)

    log.info("Best params:  %s", grid_search.best_params_)
    log.info("Best CV AUC-ROC: %.4f", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_params_


 
# Step 7 — Save artefacts
 
def save_artefacts(
    best_model,
    best_name: str,
    best_params: dict,
    df_results: pd.DataFrame,
    models_dir: Path,
    reports_dir: Path,
) -> None:
    # Save model
    model_path = models_dir / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    log.info("Saved best model: %s", model_path)

    # Save CV report as JSON
    report = {
        "best_model_name":   best_name,
        "best_params":       best_params,
        "cv_folds":          CONFIG["cv_folds"],
        "primary_metric":    CONFIG["primary_metric"],
        "cv_results":        df_results.to_dict(orient="records"),
    }
    report_path = reports_dir / "cv_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved CV report: %s", report_path)

    # Save CV results as CSV for quick inspection
    csv_path = reports_dir / "cv_results.csv"
    df_results.to_csv(csv_path, index=False)
    log.info("Saved CV table: %s", csv_path)


 
# Orchestrator
 
def train() -> None:
    log.info("=" * 55)
    log.info("  Starting training pipeline")
    log.info("=" * 55)

    reports_dir = Path(CONFIG["reports_dir"])
    models_dir  = Path(CONFIG["models_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    X_train, X_test, y_train, y_test, feature_names = load_artefacts()

    # 2. Define candidates
    models = get_candidate_models()

    # 3. Cross-validate all candidates
    log.info("-" * 55)
    log.info("Running %d-fold cross-validation on %d models...",
             CONFIG["cv_folds"], len(models))
    log.info("-" * 55)
    df_results = run_cross_validation(models, X_train, y_train)

    # 4. Print leaderboard
    log.info("\n%s", "=" * 55)
    log.info("LEADERBOARD (sorted by AUC-ROC)")
    log.info("%s", "=" * 55)
    display_cols = ["model", "roc_auc_mean", "roc_auc_std",
                    "avg_precision_mean", "f1_mean", "fit_time_s"]
    log.info("\n%s", df_results[display_cols].to_string(index=False))

    # 5. Plot results
    plot_cv_results(df_results, reports_dir)
    plot_overfit_check(df_results, reports_dir)

    # 6. Select best (excluding Dummy)
    real_models = df_results[df_results["model"] != "Dummy (majority)"]
    best_name   = real_models.iloc[0]["model"]
    best_cv_auc = real_models.iloc[0]["roc_auc_mean"]
    log.info("\nBest model: %s  (CV AUC-ROC: %.4f)", best_name, best_cv_auc)

    # 7. Tune best model
    log.info("-" * 55)
    log.info("Hyperparameter tuning...")
    log.info("-" * 55)
    best_model, best_params = tune_best_model(best_name, X_train, y_train)

    # 8. Save
    save_artefacts(
        best_model, best_name, best_params,
        df_results, models_dir, reports_dir,
    )

    log.info("=" * 55)
    log.info("  Training complete. Ready for 03_evaluate.py")
    log.info("=" * 55)


 
# Entry point
 
if __name__ == "__main__":
    train()