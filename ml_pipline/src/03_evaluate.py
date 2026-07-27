import json
import logging
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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

    # Plot style
    "color_positive": "#e74c3c",   # churned
    "color_negative": "#2ecc71",   # retained
    "color_primary":  "#3498db",
}

plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["figure.dpi"]        = 120


 
# Step 1 — Load artefacts
def load_artefacts() -> tuple:
    processed_dir = Path(CONFIG["processed_dir"])
    models_dir = Path(CONFIG["models_dir"])

    def _load_pkl(path):
        assert path.exists(), (
            f"File not found: {path}\n"
            f"Run 01_preprocessing.py and 02_train_crossval.py first."
        )
        with open(path, "rb") as f:
            return pickle.load(f)

    model = _load_pkl(models_dir / "best_model.pkl")
    X_test = _load_pkl(processed_dir / "X_test.pkl")
    y_test = _load_pkl(processed_dir / "y_test.pkl")

    with open(models_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    with open(Path(CONFIG["reports_dir"]) / "cv_results.json") as f:
        cv_report = json.load(f)

    log.info("Loaded model: %s", type(model).__name__)
    log.info("Test set — X: %s  |  churn rate: %.1f%%",
             X_test.shape, np.mean(y_test) * 100)

    return model, X_test, y_test, feature_names, cv_report


 
# Step 2 — Generate predictions
def get_predictions(
    model,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of churn
    y_pred       = model.predict(X_test)
    return y_pred, y_pred_proba


 
# Step 3 — Classification report
def print_classification_report(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    report_dict = classification_report(
        y_test, y_pred,
        target_names=["Retained (0)", "Churned (1)"],
        output_dict=True,
    )

    log.info("\n%s", "=" * 55)
    log.info("CLASSIFICATION REPORT (threshold = 0.5)")
    log.info("%s", "=" * 55)
    log.info("\n%s", classification_report(
        y_test, y_pred,
        target_names=["Retained (0)", "Churned (1)"],
    ))

    auc_roc  = roc_auc_score(y_test, y_pred_proba)
    avg_prec = average_precision_score(y_test, y_pred_proba)

    log.info("AUC-ROC:           %.4f", auc_roc)
    log.info("Average Precision: %.4f", avg_prec)

    return {
        "classification_report": report_dict,
        "auc_roc":               round(auc_roc,  4),
        "avg_precision":         round(avg_prec, 4),
    }


 
# Step 4 — Confusion matrix
def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    reports_dir: Path,
) -> None:
    cm      = confusion_matrix(y_test, y_pred)
    cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    labels  = [["TN", "FP"], ["FN", "TP"]]
    classes = ["Retained (0)", "Churned (1)"]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13)

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j, i,
                f"{labels[i][j]}\n{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center",
                fontsize=11, color=color, fontweight="bold",
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = reports_dir / "09_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)


 
# Step 5 — ROC curve
def plot_roc_curve(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    reports_dir: Path,
) -> None:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score            = roc_auc_score(y_test, y_pred_proba)

    # Find the point closest to the top-left corner (optimal threshold)
    optimal_idx       = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr,
            color=CONFIG["color_primary"], lw=2,
            label=f"ROC curve (AUC = {auc_score:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color=CONFIG["color_primary"])
    ax.plot([0, 1], [0, 1],
            color="gray", lw=1, linestyle="--", label="Random classifier")

    # Mark optimal threshold
    ax.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        color=CONFIG["color_positive"], zorder=5, s=80,
        label=f"Optimal threshold = {optimal_threshold:.3f}",
    )
    ax.annotate(
        f"  ({fpr[optimal_idx]:.2f}, {tpr[optimal_idx]:.2f})",
        (fpr[optimal_idx], tpr[optimal_idx]),
        fontsize=9,
    )

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = reports_dir / "10_roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)


 
# Step 6 — Precision-Recall curve
def plot_precision_recall_curve(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    reports_dir: Path,
) -> None:
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    avg_precision                 = average_precision_score(y_test, y_pred_proba)
    baseline                      = y_test.mean()

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(recall, precision,
            color=CONFIG["color_positive"], lw=2,
            label=f"PR curve (AP = {avg_precision:.4f})")
    ax.fill_between(recall, precision, alpha=0.08, color=CONFIG["color_positive"])
    ax.axhline(y=baseline, color="gray", lw=1, linestyle="--",
               label=f"Baseline (prevalence = {baseline:.2f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13)
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = reports_dir / "11_precision_recall_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)


 
# Step 7 — Threshold analysis
def plot_threshold_analysis(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    reports_dir: Path,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # precision/recall have one more element than thresholds
    thresholds_full = np.append(thresholds, 1.0)

    f1_scores = np.where(
        (precision + recall) == 0,
        0,
        2 * precision * recall / (precision + recall),
    )

    optimal_idx       = np.argmax(f1_scores)
    optimal_threshold = thresholds_full[optimal_idx]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(thresholds_full, precision,
            label="Precision", color="#3498db", lw=2)
    ax.plot(thresholds_full, recall,
            label="Recall",    color="#e74c3c", lw=2)
    ax.plot(thresholds_full, f1_scores,
            label="F1",        color="#2ecc71", lw=2)

    ax.axvline(x=0.5, color="gray", linestyle="--",
               lw=1, label="Default threshold (0.5)")
    ax.axvline(x=optimal_threshold, color="#e67e22", linestyle="--",
               lw=1.5, label=f"Optimal F1 threshold ({optimal_threshold:.3f})")

    ax.set_xlabel("Decision threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision / Recall / F1 vs Decision Threshold", fontsize=13)
    ax.legend()
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = reports_dir / "12_threshold_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)

    log.info(
        "Threshold analysis — optimal F1 threshold: %.3f  "
        "(Precision: %.3f  Recall: %.3f  F1: %.3f)",
        optimal_threshold,
        precision[optimal_idx],
        recall[optimal_idx],
        f1_scores[optimal_idx],
    )

    return float(optimal_threshold)


 
# Step 8 — SHAP feature importance
def plot_shap_importance(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    reports_dir: Path,
) -> None:
    try:
        import shap
    except ImportError:
        log.warning(
            "shap not installed. Run: pip install shap\n"
            "Skipping SHAP plots."
        )
        return

    log.info("Computing SHAP values (this may take ~30s)...")

    model_name = type(model).__name__

    def _select_churn_class(shap_values):
        if isinstance(shap_values, list):
            return shap_values[1]
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            return shap_values[:, :, 1]
        return shap_values

    # Choose the right explainer for the model type
    if model_name in ("RandomForestClassifier", "GradientBoostingClassifier",
                      "XGBClassifier", "LGBMClassifier"):
        explainer    = shap.TreeExplainer(model)
        shap_values  = explainer.shap_values(X_test)
        shap_values  = _select_churn_class(shap_values)
        X_explained  = X_test
    else:
        # Linear and other models — use KernelExplainer on a small background sample.
        # KernelExplainer is expensive, so only the first 200 test rows are explained —
        # X_explained tracks that subset so it stays aligned with shap_values below.
        background   = shap.sample(X_test, 100)
        explainer    = shap.KernelExplainer(model.predict_proba, background)
        X_explained  = X_test[:200]
        shap_values  = explainer.shap_values(X_explained, nsamples=100)
        shap_values  = _select_churn_class(shap_values)

    # --- Plot 1: Global bar chart (top 20 features) ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx       = np.argsort(mean_abs_shap)[-20:][::-1]
    top_features  = [feature_names[i] for i in top_idx]
    top_values    = mean_abs_shap[top_idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(
        range(len(top_features)), top_values[::-1],
        color=CONFIG["color_primary"], alpha=0.85,
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Mean |SHAP value|  (impact on model output)", fontsize=11)
    ax.set_title("SHAP Feature Importance — Top 20 Features", fontsize=13)

    for bar, val in zip(bars, top_values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    path = reports_dir / "13_shap_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)

    # --- Plot 2: SHAP summary dot plot (top 15 features) ---
    top15_idx = np.argsort(mean_abs_shap)[-15:]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values[:, top15_idx],
        X_explained[:, top15_idx],
        feature_names=[feature_names[i] for i in top15_idx],
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary Plot — Feature Impact & Direction", fontsize=13)
    plt.tight_layout()
    path = reports_dir / "14_shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)


 
# Step 9 — Save evaluation summary
def save_eval_summary(
    metrics: dict,
    optimal_threshold: float,
    best_name: str,
    best_params: dict,
    reports_dir: Path,
) -> None:
    summary = {
        "model_name":          best_name,
        "best_params":         best_params,
        "test_metrics": {
            "auc_roc":              metrics["auc_roc"],
            "avg_precision":        metrics["avg_precision"],
            "precision_churn":      round(metrics["classification_report"]["Churned (1)"]["precision"], 4),
            "recall_churn":         round(metrics["classification_report"]["Churned (1)"]["recall"],    4),
            "f1_churn":             round(metrics["classification_report"]["Churned (1)"]["f1-score"],  4),
            "precision_retained":   round(metrics["classification_report"]["Retained (0)"]["precision"],4),
            "recall_retained":      round(metrics["classification_report"]["Retained (0)"]["recall"],   4),
        },
        "optimal_threshold":   optimal_threshold,
        "plots": [
            "09_confusion_matrix.png",
            "10_roc_curve.png",
            "11_precision_recall_curve.png",
            "12_threshold_analysis.png",
            "13_shap_bar.png",
            "14_shap_summary.png",
        ],
    }

    path = reports_dir / "eval_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved evaluation summary: %s", path)

    # Pretty-print key numbers to terminal
    log.info("\n%s", "=" * 55)
    log.info("FINAL TEST SET RESULTS")
    log.info("%s", "=" * 55)
    log.info("Model: %s", best_name)
    log.info("AUC-ROC: %.4f", metrics["auc_roc"])
    log.info("Average Precision: %.4f", metrics["avg_precision"])
    log.info("Churn Recall: %.4f  (churners correctly caught)",summary["test_metrics"]["recall_churn"])
    log.info("Churn Precision: %.4f  (predicted churners that actually churned)",summary["test_metrics"]["precision_churn"])
    log.info("Optimal threshold: %.3f", optimal_threshold)
    log.info("%s", "=" * 55)


 
# Orchestrator
def evaluate() -> None:
    log.info("=" * 55)
    log.info("  Starting evaluation pipeline")
    log.info("=" * 55)

    reports_dir = Path(CONFIG["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    model, X_test, y_test, feature_names, cv_report = load_artefacts()
    best_name   = cv_report["best_model_name"]
    best_params = cv_report["best_params"]

    # 2. Predict
    y_pred, y_pred_proba = get_predictions(model, X_test)

    # 3. Classification report
    metrics = print_classification_report(y_test, y_pred, y_pred_proba)

    # 4. Confusion matrix
    log.info("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, reports_dir)

    # 5. ROC curve
    log.info("Plotting ROC curve...")
    plot_roc_curve(y_test, y_pred_proba, reports_dir)

    # 6. Precision-Recall curve
    log.info("Plotting Precision-Recall curve...")
    plot_precision_recall_curve(y_test, y_pred_proba, reports_dir)

    # 7. Threshold analysis
    log.info("Running threshold analysis...")
    optimal_threshold = plot_threshold_analysis(y_test, y_pred_proba, reports_dir)

    # 8. SHAP
    log.info("Computing SHAP feature importance...")
    plot_shap_importance(model, X_test, feature_names, reports_dir)

    # 9. Save summary
    save_eval_summary(metrics, optimal_threshold, best_name, best_params, reports_dir, )

    log.info("=" * 55)
    log.info("  Evaluation complete.")
    log.info("  All plots saved to: %s", reports_dir)
    log.info("  Summary saved to:   %s", reports_dir / "eval_summary.json")
    log.info("=" * 55)


 
# Entry point
if __name__ == "__main__":
    evaluate()