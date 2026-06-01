"""Model evaluation metrics and visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import MODELS_DIR, RESULTS_DIR


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float]:
    """Calculate classification metrics."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _predict_proba_positive(model: Any, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return None


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=120)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_curve_{model_name}.png", dpi=120)
    plt.close()


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    save_plots: bool = True,
) -> dict[str, float]:
    """Evaluate a trained model and optionally save plots and reports."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    y_proba = _predict_proba_positive(model, X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_path = RESULTS_DIR / f"classification_report_{model_name}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if save_plots:
        plot_confusion_matrix(y_test, y_pred, model_name, RESULTS_DIR)
        if y_proba is not None and not np.isnan(metrics.get("roc_auc", np.nan)):
            plot_roc_curve(y_test, y_proba, model_name, RESULTS_DIR)

    metrics_path = RESULTS_DIR / f"metrics_{model_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def evaluate_best_model(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_path: Path | None = None,
) -> dict[str, float]:
    """Load best saved model and run full evaluation."""
    path = model_path or (MODELS_DIR / "best_model.joblib")
    model = joblib.load(path)
    name = "best_model"
    best_txt = RESULTS_DIR / "best_model.txt"
    if best_txt.exists():
        name = best_txt.read_text(encoding="utf-8").strip() or name
    return evaluate_model(model, X_test, y_test, model_name=name, save_plots=True)


def plot_model_comparison(comparison_csv: Path | None = None) -> None:
    """Bar chart comparing models on key metrics."""
    path = comparison_csv or (RESULTS_DIR / "model_comparison.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    metrics_cols = [c for c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"] if c in df.columns]
    if not metrics_cols:
        return

    melted = df.melt(id_vars=["model"], value_vars=metrics_cols, var_name="metric", value_name="score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.xticks(rotation=15)
    plt.title("Model performance comparison")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison_chart.png", dpi=120)
    plt.close()
