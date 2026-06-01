"""Train and compare stroke risk classification models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_NAMES, MODELS_DIR, RESULTS_DIR
from src.evaluate import evaluate_model
from src.models.stroke_models import tune_model
from src.preprocessing import ProcessedData, preprocess


@dataclass
class TrainingResult:
    best_model_name: str
    best_model_path: Path
    comparison: pd.DataFrame


def train_all(processed: ProcessedData) -> TrainingResult:
    """Train each algorithm, tune hyperparameters, and select the best by F1."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    best_name = ""
    best_f1 = -1.0
    best_path: Path | None = None
    tuning_log: dict[str, dict] = {}

    for name in MODEL_NAMES:
        print(f"Training {name}...")
        model, tune_info = tune_model(name, processed.X_train, processed.y_train)
        tuning_log[name] = tune_info

        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)

        metrics = evaluate_model(
            model,
            processed.X_test,
            processed.y_test,
            model_name=name,
            save_plots=True,
        )
        row = {"model": name, **metrics}
        rows.append(row)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_name = name
            best_path = model_path

    comparison = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    with open(RESULTS_DIR / "hyperparameter_tuning.json", "w", encoding="utf-8") as f:
        json.dump(tuning_log, f, indent=2)

    assert best_path is not None
    joblib.dump(joblib.load(best_path), MODELS_DIR / "best_model.joblib")

    with open(RESULTS_DIR / "best_model.txt", "w", encoding="utf-8") as f:
        f.write(f"{best_name}\n")

    print(f"Best model: {best_name} (F1={best_f1:.4f})")
    return TrainingResult(
        best_model_name=best_name,
        best_model_path=MODELS_DIR / "best_model.joblib",
        comparison=comparison,
    )


def run_training_pipeline(data_dir: Path | None = None) -> tuple[TrainingResult, ProcessedData]:
    """Preprocess data then train all models."""
    processed = preprocess(data_dir=data_dir)
    result = train_all(processed)
    return result, processed


if __name__ == "__main__":
    training_result, _ = run_training_pipeline()
    print(training_result.comparison.to_string(index=False))
