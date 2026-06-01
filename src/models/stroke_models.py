"""Classification model factories and hyperparameter grids."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import CV_FOLDS, MODEL_NAMES, N_JOBS, RANDOM_STATE


def build_model(name: str) -> Any:
    """Return an untrained estimator for the given model key."""
    estimators = get_all_estimators()
    if name not in estimators:
        raise ValueError(f"Unknown model: {name}. Choose from {list(estimators)}")
    return estimators[name]


def get_all_estimators() -> dict[str, Any]:
    """Registry of classification algorithms for stroke prediction."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=20,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "svm": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def get_param_grids() -> dict[str, dict[str, list[Any]]]:
    """Hyperparameter search spaces per model."""
    return {
        "logistic_regression": {
            "C": [0.01, 0.1, 1.0],
            "solver": ["lbfgs"],
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [8, 12, None],
            "min_samples_leaf": [1, 3],
        },
        "xgboost": {
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [100, 200],
        },
        "svm": {
            "C": [0.1, 1.0],
            "gamma": ["scale", "auto"],
        },
    }


def tune_model(name: str, X_train, y_train) -> tuple[Any, dict[str, Any]]:
    """Run grid search and return best estimator with search results."""
    base = build_model(name)
    grid = get_param_grids().get(name, {})
    if not grid:
        base.fit(X_train, y_train)
        return base, {}

    search = GridSearchCV(
        base,
        grid,
        cv=CV_FOLDS,
        scoring="f1",
        n_jobs=N_JOBS,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
    }
