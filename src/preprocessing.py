"""Data cleaning, encoding, scaling, and train/test split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ID_COLUMNS,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    USE_SMOTE,
)
from src.data_loader import load_raw_data


@dataclass
class ProcessedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple derived features for stroke risk modeling."""
    out = df.copy()
    if "age" in out.columns and "avg_glucose_level" in out.columns:
        out["age_glucose_interaction"] = out["age"] * out["avg_glucose_level"] / 100.0
    if "bmi" in out.columns:
        out["bmi"] = pd.to_numeric(out["bmi"], errors="coerce")
        out["bmi_category"] = pd.cut(
            out["bmi"],
            bins=[0, 18.5, 25, 30, np.inf],
            labels=["underweight", "normal", "overweight", "obese"],
        )
    return out


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and invalid records."""
    data = df.copy()
    drop_cols = [c for c in ID_COLUMNS if c in data.columns]
    data = data.drop(columns=drop_cols, errors="ignore")

    if "gender" in data.columns:
        data = data[data["gender"] != "Other"]

    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    return data


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build sklearn preprocessor for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )



def run_eda(df: pd.DataFrame, results_dir: Path | None = None) -> None:
    """Save exploratory analysis plots to results/."""
    out = results_dir or RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN].value_counts().plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
        axes[0].set_title("Stroke class distribution")
        axes[0].set_xlabel(TARGET_COLUMN)
    if "age" in df.columns and TARGET_COLUMN in df.columns:
        sns.boxplot(data=df, x=TARGET_COLUMN, y="age", ax=axes[1])
        axes[1].set_title("Age by stroke outcome")
    plt.tight_layout()
    plt.savefig(out / "eda_class_and_age.png", dpi=120, bbox_inches="tight")
    plt.close()

    numeric = df.select_dtypes(include=[np.number])
    if TARGET_COLUMN in numeric.columns and len(numeric.columns) > 1:
        corr = numeric.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Feature correlation matrix")
        plt.tight_layout()
        plt.savefig(out / "eda_correlation_heatmap.png", dpi=120, bbox_inches="tight")
        plt.close()


def preprocess(
    data_dir: Path | None = None,
    save_artifacts: bool = True,
) -> ProcessedData:
    """Full preprocessing pipeline: clean, EDA, transform, split, optional SMOTE."""
    raw = load_raw_data(data_dir)
    cleaned = _clean_data(raw)
    engineered = _engineer_features(cleaned)

    run_eda(engineered)

    y = engineered[TARGET_COLUMN].astype(int)
    X = engineered.drop(columns=[TARGET_COLUMN])

    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_processed.shape[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if USE_SMOTE:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(
            {
                "metric": ["rows_raw", "rows_clean", "train_samples", "test_samples", "features"],
                "value": [len(raw), len(cleaned), len(y_train), len(y_test), X_processed.shape[1]],
            }
        )
        summary.to_csv(RESULTS_DIR / "preprocessing_summary.csv", index=False)

    return ProcessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        preprocessor=preprocessor,
    )
