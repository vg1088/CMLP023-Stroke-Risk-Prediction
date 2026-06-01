"""Project paths and training configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_NAMES = (
    "stroke_dataset.csv",
    "healthcare-dataset-stroke-data.csv",
)

TARGET_COLUMN = "stroke"
ID_COLUMNS = ("id",)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Resampling for class imbalance
USE_SMOTE = True

# Model registry keys
MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "svm",
)

# Hyperparameter search (lightweight defaults for reproducibility)
CV_FOLDS = 3
N_JOBS = -1
