"""Load stroke dataset from the data folder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, DATASET_NAMES


def resolve_dataset_path(data_dir: Path | None = None) -> Path:
    """Find the first available dataset file in data/."""
    base = data_dir or DATA_DIR
    for name in DATASET_NAMES:
        path = base / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No dataset found in {base}. Expected one of: {', '.join(DATASET_NAMES)}. "
        "Run: python -m src.download_data"
    )


def load_raw_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    path = resolve_dataset_path(data_dir)
    df = pd.read_csv(path)
    return df
