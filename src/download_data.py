"""Download stroke prediction dataset from Kaggle into data/."""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from src.config import DATA_DIR, PROJECT_ROOT

KAGGLE_DATASET = "fedesoriano/stroke-prediction-dataset"
SOURCE_CSV = "healthcare-dataset-stroke-data.csv"
TARGET_CSV = "stroke_dataset.csv"

# Public mirror of the Kaggle CSV (same schema); prefer Kaggle API when available.
PUBLIC_CSV_URL = (
    "https://raw.githubusercontent.com/chandanverma07/DataSets/master/"
    "healthcare-dataset-stroke-data.csv"
)


def _find_csv(directory: Path) -> Path | None:
    for name in (TARGET_CSV, SOURCE_CSV):
        path = directory / name
        if path.exists():
            return path
    for path in directory.glob("*.csv"):
        return path
    return None


def download_with_http() -> Path:
    """Download dataset from a public mirror (fallback)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / TARGET_CSV
    print(f"Downloading from public mirror to {target}...")
    urllib.request.urlretrieve(PUBLIC_CSV_URL, target)
    return target


def download_with_kaggle_api() -> Path:
    """Download dataset using the Kaggle CLI."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(DATA_DIR),
            "--unzip",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    csv_path = _find_csv(DATA_DIR)
    if csv_path is None:
        raise FileNotFoundError(f"No CSV found in {DATA_DIR} after download.")
    target = DATA_DIR / TARGET_CSV
    if csv_path != target:
        shutil.copy2(csv_path, target)
    return target


def ensure_dataset() -> Path:
    """Return path to dataset, downloading if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _find_csv(DATA_DIR)
    if existing is not None:
        target = DATA_DIR / TARGET_CSV
        if existing != target and not target.exists():
            shutil.copy2(existing, target)
        return target if target.exists() else existing

    zip_path = DATA_DIR / "stroke-prediction-dataset.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        csv_path = _find_csv(DATA_DIR)
        if csv_path:
            target = DATA_DIR / TARGET_CSV
            if csv_path != target:
                shutil.copy2(csv_path, target)
            return target

    try:
        return download_with_kaggle_api()
    except Exception as api_error:
        print(f"Kaggle API unavailable ({api_error}). Trying HTTP mirror...")
        try:
            return download_with_http()
        except Exception as http_error:
            raise RuntimeError(
                "Dataset not found. Configure Kaggle API credentials, run manual "
                f"download, or check network access. HTTP error: {http_error}"
            ) from http_error


def main() -> None:
    path = ensure_dataset()
    print(f"Dataset ready: {path}")


if __name__ == "__main__":
    main()
