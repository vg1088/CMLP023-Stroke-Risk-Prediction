#!/usr/bin/env python3
"""
Stroke Risk Prediction — main entry point.

Runs dataset check, preprocessing, model training, and evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path when run as script
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, RESULTS_DIR
from src.data_loader import resolve_dataset_path
from src.download_data import ensure_dataset
from src.evaluate import plot_model_comparison
from src.preprocessing import preprocess
from src.train import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stroke risk prediction ML pipeline (CMLP023)"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from Kaggle before running",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only preprocess data (no model training)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to data folder",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.download:
        print("Downloading dataset...")
        ensure_dataset()

    try:
        dataset_path = resolve_dataset_path(args.data_dir)
        print(f"Using dataset: {dataset_path}")
    except FileNotFoundError as exc:
        print(exc)
        print("Tip: run with --download or place stroke_dataset.csv in data/")
        return 1

    if args.skip_train:
        print("Preprocessing only...")
        preprocess(data_dir=args.data_dir)
        print(f"EDA and summaries saved to {RESULTS_DIR}")
        return 0

    print("Starting full pipeline: preprocess → train → evaluate")
    result, _processed = run_training_pipeline(data_dir=args.data_dir)
    plot_model_comparison()

    print("\nPipeline complete.")
    print(f"  Best model: {result.best_model_name}")
    print(f"  Model saved: {result.best_model_path}")
    print(f"  Results: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
