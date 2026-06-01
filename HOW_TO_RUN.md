# How to Run the Stroke Risk Prediction Project

This guide explains how to set up the environment, obtain the dataset, and execute the machine learning pipeline.

## Prerequisites

- Python 3.10 or newer
- pip
- (Optional) [Kaggle API](https://www.kaggle.com/docs/api) credentials for automatic dataset download

## 1. Clone and enter the project

```bash
git clone <repository-url>
cd stroke-risk-prediction
```

If you are already in the repository root, skip the clone step.

## 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Get the dataset

### Option A — Automatic download (Kaggle API)

1. Create a Kaggle account and API token: https://www.kaggle.com/settings → **Create New Token**
2. Place `kaggle.json` in `~/.kaggle/` and set permissions:

   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Download into the `data/` folder:

   ```bash
   python -m src.download_data
   ```

### Option B — Manual download

1. Open https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset  
2. Download and extract the CSV  
3. Save it as `data/stroke_dataset.csv`

## 5. Run the full pipeline

From the project root:

```bash
python main.py
```

This will:

1. Load data from `data/`
2. Preprocess, encode features, and apply SMOTE for class imbalance
3. Save EDA plots to `results/`
4. Train Logistic Regression, Random Forest, XGBoost, and SVM
5. Tune hyperparameters and pick the best model by F1-score
6. Save models to `models/` and metrics/plots to `results/`

### Other useful commands

| Command | Description |
|---------|-------------|
| `python main.py --download` | Download from Kaggle, then run the full pipeline |
| `python main.py --skip-train` | Preprocess and EDA only (no training) |
| `python -m src.download_data` | Download dataset only |
| `python -m src.train` | Train models (via preprocessing + training module) |

## 6. Review outputs

| Location | Contents |
|----------|----------|
| `data/` | Raw dataset (`stroke_dataset.csv`) |
| `models/` | `preprocessor.joblib`, per-model checkpoints, `best_model.joblib` |
| `results/` | Metrics JSON, comparison CSV, confusion matrices, ROC curves, EDA plots |

Key result files:

- `results/model_comparison.csv` — metrics for all models  
- `results/best_model.txt` — name of the selected model  
- `results/confusion_matrix_*.png` — confusion matrices  
- `results/roc_curve_*.png` — ROC curves  

## 7. Troubleshooting

| Issue | Solution |
|-------|----------|
| `No dataset found in data/` | Run `python -m src.download_data` or add `stroke_dataset.csv` manually |
| Kaggle 401 / 403 | Check `~/.kaggle/kaggle.json` and API token validity |
| Import errors | Activate venv and run `pip install -r requirements.txt` |
| Low recall on stroke class | Expected with severe imbalance; SMOTE is enabled by default in `src/config.py` |

## Project layout

```text
├── data/              # Dataset storage
├── models/            # Trained model checkpoints
├── results/           # Plots, metrics, reports
├── notebooks/         # Optional Jupyter notebooks
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── download_data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── models/
│       └── stroke_models.py
├── main.py            # Main entry point
├── requirements.txt
├── README.md
└── HOW_TO_RUN.md      # This file
```
