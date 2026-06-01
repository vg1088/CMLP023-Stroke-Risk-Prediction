# Data Folder

Place the stroke prediction dataset here after downloading from Kaggle.

## Download

**Option A – automatic (recommended)**

```bash
python -m src.download_data
```

Requires [Kaggle API credentials](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json`.

**Option B – manual**

1. Open https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset  
2. Download the dataset  
3. Extract `healthcare-dataset-stroke-data.csv` into this folder as `stroke_dataset.csv`

## Expected file

| File | Description |
|------|-------------|
| `stroke_dataset.csv` | Main dataset used by the pipeline |

The loader also accepts `healthcare-dataset-stroke-data.csv` if present.
