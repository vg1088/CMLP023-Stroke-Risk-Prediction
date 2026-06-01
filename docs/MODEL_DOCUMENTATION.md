# Stroke Risk Prediction — Model Documentation

## Problem definition

**Task:** Binary classification — predict whether a patient experienced a stroke (`stroke` = 1) from demographic and clinical features.

**Business goal:** Support early identification of high-risk individuals for preventive care and clinical decision support.

## Dataset

| Attribute | Description |
|-----------|-------------|
| Source | [Kaggle — Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
| Records | ~5,110 patients |
| Target | `stroke` (0 = no, 1 = yes) |
| Challenge | Severe class imbalance (~5% positive class) |

## Preprocessing pipeline

1. Remove identifier column (`id`) and rare `gender == Other` rows  
2. Drop duplicate records  
3. Feature engineering: BMI categories, age–glucose interaction  
4. Impute missing BMI (median) and encode categoricals (one-hot)  
5. Scale numeric features (standardization)  
6. Stratified 80/20 train/test split  
7. **SMOTE** on training data to mitigate imbalance  

Artifacts: `models/preprocessor.joblib`, `results/preprocessing_summary.csv`, EDA plots in `results/`.

## Models evaluated

| Model | Rationale |
|-------|-----------|
| Logistic Regression | Interpretable baseline for clinical settings |
| Random Forest | Non-linear interactions, robust to outliers |
| XGBoost | Strong tabular performance with imbalance handling |
| SVM (RBF) | Alternative kernel-based boundary |

Hyperparameter tuning uses 3-fold cross-validation with **F1-score** as the selection metric (appropriate for imbalanced classification).

## Evaluation metrics

- Accuracy, precision, recall, F1-score  
- ROC-AUC  
- Confusion matrix and ROC curve plots per model  
- Classification report (JSON) per model  

Best model is saved as `models/best_model.joblib`; selection criterion is highest **F1-score** on the held-out test set.

## Known limitations

- Class imbalance remains difficult; high recall often trades off against precision.  
- BMI has many missing values in the raw data.  
- Model is trained on a single public dataset and is not validated for deployment in clinical workflows without further review.  
- No SHAP/LIME explainability in this version (see Future Improvements in README).

## Reproducibility

Random seed: `42` (see `src/config.py`). Re-run the full pipeline with `python main.py`.
