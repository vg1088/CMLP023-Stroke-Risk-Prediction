# CMLP023 – Stroke Risk Prediction

## Repository Name
`stroke-risk-prediction`

## Dataset
Kaggle Dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## Project Objective

Develop a Machine Learning model to predict stroke risk using patient demographic and healthcare data to support early diagnosis, preventive care, and clinical decision-making.

---

## Milestone 1: Project Setup and Dataset Preparation

### Tasks
- Create GitHub repository structure
- Create a `data/` folder for dataset storage
- Download stroke prediction dataset from Kaggle
- Place dataset files inside the `data/` folder
- Organize project folders and files
- Install required Python libraries
- Configure development environment
- Explore dataset structure and healthcare attributes
- Define project workflow and objectives

### Deliverable
Project environment configured and dataset prepared inside the `data/` folder.

---

## Milestone 2: Data Preprocessing and Exploratory Analysis

### Tasks
- Load dataset from the `data/` folder into Python
- Handle missing and duplicate records
- Encode categorical variables
- Normalize numerical feature values
- Perform feature engineering
- Conduct exploratory data analysis (EDA)
- Visualize patient health patterns and risk factors
- Analyze feature correlations with stroke occurrence
- Split dataset into training and testing datasets

### Deliverable
Processed dataset ready for model training.

---

## Milestone 3: Model Development and Training

### Tasks
- Select classification algorithm (Random Forest, XGBoost, Logistic Regression, or SVM)
- Configure model parameters
- Train stroke risk prediction model
- Handle class imbalance using resampling techniques
- Monitor training performance
- Perform hyperparameter tuning
- Save trained model checkpoints
- Compare model performance iterations
- Improve prediction accuracy

### Deliverable
Trained stroke risk prediction model.

---

## Milestone 4: Model Evaluation and Results Analysis

### Tasks
- Evaluate model performance
- Calculate accuracy, precision, recall, and F1-score
- Generate confusion matrix
- Measure ROC-AUC score
- Analyze prediction outputs
- Visualize classification performance
- Identify model limitations
- Document findings and observations
- Update README documentation

### Deliverable
Final stroke risk prediction system with performance evaluation.

---

## Expected Outcome

Develop an AI-powered stroke risk prediction system capable of identifying high-risk individuals and supporting proactive healthcare interventions.

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Project Structure

```text
stroke-risk-prediction/
│
├── data/
│   └── stroke_dataset.csv
│
├── notebooks/
├── models/
├── results/
├── src/
├── requirements.txt
├── README.md
└── main.py
```

---

## Future Improvements

- Improve prediction accuracy
- Integrate additional clinical health indicators
- Add explainable AI techniques for healthcare insights
- Optimize model performance
- Deploy as a healthcare risk assessment application
