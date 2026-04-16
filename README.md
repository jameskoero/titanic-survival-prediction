# Titanic Survival Prediction (Production-Ready ML Pipeline)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Production%20Structure-brightgreen)

## Project Overview
This project predicts whether a passenger survived the Titanic disaster using supervised machine learning.
It is structured as a clean, modular, recruiter-ready repository with CLI scripts, notebooks, automated artifact generation, and professional documentation.

## Problem Statement
Build a robust binary classification system that predicts `Survived` (`1`) vs `Not Survived` (`0`) from passenger attributes such as class, sex, age, fare, embarkation point, and engineered family/title features.

## Dataset Source
- **Kaggle Titanic Competition Dataset**
- URL: https://www.kaggle.com/c/titanic/data
- Required file for this repo: `data/raw/train.csv`

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

```text
titanic-survival-prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── EDA.ipynb
│   └── modelling.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── outputs/
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── report/
│   └── titanic_report.pdf
└── .gitignore
```

## Setup Instructions

### 1) Clone and enter repository
```bash
git clone https://github.com/jameskoero/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2) Create virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Download Titanic dataset
Place Kaggle `train.csv` in:
```text
data/raw/train.csv
```

Optional via Kaggle CLI:
```bash
kaggle competitions download -c titanic -p data/raw
unzip data/raw/titanic.zip -d data/raw
```

## Run End-to-End Pipeline

### Preprocess data
```bash
python src/preprocess.py
```

### Train model(s)
```bash
python src/train.py --compare-models --cross-validate
```

### Evaluate and generate artifacts
```bash
python src/evaluate.py
```

After running evaluation, these artifacts are generated:
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `report/titanic_report.pdf`

## Model Performance Metrics
Current pipeline tracks:
- Accuracy
- ROC-AUC
- Classification Report (Precision / Recall / F1)

Metrics are saved to:
- `outputs/metrics.json`

> Exact values depend on the current train/test split and model selected.

## Visual Outputs
- **Confusion Matrix:** `outputs/confusion_matrix.png`
- **ROC Curve:** `outputs/roc_curve.png`

## Notebooks
- `notebooks/EDA.ipynb`: missing values, distributions, heatmap, survival by sex/class/age
- `notebooks/modelling.ipynb`: feature engineering, split, model training, evaluation

## Future Improvements
- Hyperparameter tuning with Optuna/GridSearchCV
- Experiment tracking (MLflow / Weights & Biases)
- Model calibration and threshold optimization
- FastAPI inference endpoint + Docker deployment
- CI pipeline with model quality gates

## Author
James Koero
