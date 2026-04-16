# Titanic Survival Prediction 🚢

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

## Project Overview
A modular machine learning project that predicts passenger survival on the Titanic using the Kaggle Titanic dataset. The repository is structured for end-to-end reproducibility with preprocessing, model training, evaluation artifacts, and a concise PDF report.

## Problem Statement
Build a reliable binary classification system to predict whether a passenger survived (`1`) or did not survive (`0`) based on demographic and ticket-related features.

## Dataset Source
- **Kaggle Titanic: Machine Learning from Disaster**
- URL: https://www.kaggle.com/c/titanic/data
- Expected file: `train.csv` in `data/raw/`

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
│   │   └── train.csv
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
1. **Clone repository**
   ```bash
   git clone https://github.com/jameskoero/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   # Windows (PowerShell): venv\Scripts\Activate.ps1
   # Windows (CMD): venv\Scripts\activate.bat
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download dataset**
   - Download `train.csv` from Kaggle and place it in `data/raw/`.

## Run the Pipeline
```bash
python src/train.py
python src/evaluate.py
```

## Model Performance Metrics
The training pipeline compares:
- Logistic Regression (required baseline)
- Random Forest (bonus model)

Saved metrics include:
- Accuracy
- ROC-AUC
- Cross-validation ROC-AUC (mean/std)

Example baseline target:
- Accuracy: ~0.80+
- ROC-AUC: ~0.85+

## Visual Outputs
After evaluation, these files are generated automatically:
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `report/titanic_report.pdf`

## Future Improvements
- Hyperparameter tuning with Optuna/GridSearchCV
- Advanced feature engineering from ticket and cabin fields
- Model explainability (SHAP)
- FastAPI inference service and Docker packaging
- CI pipeline with automated tests and model checks
