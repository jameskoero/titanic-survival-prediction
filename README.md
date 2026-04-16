# Titanic Survival Prediction

A production-style machine learning project that predicts Titanic passenger survival using a modular Python workflow based on logistic regression.

## Project Overview

This repository provides:
- Cleaned and structured project layout
- Reproducible preprocessing, training, and evaluation scripts
- Notebook-based EDA and modelling walkthroughs
- Evaluation artifacts (confusion matrix and ROC curve)
- A report documenting findings

## Repository Structure

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
│   └── titanic-survival-prediction-report.pdf
└── .gitignore
```

## Dataset

- Raw data is stored in `data/raw/train.csv`.
- Source: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)

## Installation

```bash
pip install -r requirements.txt
```

## Reproducible Pipeline

Run from the repository root:

```bash
python src/preprocess.py --input data/raw/train.csv --output data/processed/train_processed.csv
python src/train.py --input data/processed/train_processed.csv
python src/evaluate.py --input data/processed/train_processed.csv
```

## Outputs

Evaluation files are generated in `outputs/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `train_metrics.json`
- `evaluation_metrics.json`

## Model

- Algorithm: Logistic Regression (`scikit-learn`)
- Feature set: `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `FamilySize`, `IsAlone`, `Title`
- Typical benchmark (depends on split): ~0.80 accuracy and ~0.85+ ROC-AUC

## Author

James Koero
