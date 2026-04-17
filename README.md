# 🚢 Titanic Survival Prediction
### *Can a machine learn who survives a disaster? — Yes, with 81% accuracy.*

![Titanic Survival Prediction Banner](https://raw.githubusercontent.com/jameskoero/titanic-survival-prediction/main/banner.png)


![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-27AE60?style=flat)
![License](https://img.shields.io/badge/License-MIT-0D2137?style=flat)

---

## 📋 Problem Statement

The sinking of the Titanic is one of history's most infamous maritime disasters. This project aims to build a machine learning model that predicts whether a passenger survived or not based on features like age, gender, passenger class, and fare.

---

## 📌 Project Overview

On **April 15, 1912**, the RMS Titanic sank after colliding with an iceberg, killing over **1,500 people**.  
This project builds a **binary classification machine learning model** to predict which passengers survived — using age, gender, ticket class, and other features.

> *"The machine doesn't judge. It simply learns from the data — and in doing so, exposes the harsh realities of 1912 society."*  
> — James Koero, ML Engineer

**Business Use Case:** This type of binary classification pipeline is directly applicable to:
- 🏦 Credit risk scoring (will a customer default?)
- 📉 Customer churn prediction (will a customer leave?)
- 🏥 Medical outcome prediction (will a patient recover?)

---

## 🎯 Results at a Glance

| Metric | Score |
|---|---|
| ✅ **Accuracy** | **81%** |
| ✅ **ROC-AUC Score** | **0.86** |
| ✅ **Precision** | 79% |
| ✅ **Recall** | 75% |
| ✅ **Algorithm** | Logistic Regression (scikit-learn) |
| ✅ **Dataset** | Kaggle Titanic — 891 passengers |

---

## 📊 Visual Results

### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)

> The model correctly classified **81% of passengers**. True Negatives (correctly predicted non-survivors) and True Positives (correctly predicted survivors) dominate the matrix.

---

### ROC Curve
![ROC Curve](outputs/roc_curve.png)

> An **AUC of 0.86** means the model is excellent at distinguishing survivors from non-survivors — far above the 0.5 baseline of a random classifier. The optimal threshold is marked in gold.

---

### Feature Distribution & Importance
![Feature Distribution](outputs/feature_distribution.png)

> **Key insights revealed by the data:**
> - 🟢 **Sex** was the strongest predictor — female passengers survived at a dramatically higher rate ("Women and children first")
> - 🟢 **1st Class** passengers had a massive survival advantage over 3rd Class
> - 🔴 **Being male** in 3rd class was the highest-risk combination
> - Age, Fare, and FamilySize contributed meaningful signal to the model

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.9+` | Core language |
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Logistic Regression, preprocessing, evaluation |
| `matplotlib` | Visualisation (confusion matrix, ROC, charts) |

---

## 🗂️ Project Structure

```
titanic-survival-prediction/
│
├── titanic_model.py              ← Main ML pipeline (train → evaluate → visualise)
├── requirements.txt              ← All dependencies
├── README.md                     ← This file
│
├── outputs/
│   ├── confusion_matrix.png      ← Confusion matrix visualisation
│   ├── roc_curve.png             ← ROC curve with AUC score
│   └── feature_distribution.png ← Feature analysis by survival status
│
├── Titanic Survival prediction final.pdf  ← Full project report (PDF)
└── train.csv                             ← Download from Kaggle (see instructions below)
```

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/jameskoero/titanic-survival-prediction.git
cd titanic-survival-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**
- Go to: https://www.kaggle.com/c/titanic/data
- Download `train.csv`
- Place it in the project root folder

**4. Run the model**
```bash
python titanic_model.py
```

The script will:
- Load and preprocess the data
- Engineer features (Title, FamilySize, IsAlone)
- Train a Logistic Regression classifier
- Print accuracy, ROC-AUC, and classification report
- Save `confusion_matrix.png`, `roc_curve.png`, `feature_distribution.png`, and `feature_importance.png`

---

## 🧠 ML Pipeline — Step by Step

```
Raw Data (train.csv)
      │
      ▼
1. Exploratory Data Analysis (EDA)
      │  → missing values, survival rates, distributions
      ▼
2. Feature Engineering
      │  → Title extraction (Mr, Miss, Mrs, Master, Rare)
      │  → FamilySize = SibSp + Parch + 1
      │  → IsAlone flag
      ▼
3. Data Preprocessing
      │  → Median imputation for Age & Fare
      │  → Mode imputation for Embarked
      │  → Label encoding for Sex, Embarked, Title
      ▼
4. Train/Test Split (80% / 20%, stratified)
      │
      ▼
5. Feature Scaling (StandardScaler)
      │
      ▼
6. Model Training (Logistic Regression, max_iter=1000)
      │
      ▼
7. Evaluation
      │  → Accuracy, ROC-AUC, Precision, Recall
      │  → Confusion Matrix, Classification Report
      ▼
8. Visualisation & Reporting
      └  → PNG charts + PDF report
```

---

## 📋 Features Used

| Feature | Type | Description |
|---|---|---|
| `Pclass` | Ordinal | Passenger class (1=First, 2=Second, 3=Third) |
| `Sex` | Binary | Gender (encoded: 0=male, 1=female) |
| `Age` | Continuous | Age in years (median-imputed) |
| `Fare` | Continuous | Ticket fare paid |
| `Embarked` | Categorical | Port of embarkation (S/C/Q) |
| `FamilySize` | Engineered | SibSp + Parch + 1 |
| `IsAlone` | Engineered | 1 if travelling alone, 0 otherwise |
| `Title` | Engineered | Extracted from Name (Mr, Miss, Mrs, etc.) |

---

## 💡 Key Learnings

1. **Data cleaning consumed ~80% of the effort** — missing Age, Embarked, and Fare values required careful imputation strategy
2. **Simple models can deliver powerful insights** — Logistic Regression achieved 81% accuracy with proper feature engineering
3. **Feature engineering drives performance** — engineered features (Title, FamilySize, IsAlone) were more predictive than raw inputs alone
4. **Asking "why" behind results matters** — the model confirms documented historical accounts of evacuation priority
5. **Class imbalance awareness** — stratified splitting ensured fair train/test representation

---

## 📄 Full Report

A detailed PDF report is available in this repository:  
📎 [`Titanic Survival prediction final.pdf`](./Titanic%20Survival%20prediction%20final.pdf)

Includes full EDA, methodology, results analysis, and visualisations.

---

## 🔗 Connect

**James Onyango Koero** — Junior ML Engineer | Kisumu, Kenya

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/jameskoero)
[![GitHub](https://img.shields.io/badge/GitHub-jameskoero-181717?style=flat&logo=github)](https://github.com/jameskoero)
[![Email](https://img.shields.io/badge/Email-jmskoero@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:jmskoero@gmail.com)

> *Self-taught ML Engineer | B.Sc. Physics & Mathematics | Open to remote opportunities worldwide*

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

*⭐ If this project helped you, please give it a star — it keeps me motivated to build more!*
