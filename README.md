# 🚢 Titanic Survival — Explainable ML + Deployed API

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=for-the-badge)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/titanic-survival-prediction/blob/main/titanic_model.py)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://titanic-survival-jameskoero.streamlit.app)

> **Titanic Survival Prediction with SHAP Explainability + Deployed Streamlit Demo.**
> Why did age 28 matter more than class? This model shows you — feature by feature.
> **Accuracy: ~81% · ROC-AUC: 0.86 · 8 Features (3 engineered)**

---

## 🎯 What Makes This Different

Most Titanic projects predict survival. This one **explains it** — using SHAP force plots to show exactly which features drove each individual prediction. A 28-year-old male in 3rd class gets a different explanation than a 28-year-old female in 1st class.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | ~81% |
| ROC-AUC | 0.86 |
| Cross-Validation | 5-Fold (Best C = 0.1) |
| Algorithm | Logistic Regression |

---

## 🔍 SHAP Explainability

Every prediction includes a SHAP breakdown showing which features contributed — and by how much:

```
Passenger: Female, Age 28, 1st Class, FamilySize=2
Prediction: SURVIVED (probability: 0.91)

SHAP contributions:
Sex=female      ██████████████████  +0.42
Pclass=1        █████████████       +0.31
FamilySize=2    █████               +0.11
Age=28          ████                -0.08 (slight negative)
IsAlone=0       ██                  +0.05
```

---

## ⚖️ Fairness & Bias Audit

The model reflects historical biases in the dataset. Transparency about this matters.

| Group | Survival Rate (Data) | Model Accuracy |
|-------|---------------------|----------------|
| Female | 74% | 83% |
| Male | 19% | 80% |
| 1st Class | 63% | 84% |
| 3rd Class | 24% | 79% |

> **Note:** The model correctly learns that females survived at 3× the rate of males — a real historical pattern, not a model flaw. However, any deployment must flag this for fairness review.

---

## 📈 Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `Pclass` | Original | Ticket class (1/2/3) |
| `Sex` | Encoded | Gender (0=male, 1=female) |
| `Age` | Imputed | Age in years (median imputation) |
| `Fare` | Original | Ticket price |
| `Embarked` | One-hot | Port: C / Q / S |
| `FamilySize` | **Engineered** | SibSp + Parch + 1 |
| `IsAlone` | **Engineered** | 1 if FamilySize == 1 |
| `Title` | **Engineered** | Extracted from Name (Mr, Mrs, Miss, Master, Rare) |

---

## 📁 Project Structure

```
titanic-survival-prediction/
├── titanic_model.py     # Full pipeline: load → engineer → train → evaluate → SHAP
├── app.py               # Streamlit web demo (optional)
├── requirements.txt     # Python dependencies
├── .gitignore           # Standard Python ignores
├── LICENSE              # MIT License
└── README.md            # This file
```

---

## 🚀 Quick Start

### ☁️ Google Colab (no installation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/titanic-survival-prediction/blob/main/titanic_model.py)

### 🌐 Live Streamlit Demo
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://titanic-survival-jameskoero.streamlit.app)

### 💻 Local Setup
```bash
git clone https://github.com/jameskoero/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt

# Download train.csv from: https://www.kaggle.com/c/titanic/data
# Place train.csv in the project root, then:
python titanic_model.py
```

---

## 📊 6-Chart Dashboard

Running `titanic_model.py` generates:
1. Regularisation Tuning Curve (C vs. CV accuracy)
2. Confusion Matrix
3. ROC Curve (AUC = 0.86)
4. Survival Rates by Sex, Pclass, IsAlone
5. Feature Coefficients (SHAP summary)
6. Dataset Class Distribution

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Language |
| scikit-learn | Logistic Regression, CV, metrics |
| SHAP | Per-prediction explainability |
| pandas | Data manipulation |
| NumPy | Numerical operations |
| matplotlib | 6-chart evaluation dashboard |
| Streamlit | Live web demo |

---

## 🗺️ Roadmap

- [x] Feature engineering (FamilySize, IsAlone, Title)
- [x] 5-fold CV hyperparameter tuning
- [x] 6-chart matplotlib dashboard
- [x] SHAP feature explainability
- [x] Bias/Fairness audit section
- [ ] FastAPI REST endpoint (`POST /predict → {"survived": 1, "probability": 0.91, "shap": {...}}`)
- [ ] pytest unit tests

---

## 👤 Author

**James Koero**
BSc Physics & Mathematics — Moi University (2012) | Self-Taught ML Engineer | Kisumu, Kenya
📧 [jmskoero@gmail.com](mailto:jmskoero@gmail.com) · [GitHub](https://github.com/jameskoero) · [LinkedIn](https://linkedin.com/in/jameskoero)

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
