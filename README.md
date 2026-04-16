# 🚢 Titanic Survival Prediction

![Titanic Survival Prediction Banner](https://raw.githubusercontent.com/jameskoero/titanic-survival-prediction/main/banner.png)

## 📋 Problem Statement

The sinking of the Titanic is one of history's most infamous maritime disasters. This project aims to build a machine learning model that predicts whether a passenger survived or not based on features like age, gender, ticket class, fare, and embarkation point. This classification problem serves as an excellent demonstration of real-world predictive modeling.

---

## 🏆 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **81%** |
| **ROC-AUC** | **0.86** |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-green?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-lightblue?style=flat-square&logo=numpy)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red?style=flat-square&logo=streamlit)

---

## 🚀 How to Run

```bash
# 1. Clone repository
git clone https://github.com/jameskoero/titanic-survival-prediction.git && cd titanic-survival-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the model
python main.py
```

---

## 💡 Key Findings & Insights

- **Gender is the strongest predictor**: Female passengers had significantly higher survival rates (~74%) compared to males (~19%)
- **Passenger class matters**: First-class passengers had ~63% survival rate vs. third-class at ~24%
- **Age dependency**: Children and younger passengers had better survival odds
- **Family size impact**: Traveling alone reduced survival chances; small family groups had better outcomes
- **Logistic Regression effectiveness**: Despite its simplicity, the model achieved strong performance (81% accuracy, 0.86 ROC-AUC) showing the importance of data preprocessing and feature engineering

---

## 📊 Model Performance Visualizations

### Confusion Matrix
![Confusion Matrix](https://raw.githubusercontent.com/jameskoero/titanic-survival-prediction/main/confusion_matrix.png)

### ROC Curve
![ROC Curve](https://raw.githubusercontent.com/jameskoero/titanic-survival-prediction/main/roc_curve.png)

### Feature Distribution
![Feature Distribution](https://raw.githubusercontent.com/jameskoero/titanic-survival-prediction/main/feature_distribution.png)

---

## 📌 Algorithm & Features

**Model**: Logistic Regression (Binary Classification)

**Features Used**:
- Pclass (Passenger Class)
- Sex (Gender)
- Age
- Fare
- Embarked (Port)
- FamilySize
- IsAlone
- Title (from Name)

---

## 📚 Dataset

Training data available from: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)

---

## 👤 About

**James Koero** | BSc Physics and Mathematics | Self-taught ML Engineer | Kisumu, Kenya  
📧 [jmskoero@gmail.com](mailto:jmskoero@gmail.com)

---

*Last Updated: April 2026*