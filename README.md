# Titanic Survival Prediction
## Overview
A machine learning project that predicts whether a Titanic passenger
survived or not using Logistic Regression.
## Algorithm
- Logistic Regression (scikit-learn)
- Binary Classification: Survived (1) or Not (0)
## Results
| Metric | Score |
|-----------|--------|
| Accuracy | ~81% |
| ROC-AUC | ~0.86 |

## 📊 Model Performance

### Confusion Matrix
<img src="confusion_matrix.png" alt="Confusion Matrix" width="450">

### ROC Curve
<img src="roc_curve.png" alt="ROC Curve" width="450">
## Features Used
Pclass, Sex, Age, Fare, Embarked, FamilySize, IsAlone, Title
## How to Run
```
pip install -r requirements.txt
python titanic_model.py
```
## Dataset
Download train.csv from: https://www.kaggle.com/c/titanic/data
## Author
James Koero |Bsc Physicsand Mathematics| Self-taught ML Engineer| Kisumu, Kenya
 | Email: [jmskoero@gmail.com]
