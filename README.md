<div align="center">



![Banner](images/00_banner.png)



# Titanic Survival Prediction

[

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)

](https://www.python.org/)
[

![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)

](https://scikit-learn.org/)
[

![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)

](https://numpy.org/)
[

![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white)

](https://pandas.pydata.org/)
[

![Tests](https://img.shields.io/badge/tests-13%2F13_passing-2E7D5B)

]()
[

![License: MIT](https://img.shields.io/badge/license-MIT-C9A84C)

](LICENSE)

**Built a Logistic Regression classifier on the Kaggle Titanic dataset using scikit-learn — 78.77 % accuracy, ROC-AUC 0.8456, with 11 engineered features across `Pclass`, `Sex`, `Age`, `Fare`, and family-size variables. Optimal regularisation `C = 0.1` selected via 5-fold cross-validation.**

</div>

---

## Problem Statement

On 15 April 1912, the RMS *Titanic* sank after striking an iceberg, killing 1 502 of the 2 224 souls aboard. The disaster's evacuation policy — *"women and children first"* — combined with stark class disparities aboard ship to produce one of the most studied datasets in machine learning.

This project builds a **calibrated, interpretable, leak-free classifier** that predicts whether a passenger survived, using only features available on the manifest. The deliverable is **not** a leaderboard score — it is a sociological audit tool: the trained coefficients are mathematical evidence of how class and gender determined who lived in 1912.

> *The model is not the point. The coefficient that says `Sex=female: +2.6` is the point. It is mathematical evidence that being a woman multiplied your survival odds by ~13×.*

---

## Dataset

| Attribute | Value |
|---|---|
| Source | [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) |
| Rows | 891 (training set) |
| Target | `Survived` ∈ {0, 1}; baseline class rate ≈ 38.4 % |
| Provenance | White Star Line manifests + survivor testimony + lifeboat logs |
| Known biases | Survivor bias (the dead gave no testimony), class bias (3rd-class names often unrecorded), recording bias (immigrant ages estimated) |

The pipeline includes a 3-tier loader: local `train.csv` → public GitHub mirror → Stanford mirror. No file ⇒ no run failure.

---

## Exploratory Data Analysis



![EDA](images/01_eda_overview.png)



**Six findings that shaped the modelling:**

1. **Sex dominates.** Females in 1st class survived at ~97 %; males in 3rd class at ~16 %.
2. **Class is a steep gradient.** 1st: 63 % · 2nd: 47 % · 3rd: 24 %.
3. **Age is non-linear.** Children under 10 had ~59 % survival; men aged 15–35 had ~17 %.
4. **Family size is U-shaped.** Solo travellers and families ≥ 5 fared worst; small families (2–4) best.
5. **Embarkation port is a class proxy.** Cherbourg passengers boarded mostly 1st class, hence the higher survival; the port itself is not causal.
6. **Missing data is information.** `Cabin` is 77 % missing — but the *fact of having a cabin* is a clean binary signal for upper-class deck access (`HasCabin`).

---

## Feature Engineering — the 11 features

| # | Feature | Type | Source | Why it matters |
|---|---|---|---|---|
| 1 | `Pclass` | numeric | raw | wealth proxy + deck location |
| 2 | `Sex` | categorical | raw | the dominant signal |
| 3 | `Age` | numeric | raw + group-median impute | non-linear, child-priority effect |
| 4 | `SibSp` | numeric | raw | sibling/spouse count |
| 5 | `Parch` | numeric | raw | parent/child count |
| 6 | `Fare` | numeric | raw + median impute | redundant with class but kept for granularity |
| 7 | `Embarked` | categorical | raw + mode impute | port of boarding |
| 8 | `FamilySize` | numeric | engineered | `SibSp + Parch + 1` |
| 9 | `IsAlone` | numeric | engineered | `FamilySize == 1` |
| 10 | `Title` | categorical | engineered from `Name` | `{Mr, Mrs, Miss, Master, Rare}` — captures age × sex × status cleanly |
| 11 | `HasCabin` | numeric | engineered from `Cabin` | binary missingness signal |

**Age imputation is group-aware:** median of `Pclass × Sex`, not the global median. A 3rd-class male gets the median age of 3rd-class males, which corrects for the fact that missing ages cluster in 3rd class and among the dead.

---

## Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
