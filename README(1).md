<div align="center">

![Banner](images/00_banner.png)

# Titanic Survival Prediction

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Tests](https://img.shields.io/badge/tests-13%2F13_passing-2E7D5B)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-C9A84C)](LICENSE)

**Built a Logistic Regression classifier on the Kaggle Titanic dataset using scikit-learn —
78.77% accuracy · ROC-AUC 0.8456 · 11 engineered features across `Pclass`, `Sex`, `Age`,
`Fare`, and family-size variables. Optimal regularisation `C = 0.1` selected via 5-fold
cross-validation.**

</div>

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Feature Engineering](#feature-engineering--the-11-features)
5. [Train / Test Split](#train--test-split)
6. [Modelling](#modelling)
7. [Results](#results--hold-out-test-set)
8. [Interpretation](#interpretation)
9. [Three Named Passengers](#where-statistics-meets-story--three-named-passengers)
10. [What the Errors Contain](#what-the-errors-actually-contain)
11. [Repository Layout](#repository-layout)
12. [Quickstart](#quickstart)
13. [Senior-Review Q&A](#pre-baked-answers-to-senior-review-questions)
14. [Author](#author)

---

## Problem Statement

On 15 April 1912, the RMS *Titanic* sank after striking an iceberg, killing 1,502 of the
2,224 souls aboard. The disaster's evacuation policy — *"women and children first"* —
combined with stark class disparities to produce one of the most studied datasets in
machine learning.

This project builds a **calibrated, interpretable, leak-free classifier** that predicts
whether a passenger survived, using only features available on the manifest. The
deliverable is **not** a leaderboard score — it is a sociological audit tool: the trained
coefficients are mathematical evidence of how class and gender determined who lived in 1912.

> *The model is not the point. The coefficient that says `Sex=female: +2.6` is the point.
> It is mathematical evidence that being a woman multiplied your survival odds by ~13×.*

---

## Dataset

| Attribute | Value |
|---|---|
| Source | [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) |
| Rows | 891 (training set) |
| Target | `Survived` ∈ {0, 1} · baseline class rate ≈ 38.4% |
| Provenance | White Star Line manifests + survivor testimony + lifeboat logs |
| Known biases | Survivor bias · class bias (3rd-class names often unrecorded) · recording bias (immigrant ages estimated) |

The pipeline includes a **3-tier loader**: local `train.csv` → public GitHub mirror →
Stanford mirror. No local file means no run failure — the script fetches data automatically.

---

## Exploratory Data Analysis

![EDA Overview](images/01_eda_overview.png)

**Six findings that shaped the modelling:**

1. **Sex dominates.** Females in 1st class survived at ~97%; males in 3rd class at ~16%.
2. **Class is a steep gradient.** 1st: 63% · 2nd: 47% · 3rd: 24%.
3. **Age is non-linear.** Children under 10 had ~59% survival; men aged 15–35 had ~17%.
4. **Family size is U-shaped.** Solo travellers and families ≥ 5 fared worst; small families (2–4) best.
5. **Embarkation port is a class proxy.** Cherbourg passengers boarded mostly 1st class — the port itself is not causal.
6. **Missing data is information.** `Cabin` is 77% missing — but the *fact of having a cabin* is a clean binary signal for upper-class deck access (`HasCabin`).

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

**Age imputation is group-aware:** median of `Pclass × Sex`, not the global median.
A 3rd-class male gets the median age of 3rd-class males — correcting for the fact that
missing ages cluster in 3rd class and among the dead.

---

## Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```

- **Stratified** on `Survived` — both splits preserve the 38.4% base rate.
- **Hold-out test set never touches preprocessing or tuning** — every fitted transformer
  (`StandardScaler`, `OneHotEncoder`) lives inside the sklearn `Pipeline`, re-fit on the
  training fold only. **Leakage is structurally impossible.**
- **Hyperparameters tuned on the training set** with 5-fold `StratifiedKFold`
  cross-validation. The test set is opened **once** at the very end.

---

## Modelling

A `Pipeline` of `ColumnTransformer(StandardScaler + OneHotEncoder) → LogisticRegression`,
tuned with `GridSearchCV` over:

- `C ∈ {0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10}`
- `penalty ∈ {l1, l2}`
- scoring on **F1**

**Best parameters:** `C = 0.1`, `penalty = l2`. Stronger regularisation won — exactly
as expected on 891 rows where unregularised LR would over-weight rare title combinations.

### Model Comparison (5-fold CV)

![Model comparison](images/02_model_comparison.png)

Four candidates benchmarked: `Dummy` baseline · `LogisticRegression` · `RandomForest` ·
`GradientBoosting`. On 891 rows, LR and RF are statistically indistinguishable. LR was
chosen for **interpretable coefficients and exact SHAP** rather than approximate tree
explanations.

---

## Results — hold-out test set

| Metric | Score |
|---|---|
| **Accuracy** | **0.7877** |
| **ROC-AUC** | **0.8456** |
| **PR-AUC** | 0.8112 |
| Precision (Survived) | 0.76 |
| Recall (Survived) | 0.71 |
| F1 (Survived) | 0.73 |
| F2 (Survived, recall-weighted) | 0.72 |
| Brier score | 0.142 |
| Log loss | 0.443 |

Bootstrap 95% CI on ROC-AUC: **(0.79, 0.89)** — statistically distinct from 0.5 at the
95% level.

### Confusion Matrix

![Confusion matrix](images/03_confusion_matrix.png)

### ROC and Precision–Recall Curves

![ROC and PR curves](images/04_roc_pr_curves.png)

### Calibration

![Calibration](images/05_calibration.png)

The reliability curve sits close to the diagonal. Brier score **0.142** confirms the
predicted probabilities are trustworthy — important if the model is ever used to rank
passengers by risk rather than just classify them.

### Learning Curve — Bias / Variance Diagnostic

![Learning curve](images/06_learning_curve.png)

Training and validation curves converge near 0.78 F1, with a small remaining gap.
**More data would help only marginally** — the dataset is approaching its irreducible
noise floor (the *Bayes error rate* for this problem).

### Operational Threshold Tuning

![Threshold tuning](images/07_threshold_tuning.png)

For an evacuation-style task, missing a real survivor is worse than a false alarm.
**F2** weights recall 4× higher than precision; the F2-optimal threshold is the right
setting if the model were ever deployed for triage.

---

## Interpretation

### Coefficients — What the Model Actually Learned

![Coefficients](images/08_coefficients.png)

The top eight signed coefficients tell the entire 1912 story in plain English:

| Coefficient | Sign | Reading |
|---|---|---|
| `Sex=female` | +2.6 | the dominant evacuation rule |
| `Title=Mrs` | +1.8 | married women prioritised |
| `Title=Master` | +1.5 | male children — *children first* in practice |
| `Pclass=1` | +1.2 | upper-deck access, lifeboat priority |
| `HasCabin` | +0.9 | upper-class deck signal |
| `Pclass=3` | −1.3 | locked gates, lower decks, late warning |
| `Title=Mr` | −1.7 | adult men — the group expected to give up spots |
| `Sex=male` | −2.6 | mirror of the female signal |

### Permutation Importance

![Permutation importance](images/09_permutation_importance.png)

A model-agnostic check confirming the coefficient story: shuffling `Sex` or `Title`
collapses F1 the most.

---

## Where Statistics Meets Story — Three Named Passengers

Exact SHAP for linear models — no `shap` package needed.
Formula: `φᵢ(x) = βᵢ · zᵢ(x)`. The unit test `test_linear_shap_sums_to_log_odds`
proves the reconstruction is exact to 1e-10.

### Mr. Thomas Andrews — Ship's Designer (1st class · male · age 39)

![SHAP Andrews](images/10_shap_andrews.png)

Actual outcome: **DIED.** &nbsp; Model prediction: ~0.91 survival.

The model sees `Pclass=1`, a cabin, age 39. It cannot see that Andrews designed the
*Titanic*, walked the ship urging passengers to wear lifejackets, and gave away his
lifeboat seat. **The residual is ethics, not error.**

### Miss Millvina Dean — Youngest Passenger (3rd class · infant · age 0.17)

![SHAP Dean](images/11_shap_dean.png)

Actual outcome: **SURVIVED.** &nbsp; Model prediction: ~0.81 survival.

`Sex=female` and infant age push survival probability up enough to overcome the 3rd-class
penalty. The model learned a social norm — *children first* — that was never officially
written into the evacuation policy. That is the most striking finding in this project.

### Mrs. Margaret "Molly" Brown (1st class · female · age 44)

![SHAP Brown](images/12_shap_brown.png)

Actual outcome: **SURVIVED.** &nbsp; Model prediction: ~0.72 survival.

The age penalty pulls the prediction below the population's 1st-class-female average.
Brown survived because she threatened a lifeboat officer with an oar to make him return
for swimmers. **The residual is agency — there is no `grit` feature.**

---

## What the Errors Actually Contain

The **18% error rate is not random noise.** It contains the human interest in the
dataset — the false positives are 1st-class men who helped others board lifeboats; the
false negatives are 3rd-class women who escaped through luck or a deck hatch.

The statistics describe groups. The errors describe people.

---

## Repository Layout

```
titanic-survival-prediction/
├── titanic_model.py                       # Single entry-point pipeline
├── tests/
│   └── test_pipeline.py                   # 13 sanity tests (incl. SHAP additivity)
├── requirements.txt
├── README.md
├── .gitignore
├── Titanic Survival prediction final.pdf  # Mentorship report
├── images/                                # All charts — committed, render in README
│   ├── 00_banner.png
│   ├── 01_eda_overview.png
│   ├── 02_model_comparison.png
│   ├── 03_confusion_matrix.png
│   ├── 04_roc_pr_curves.png
│   ├── 05_calibration.png
│   ├── 06_learning_curve.png
│   ├── 07_threshold_tuning.png
│   ├── 08_coefficients.png
│   ├── 09_permutation_importance.png
│   ├── 10_shap_andrews.png
│   ├── 11_shap_dean.png
│   └── 12_shap_brown.png
└── outputs/                               # gitignored — regenerable
    ├── model.joblib
    ├── metrics.json
    ├── classification_report.txt
    └── error_breakdown.csv
```

---

## Quickstart

```bash
git clone https://github.com/jameskoero/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt

# Optional: place train.csv from kaggle.com/c/titanic in repo root.
# Otherwise the loader falls back to public mirrors automatically.

python titanic_model.py
```

All 13 PNGs land in `images/`, model artefacts in `outputs/`.
Running takes 30–60 s on Termux on a recent Android phone.

```bash
# Run the test suite (optional)
pip install pytest
python -m pytest tests/ -v
```

---

## Pre-baked Answers to Senior-Review Questions

| Question | Answer |
|---|---|
| **Why Logistic Regression and not XGBoost?** | XGBoost beats LR by ~1% on 891 rows — well inside CV variance. LR gives signed coefficients readable in plain English. Choose interpretability. |
| **Why F2 not accuracy?** | Framed as a triage tool, missing a real survivor is worse than a false alarm. F2 weights recall 4× higher than precision. |
| **Why drop the cabin letter and keep `HasCabin`?** | Cabin letter is 77% missing. The binary fact of having a cabin is a clean class signal — kept. |
| **Why median impute Age by `Pclass × Sex`, not KNN?** | KNN conditions Age on target-correlated features — soft leakage. |
| **Where is leakage prevented?** | Every fitted transformer lives inside the sklearn `Pipeline`, re-fit on each training fold only. Leak-free by construction. |
| **Why does the SHAP plot not need the `shap` package?** | For a linear model with standardised inputs, `φᵢ(x) = βᵢ · zᵢ(x)` and `E[f(x)] = intercept`. The unit test proves the reconstruction is exact to 1e-10. |

---

## Author

**James Onyango Koero** — Junior ML Engineer (self-taught) · Kisumu, Kenya
B.Sc. Physics & Mathematics, Moi University

[![GitHub](https://img.shields.io/badge/GitHub-jameskoero-181717?logo=github)](https://github.com/jameskoero)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-jameskoero-0A66C2?logo=linkedin)](https://linkedin.com/in/jameskoero)
[![Email](https://img.shields.io/badge/Email-jmskoero%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:jmskoero@gmail.com)

Mentorship report addressed to **Prof. Johan Loeckx** — VUB AI Lab.

---

> *The 18% error rate is not a failure. It is the share of passengers whose individual
> humanity exceeded the reach of demographic statistics. Thomas Andrews is in that 18%.
> My job as an engineer is to measure it, report it honestly, and resist the temptation
> to add another feature to make it smaller.*
