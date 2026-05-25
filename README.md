# 🚢 Titanic Survival Prediction — Senior-Grade ML Pipeline

![Banner](images/00_banner.png)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![CI](https://github.com/jameskoero/titanic-survival-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/jameskoero/titanic-survival-prediction/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/titanic-survival-prediction/blob/main/titanic_model.py)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://titanic-koero.streamlit.app)
[![Release](https://img.shields.io/github/v/release/jameskoero/titanic-survival-prediction?color=blue&label=v1.0.0)](https://github.com/jameskoero/titanic-survival-prediction/releases)

---

> *"I started this project to predict who survived the Titanic.*
> *I finished it understanding why we probably shouldn't."*
>
> — James Koero · [LinkedIn post](https://www.linkedin.com/in/jameskoero)

---

## The Number That Stopped Me

After training Logistic Regression on 891 passengers, the coefficient for
`Sex = female` came out at **+2.61**.

That means being a woman multiplied your survival odds by **e^2.61 — roughly
13.5 times.**

That is not a model output. That is a **historical document.**

---

## The Moment That Changed How I Think About ML

**Thomas Andrews. Ship's designer. Age 39. First class. Had a cabin.**

My model predicted: **91% chance of survival.**

He died.

He spent his final hours giving away his lifeboat seat and telling passengers
to put on lifejackets. He was last seen alone in the first class smoking room,
staring at a painting.

The model was not wrong about the statistics.
He was in the **18% error rate.**
That 18% contains all the humanity.

→ See his SHAP waterfall: [`images/10_shap_andrews.png`](images/10_shap_andrews.png)

---

## 📊 Results

### Hold-Out Test Set (20% stratified split — never seen during training)

| Metric | Score | 95% Bootstrap CI |
|--------|-------|-----------------|
| **Accuracy** | **81.01%** | (74.86%, 86.03%) |
| **ROC-AUC** | **0.8661** | (0.8024, 0.9161) |
| **F1-Score** | **0.7463** | (0.6428, 0.8212) |
| **F2-Score** | 0.7331 | — |
| **Precision** | 0.7692 | — |
| **Recall** | 0.7246 | — |
| **PR-AUC** | 0.8184 | — |
| **Brier Score** | 0.1368 | lower = better |
| **Log-Loss** | 0.4351 | lower = better |
| **Best C** | 0.5 | from GridSearchCV |
| **Best Penalty** | L2 | — |

### Cross-Validated Benchmark (5-fold StratifiedKFold, training set)

| Model | CV Accuracy | CV F1 | CV ROC-AUC |
|-------|------------|-------|------------|
| **Logistic Regression** ✅ | **0.7767** | **0.6941** | **0.8299** |
| Random Forest | 0.7753 | 0.6718 | 0.8264 |
| Gradient Boosting | 0.7570 | 0.6562 | 0.8088 |
| Dummy (most_frequent) | 0.6166 | 0.0000 | 0.5000 |

### Threshold Tuning

| Threshold | Optimises | Score |
|-----------|-----------|-------|
| Default | — | 0.50 |
| **F1-optimal** | Precision/Recall balance | **0.36** (F1 = 0.7785) |
| **F2-optimal** | Recall — evacuation framing | **0.31** (F2 = 0.8172) |

> **On metric reporting:** CV accuracy (from training folds) gives the
> published estimate. Hold-out accuracy (unseen 20% split) is the conservative
> real-world estimate. The [LinkedIn post](https://www.linkedin.com/in/jameskoero)
> reported **CV accuracy 82.4% · ROC-AUC 0.868** — directionally consistent
> with these results (hold-out 81.01%, ROC-AUC 0.8661).

---

## Why Logistic Regression Over Gradient Boosting?

Gradient Boosting scored higher on some runs. Logistic Regression was
still chosen — not because the number was better, but because
**the coefficients are readable.**

When `Sex=female` coefficient tells you more about 1912 society than
any leaderboard position, that is the right model.

---

## 🧠 Methodology

### 11 Features → ~18 Model Columns After Encoding

| # | Feature | Type | Engineering |
|---|---------|------|-------------|
| 1 | `Pclass` | Numeric | Original — ticket class (1/2/3) |
| 2 | `Age` | Numeric | **Group-aware imputation: Pclass × Sex median** |
| 3 | `SibSp` | Numeric | Original |
| 4 | `Parch` | Numeric | Original |
| 5 | `Fare` | Numeric | Original (median-imputed for 1 missing) |
| 6 | `FamilySize` | Numeric | **Engineered**: SibSp + Parch + 1 |
| 7 | `IsAlone` | Numeric | **Engineered**: 1 if FamilySize == 1 |
| 8 | `HasCabin` | Numeric | **Engineered**: 1 if cabin record exists |
| 9 | `Sex` | Categorical → OneHot | Original |
| 10 | `Title` | Categorical → OneHot | **Engineered**: extracted from Name |
| 11 | `Embarked` | Categorical → OneHot | Original |

### Leak-Free sklearn Pipeline

```
engineer_features(df)
        │
        ▼
Pipeline([
  ("prep", ColumnTransformer([
      ("num", StandardScaler(),    NUMERIC[8])   ← fit on X_train ONLY
      ("cat", OneHotEncoder(),  CATEGORICAL[3])  ← fit on X_train ONLY
  ])),
  ("clf", LogisticRegression(...))
])
        │
        ▼
GridSearchCV(scoring="f1", cv=StratifiedKFold(5), C=[0.001→10], penalty=[L1,L2])
        │
        ▼
evaluate_holdout(X_test)    ← test set never touched during fitting
```

---

## What the Errors Actually Are

> *"The model learned the average. The errors are the exceptions.*
> *And in the exceptions, you find the actual human story."*

**False Positives** — predicted survival, actually died:
First-class men who statistically should have survived.
Many gave up their lifeboat seats. Thomas Andrews was the starkest case.

**False Negatives** — predicted death, actually survived:
Third-class women the model marked as unlikely to survive.
Several got through by persistence, luck, or finding an unlocked deck hatch.


#### Named Examples from the Hold-Out Set

| Category | Passenger | Prediction | Actual | Why the Model Was Wrong |
|----------|-----------|-----------|--------|------------------------|
| **False Positive** | Thomas Andrews — 1st class, male, HasCabin, Age 39 | **91% survive** | **DIED** | Gave his lifeboat seat away. The model sees class and cabin number. It cannot see a moral decision. |
| **False Negative** | Third-class women, no cabin, travelling alone | ~30–45% survive | **SURVIVED** | Deck gates on lower decks were locked or delayed. Some found gaps. Persistence is not a column in any CSV. |

> The features that actually determined survival — proximity to a lifeboat,
> whether the gates were locked, whether a crew member helped you —
> **are not in the CSV. They are in the historical archive.**

---

## 🔍 SHAP — Three Named Passengers

Exact SHAP values computed analytically from LR coefficients
(`φ_i = β_i × z_i` — no external shap package needed):

| Passenger | Profile | Model Prediction | Historical Outcome |
|-----------|---------|-----------------|-------------------|
| **Mr. Thomas Andrews** | 1st · male · 39 · HasCabin | **91% survival** | **DIED** — gave lifeboat seat away |
| **Miss Millvina Dean** | 3rd · female · 0.17yrs | High survival | **SURVIVED** — youngest; last survivor (d.2009) |
| **Mrs. Molly Brown** | 1st · female · 44 · HasCabin | High survival | **SURVIVED** — helped row Lifeboat 6 |

Charts: [`10_shap_andrews.png`](images/10_shap_andrews.png) ·
[`11_shap_dean.png`](images/11_shap_dean.png) ·
[`12_shap_brown.png`](images/12_shap_brown.png)

---

## 📈 13 Charts Generated

**`images/01_eda_overview.png`** — 6-panel EDA: class×sex, age, fare, family, embarkation, missingness

![EDA Overview](images/01_eda_overview.png)

---

**`images/02_model_comparison.png`** — 4-model CV benchmark

![Model Comparison](images/02_model_comparison.png)

---

**`images/03_confusion_matrix.png`** — Hold-out confusion matrix

![Confusion Matrix](images/03_confusion_matrix.png)

---

**`images/04_roc_pr_curves.png`** — ROC (AUC=0.8661) + Precision-Recall

![ROC & PR Curves](images/04_roc_pr_curves.png)

---

**`images/05_calibration.png`** — Reliability curve — Brier=0.1368

![Calibration](images/05_calibration.png)

---

**`images/06_learning_curve.png`** — Train vs validation F1 — bias/variance

![Learning Curve](images/06_learning_curve.png)

---

**`images/07_threshold_tuning.png`** — F1/F2 optimal thresholds (0.36 / 0.31)

![Threshold Tuning](images/07_threshold_tuning.png)

---

**`images/08_coefficients.png`** — Standardised LR coefficients

![Coefficients](images/08_coefficients.png)

---

**`images/09_permutation_importance.png`** — Permutation Δ F1 by feature

![Permutation Importance](images/09_permutation_importance.png)

---

**`images/10_shap_andrews.png`** — SHAP waterfall — Thomas Andrews

![SHAP Andrews](images/10_shap_andrews.png)

---

**`images/11_shap_dean.png`** — SHAP waterfall — Millvina Dean

![SHAP Dean](images/11_shap_dean.png)

---

**`images/12_shap_brown.png`** — SHAP waterfall — Molly Brown

![SHAP Brown](images/12_shap_brown.png)

---

## 🃏 Model Card

> Ethics and production standards — GitHub Action Ready ✅

| Field | Detail |
|-------|--------|
| **Type** | Logistic Regression (scikit-learn 1.3+) |
| **Version** | v2.0 — leak-free Pipeline |
| **Author** | James Onyango Koero (jmskoero@gmail.com) |
| **Date** | May 2026 |
| **Repo** | [github.com/jameskoero/titanic-survival-prediction](https://github.com/jameskoero/titanic-survival-prediction) |
| **Intended Use** | Educational · Portfolio · Historical analysis |
| **NOT for** | Modern survival prediction of real people |

### Performance

| Metric | Score | 95% CI |
|--------|-------|--------|
| Accuracy | **81.01%** | (74.86%, 86.03%) |
| ROC-AUC | **0.8661** | (0.8024, 0.9161) |
| F1-Score | **0.7463** | (0.6428, 0.8212) |
| Brier Score | **0.1368** | lower = better |

### Limitations

- Trained on **1912 historical data** — NOT applicable to modern survival scenarios
- Missing key real-world features: lifeboat proximity, deck access, crew assistance
- ~18% error rate contains cases where **human agency overrode statistics**
- Thomas Andrews (91% predicted survival) died — the model cannot capture heroism

### Ethical Considerations

> ⚠️ **This model must NEVER be used to infer survival likelihood of real people.**

- `Sex=female` coefficient **+2.61** reflects historical evacuation **policy** ("women and children first") — **not** a causal ML finding
- Thomas Andrews had a 91% predicted survival probability and chose to die giving away his lifeboat seat — the 18% error rate contains all the humanity
- Third-class passengers were systematically under-served in 1912 — the model learns this historical injustice as a statistical pattern

### Bias

Third-class passengers have lower predicted survival rates not because of any
intrinsic characteristic, but because the 1912 evacuation was structurally
biased against them. Pclass is a proxy for **class-based discrimination**,
not individual capability.

### SHAP Interpretation (Analytical)

Formula: `φ_i = β_i × z_i` — exact for linear models, no external shap package required.

**Top features by absolute coefficient:**

| Feature | Coefficient | Interpretation |
|---------|------------|----------------|
| `Sex=female` | **+2.61** | e^2.61 ≈ 13.5× survival odds — historical document |
| `Pclass=3` | negative | Third-class structural disadvantage |
| `HasCabin` | positive | Proxy for wealth and deck proximity |
| `Title` | varies | Social status signal |
| `Age` | negative | Younger passengers prioritised |

---

## 📁 Project Structure

```
titanic-survival-prediction/
├── titanic_model.py           # Senior-grade pipeline — run this
├── api.py                     # FastAPI endpoint
├── app.py                     # Streamlit demo
├── outputs/                   # gitignored
│   ├── model.joblib
│   ├── metrics.json
│   ├── error_breakdown.csv
│   └── classification_report.txt
├── images/                    # 13 PNG charts — committed to repo
├── tests/
│   └── test_pipeline.py       # 12 pytest unit tests
├── .github/workflows/
│   └── ci.yml                 # GitHub Actions CI
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### ☁️ Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/titanic-survival-prediction/blob/main/titanic_model.py)

```python
!git clone https://github.com/jameskoero/titanic-survival-prediction.git
%cd titanic-survival-prediction
!pip install -r requirements.txt -q
!python titanic_model.py  # train.csv auto-downloads
```

### 💻 Local
```bash
git clone https://github.com/jameskoero/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
python titanic_model.py
```

### 📱 Android (PyramIDE / Termux)
```bash
pip install scikit-learn pandas numpy matplotlib joblib
python titanic_model.py
# matplotlib.use("Agg") already set — charts save to images/ silently
```

### 🌐 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://titanic-koero.streamlit.app)

```bash
# API — POST /predict with passenger JSON
curl -X POST https://titanic-api-56uu.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass":1,"Sex":"male","Age":39,"SibSp":0,"Parch":0,"Fare":42,"Embarked":"S","Cabin":"A36"}'
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Language |
| scikit-learn | ≥ 1.3 | Pipeline, GridSearchCV, StratifiedKFold, all metrics |
| pandas | ≥ 1.5 | Data + feature engineering |
| NumPy | ≥ 1.24 | Numerics + bootstrap CIs |
| matplotlib | ≥ 3.5 | 13 charts — no seaborn |
| joblib | ≥ 1.3 | Pipeline serialisation |
| FastAPI | ≥ 0.100 | REST API endpoint |
| Streamlit | ≥ 1.10 | Interactive web demo |

> No `shap` package required — SHAP computed analytically: `φ_i = β_i × z_i`

---

## 🌍 Why This Matters for African ML Engineers

One thing not in any tutorial: this project was built entirely on an Android phone.
Termux terminal. PyramIDE editor. Google Colab via mobile browser for jobs that
needed more RAM. No MacBook, no conda, no reliable broadband.

When numpy refused to compile on ARM64, I had to understand the difference between
a binary wheel and a source build — not by choice, but because the error left me
no option. The fix: `pkg install python-numpy` first, then
`pip install --break-system-packages` for the rest. Two hours of reading error
messages produced one working command. That is a different kind of learning than
following a tutorial on a fast machine.

If a production pipeline runs from a phone in Kisumu with bad internet, it runs
anywhere. That constraint shaped everything about how I built this.


> *"I am building this from Kisumu, Kenya, mentored by Prof. Johan Loeckx at VUB AI Lab.*
> *Production ML is not about the accuracy score.*
> *It is about understanding what your model measures — and being honest about what it does not.*
> *That principle applies whether you are predicting Titanic survival or African tech salaries."*

---

## 🗺️ Roadmap

- [x] Leak-free sklearn Pipeline + ColumnTransformer
- [x] StratifiedKFold — preserves 38/62 class ratio per fold
- [x] Group-aware Age imputation (Pclass × Sex median)
- [x] HasCabin feature
- [x] GridSearchCV on F2 — 8 C values × L1/L2
- [x] DummyClassifier baseline
- [x] Bootstrap 95% CIs on accuracy, F1, ROC-AUC
- [x] Brier score + calibration reliability curve
- [x] F1 and F2 threshold tuning
- [x] 13-chart evaluation dashboard
- [x] Exact SHAP waterfalls: Andrews / Dean / Brown
- [x] Error breakdown by Pclass × Sex
- [x] 3-tier data download fallback
- [x] `tests/test_pipeline.py` — 12 pytest unit tests + CI badge
- [x] FastAPI endpoint — live at https://titanic-api-56uu.onrender.com/docs
- [x] Streamlit web demo — live at https://titanic-koero.streamlit.app
- [x] Model Card — ethics, bias, SHAP interpretation
- [x] LEARNINGS.md — v1 to v2 growth journey
- [ ] Zenodo DOI — citable research output

---

## 👤 Author

> **On AI tooling:** GitHub Copilot was used as a coding assistant during
> development. Every suggestion was read, tested, and understood before accepting.
> At one point Copilot generated a tree-based SHAP approximation for what was
> logistic regression code — the analytical formula φᵢ = βᵢ × zᵢ is exact for
> linear models and needs no approximation. Caught it because I had read the
> formula first. Using AI tools is part of modern ML engineering. Knowing when to
> reject the output is the skill.


**James Onyango Koero**
BSc Physics & Mathematics — Moi University (2012) | Self-Taught ML Engineer | Kisumu, Kenya
📧 [jmskoero@gmail.com](mailto:jmskoero@gmail.com) · [GitHub](https://github.com/jameskoero) · [LinkedIn](https://linkedin.com/in/jameskoero)

*Mentored by:*
- **Prof. Johan Loeckx** — VUB AI Lab, Vrije Universiteit Brussel, Belgium
- **Prof. Samuel Liyala** — JOOUST, Kenya

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

> *"What is living in YOUR model's 18% error rate?"*
