# Model Card — Titanic Survival Prediction

## Model Details
- **Type:** Logistic Regression (scikit-learn 1.3+)
- **Version:** v2.0 — leak-free Pipeline
- **Author:** James Onyango Koero (jmskoero@gmail.com)
- **Date:** May 2026
- **Repo:** https://github.com/jameskoero/titanic-survival-prediction

## Intended Use
Educational demonstration of ML engineering pipeline for senior-grade roles and MSc applications. Historical analysis of 1912 Titanic survival patterns.

## Data
- **Source:** Kaggle Titanic Dataset (`train.csv`)
- **Size:** 891 passengers, 12 features
- **Period:** April 15, 1912 — historical record
- **License:** Public domain (Kaggle competition data)

## Performance
| Metric       | Score          | 95% CI              |
|--------------|----------------|---------------------|
| Accuracy     | 81.01%         | (74.86%, 86.03%)    |
| ROC-AUC      | 0.8661         | (0.8024, 0.9161)    |
| F1-Score     | 0.7463         | (0.6428, 0.8212)    |
| Brier Score  | 0.1368         | -                   |

## Limitations
- Trained on 1912 data — **NOT** for modern survival prediction
- Missing key features: lifeboat proximity, deck access, crew assistance
- ~18% error rate contains cases where human agency overrode statistics

## Ethical Considerations
- Thomas Andrews (91% predicted survival) died
- Model cannot capture heroism
- **Sex=female** coefficient **+2.6** reflects historical evacuation policy ("women and children first") — **not** a causal ML finding
- **Model must NEVER be used to infer survival likelihood of real people**

## Bias
Third-class passengers systematically under-served in 1912 — model learns this historical injustice.

## SHAP Interpretation (Analytical)
`φ_i = β_1 × z_i` (exact for linear models)  
**Top features:** Sex=female (+2.61), Pclass, HasCabin, Title, Age

---

**GitHub Action Ready** — Ethics & Production Standards ✅
