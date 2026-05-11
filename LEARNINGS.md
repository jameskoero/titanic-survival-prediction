# Learning Journey — Titanic Survival Prediction

## Evolution: v1 (2024) → v2 (2026)

| Version | Approach                                              | Accuracy | Key Gap                          |
|---------|-------------------------------------------------------|----------|----------------------------------|
| v1      | Single script, global median impute, no pipeline     | ~78%     | Data leakage risk, no tests      |
| v2      | Leak-free Pipeline, group imputation, GridSearchCV   | 81.01%   | Production-ready                 |

## Key Technical Lessons

1. **Accuracy alone is dishonest** — always report Bootstrap CI, Brier score, F2.
2. **Data leakage** — fitting scaler on full dataset before split is the most common interview failure point. Fixed with Pipeline + ColumnTransformer.
3. **The 18% error rate** — Thomas Andrews had 91% predicted survival. He died giving away his lifeboat seat. The errors contain the humanity.
4. **Android engineering** — built entirely on a phone in Kisumu, Kenya using PyramIDE and Termux. Environment is not an excuse.
5. **Analytical SHAP** — φ_i = β_i × z_i is mathematically exact for linear models and requires no external package.

## What I Would Do Differently
- Start with Pipeline architecture from day one
- Add CI before writing the first test, not after
- Report hold-out AND CV metrics from the beginning — not just one

## Metrics Honesty Note
LinkedIn post reported CV accuracy **82.4%** / AUC **0.868** (training folds).  
Hold-out results: **81.01%** accuracy / **0.8661** AUC (unseen 20% split).  
Both are valid. The difference is explained in the README.
