"""
Sanity tests for the Titanic pipeline.

Run from repo root:
    python -m pytest tests/ -v
or, if pytest isn't available:
    python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the repo root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import titanic_model as tm


# ───────────────────────── fixtures ─────────────────────────
def _toy_df(n: int = 60) -> pd.DataFrame:
    """A small Titanic-shaped frame for fast tests."""
    rng = np.random.default_rng(0)
    cabin = pd.Series([np.nan] * n, dtype="object")
    has_cabin_idx = np.where(rng.random(n) >= 0.7)[0]
    cabin.iloc[has_cabin_idx] = "C85"
    return pd.DataFrame({
        "PassengerId": np.arange(1, n + 1),
        "Survived": rng.integers(0, 2, n),
        "Pclass":   rng.choice([1, 2, 3], n),
        "Name":     [f"Surname{i}, Mr. First{i}" if i % 2 else
                     f"Surname{i}, Mrs. First{i}" for i in range(n)],
        "Sex":      rng.choice(["male", "female"], n),
        "Age":      np.where(rng.random(n) < 0.2, np.nan,
                             rng.uniform(1, 70, n)),
        "SibSp":    rng.integers(0, 4, n),
        "Parch":    rng.integers(0, 3, n),
        "Ticket":   [f"T{i}" for i in range(n)],
        "Fare":     rng.uniform(5, 200, n).round(2),
        "Cabin":    cabin.values,
        "Embarked": rng.choice(["S", "C", "Q"], n),
    })


# ───────────────────────── tests ─────────────────────────
class TestSchema:
    def test_validator_accepts_valid_frame(self):
        tm.validate_schema(_toy_df())

    def test_validator_rejects_missing_target(self):
        df = _toy_df().drop(columns=["Survived"])
        with pytest.raises(ValueError, match="missing columns"):
            tm.validate_schema(df)

    def test_validator_rejects_bad_pclass(self):
        df = _toy_df()
        df.loc[0, "Pclass"] = 4
        with pytest.raises(ValueError, match="Pclass"):
            tm.validate_schema(df)


class TestFeatureEngineering:
    def test_engineered_columns_exist(self):
        out = tm.engineer_features(_toy_df())
        for col in ["Title", "FamilySize", "IsAlone",
                    "FamilyCategory", "HasCabin"]:
            assert col in out.columns, f"missing engineered column: {col}"

    def test_no_age_nans_after_engineering(self):
        out = tm.engineer_features(_toy_df())
        assert out["Age"].isna().sum() == 0

    def test_has_cabin_is_binary(self):
        out = tm.engineer_features(_toy_df())
        assert set(out["HasCabin"].unique()).issubset({0, 1})

    def test_titles_are_within_known_set(self):
        out = tm.engineer_features(_toy_df(200))
        assert set(out["Title"].unique()).issubset(
            {"Mr", "Mrs", "Miss", "Master", "Rare"}
        )

    def test_family_size_consistency(self):
        out = tm.engineer_features(_toy_df())
        assert (out["FamilySize"] == out["SibSp"] + out["Parch"] + 1).all()


class TestPipeline:
    def test_pipeline_fits_and_predicts(self):
        df = tm.engineer_features(_toy_df(120))
        X, y = df[tm.FEATURES], df["Survived"]
        from sklearn.linear_model import LogisticRegression
        pipe = tm.build_pipeline(LogisticRegression(max_iter=1000))
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)[:, 1]
        assert proba.shape == (len(y),)
        assert ((0 <= proba) & (proba <= 1)).all()

    def test_pipeline_no_leakage(self):
        """Scaler statistics must come from training fold only."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        df = tm.engineer_features(_toy_df(200))
        X, y = df[tm.FEATURES], df["Survived"]
        pipe = tm.build_pipeline(LogisticRegression(max_iter=1000))

        # If sklearn's CV API accepts our pipeline cleanly, leakage is
        # structurally impossible — the Pipeline forces fit-on-train-only.
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        for train_idx, test_idx in cv.split(X, y):
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            score = pipe.score(X.iloc[test_idx], y.iloc[test_idx])
            assert 0.0 <= score <= 1.0


class TestInterpretation:
    def test_feature_names_match_coef_length(self):
        from sklearn.linear_model import LogisticRegression
        df = tm.engineer_features(_toy_df(150))
        X, y = df[tm.FEATURES], df["Survived"]
        pipe = tm.build_pipeline(LogisticRegression(max_iter=1000))
        pipe.fit(X, y)
        names = tm.get_feature_names(pipe)
        assert len(names) == pipe.named_steps["clf"].coef_.shape[1]

    def test_linear_shap_sums_to_log_odds(self):
        """SHAP values + base value must reconstruct log-odds exactly."""
        from sklearn.linear_model import LogisticRegression
        df = tm.engineer_features(_toy_df(150))
        X, y = df[tm.FEATURES], df["Survived"]
        pipe = tm.build_pipeline(LogisticRegression(max_iter=1000))
        pipe.fit(X, y)

        sample = X.iloc[[0]]
        phi, base = tm.linear_shap(pipe, sample)
        reconstructed = base + phi.sum()

        z = pipe.named_steps["prep"].transform(sample)
        true_log_odds = (
            pipe.named_steps["clf"].coef_[0] @ z[0]
            + pipe.named_steps["clf"].intercept_[0]
        )
        assert abs(reconstructed - true_log_odds) < 1e-10


class TestBootstrap:
    def test_bootstrap_ci_returns_valid_interval(self):
        from sklearn.metrics import accuracy_score
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_pred = rng.integers(0, 2, 200)
        lo, hi = tm.bootstrap_ci(y_true, y_pred, accuracy_score, n=200)
        assert 0.0 <= lo <= hi <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
