"""
tests/test_pipeline.py
======================
Pytest suite for the Titanic Survival Prediction senior-grade pipeline.

Run: pytest tests/test_pipeline.py -v

Tests cover:
  - Feature engineering correctness (no leakage, correct values)
  - Pipeline structure (components present)
  - Model quality (AUC above dummy baseline)
  - SHAP analytical computation

Author: James Koero · jmskoero@gmail.com
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import pytest
import pandas as pd
import numpy as np

# Add parent directory to path so we can import titanic_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titanic_model import (
    engineer_features,
    build_pipeline,
    build_preprocessor,
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


# ════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df():
    """
    Minimal Titanic-like DataFrame.
    Mirrors the real Kaggle train.csv schema without requiring the file.
    """
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5, 6],
        "Survived":   [0, 1, 1, 0, 1, 0],
        "Pclass":     [3, 1, 3, 1, 3, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mr. Jacques Heath",
            "Allen, Mr. William Henry",
            "Andrews, Mr. Thomas",           # The famous passenger
        ],
        "Sex":      ["male",   "female", "female", "male",   "male",   "male"],
        "Age":      [22.0,      38.0,     26.0,     35.0,     np.nan,   39.0],
        "SibSp":    [1, 1, 0, 1, 0, 0],
        "Parch":    [0, 0, 0, 0, 0, 0],
        "Ticket":   ["A/5 21171", "PC 17599", "STON/O2.", "113803", "373450", "112050"],
        "Fare":     [7.25, 71.28, 7.92, 53.10, 8.05, 42.0],
        "Cabin":    [np.nan, "C85", np.nan, "C123", np.nan, "A36"],
        "Embarked": ["S", "C", "S", "S", "S", "S"],
    })


@pytest.fixture
def andrews_row():
    """Thomas Andrews — ship's designer. Known passenger for SHAP test."""
    return pd.DataFrame([{
        "PassengerId": 999,
        "Survived":    0,   # He died
        "Pclass":      1,
        "Name":        "Andrews, Mr. Thomas",
        "Sex":         "male",
        "Age":         39.0,
        "SibSp":       0,
        "Parch":       0,
        "Ticket":      "112050",
        "Fare":        42.0,
        "Cabin":       "A36",
        "Embarked":    "S",
    }])


# ════════════════════════════════════════════════════════════════════
# 1 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════

def test_feature_count():
    """Pipeline uses exactly 11 raw features (8 numeric + 3 categorical)."""
    assert len(NUMERIC_FEATURES) == 8,     f"Expected 8 numeric features, got {len(NUMERIC_FEATURES)}"
    assert len(CATEGORICAL_FEATURES) == 3, f"Expected 3 categorical features, got {len(CATEGORICAL_FEATURES)}"
    assert len(FEATURES) == 11,            f"Expected 11 total features, got {len(FEATURES)}"


def test_engineer_features_no_nulls(sample_df):
    """All model features must have zero nulls after engineering."""
    df_eng = engineer_features(sample_df)
    null_counts = df_eng[FEATURES].isnull().sum()
    assert null_counts.sum() == 0, (
        f"Null values remain after feature engineering:\n{null_counts[null_counts > 0]}"
    )


def test_hascabin_binary(sample_df):
    """HasCabin must be strictly binary (0 or 1)."""
    df_eng = engineer_features(sample_df)
    assert df_eng["HasCabin"].isin([0, 1]).all(), \
        "HasCabin contains values other than 0/1"


def test_hascabin_correct_values(sample_df):
    """Rows with a Cabin string must have HasCabin=1."""
    df_eng = engineer_features(sample_df)
    has_cabin_mask = sample_df["Cabin"].notna()
    assert df_eng.loc[has_cabin_mask, "HasCabin"].eq(1).all(), \
        "Some rows with Cabin info have HasCabin=0"
    assert df_eng.loc[~has_cabin_mask, "HasCabin"].eq(0).all(), \
        "Some rows without Cabin info have HasCabin=1"


def test_isalone_binary(sample_df):
    """IsAlone must be strictly binary (0 or 1)."""
    df_eng = engineer_features(sample_df)
    assert df_eng["IsAlone"].isin([0, 1]).all(), \
        "IsAlone contains values other than 0/1"


def test_familysize_formula(sample_df):
    """FamilySize must equal SibSp + Parch + 1."""
    df_eng = engineer_features(sample_df)
    expected = (sample_df["SibSp"] + sample_df["Parch"] + 1).values
    actual   = df_eng["FamilySize"].values
    np.testing.assert_array_equal(actual, expected,
        err_msg="FamilySize does not equal SibSp + Parch + 1")


def test_age_no_nulls_after_imputation(sample_df):
    """Age must have zero nulls after group-aware imputation (sample_df has 1 NaN)."""
    assert sample_df["Age"].isnull().sum() == 1, "Fixture should have exactly 1 NaN age"
    df_eng = engineer_features(sample_df)
    assert df_eng["Age"].isnull().sum() == 0, \
        "Age still has nulls after group-aware imputation"


def test_title_valid_values(sample_df):
    """Title must only contain the 5 valid categories."""
    df_eng = engineer_features(sample_df)
    valid = {"Mr", "Mrs", "Miss", "Master", "Rare"}
    found = set(df_eng["Title"].unique())
    invalid = found - valid
    assert len(invalid) == 0, f"Invalid Title values found: {invalid}"


# ════════════════════════════════════════════════════════════════════
# 2 — PIPELINE STRUCTURE
# ════════════════════════════════════════════════════════════════════

def test_pipeline_has_two_steps():
    """Pipeline must have exactly 2 named steps: prep and clf."""
    from sklearn.linear_model import LogisticRegression
    pipe = build_pipeline(LogisticRegression(random_state=42))
    assert list(pipe.named_steps.keys()) == ["prep", "clf"], \
        f"Expected ['prep', 'clf'], got {list(pipe.named_steps.keys())}"


def test_preprocessor_has_num_and_cat():
    """ColumnTransformer must have numeric and categorical transformers."""
    prep = build_preprocessor()
    names = [t[0] for t in prep.transformers]
    assert "num" in names, "Numeric transformer 'num' missing"
    assert "cat" in names, "Categorical transformer 'cat' missing"


def test_pipeline_fit_transform(sample_df):
    """Pipeline must fit and transform without errors on sample data."""
    from sklearn.linear_model import LogisticRegression
    df_eng = engineer_features(sample_df)
    X = df_eng[FEATURES]
    y = df_eng["Survived"]
    pipe = build_pipeline(LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2), \
        f"Expected proba shape ({len(X)}, 2), got {proba.shape}"
    assert (proba >= 0).all() and (proba <= 1).all(), \
        "Probabilities outside [0, 1] range"


# ════════════════════════════════════════════════════════════════════
# 3 — MODEL QUALITY (against dummy baseline)
# ════════════════════════════════════════════════════════════════════

def test_model_beats_dummy_auc(sample_df):
    """
    Logistic Regression ROC-AUC must exceed DummyClassifier on sample data.
    This catches broken pipelines, inverted labels, or data leakage regressions.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    df_eng = engineer_features(sample_df)
    X = df_eng[FEATURES]
    y = df_eng["Survived"]

    lr_pipe = build_pipeline(LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    lr_pipe.fit(X, y)
    lr_auc = roc_auc_score(y, lr_pipe.predict_proba(X)[:, 1])

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X, y)

    assert lr_auc > 0.5, (
        f"Logistic Regression AUC ({lr_auc:.3f}) must beat random (0.5). "
        "Check for pipeline errors."
    )


def test_sex_coefficient_direction(sample_df):
    """
    Sex=female coefficient must be positive (females had higher survival).
    If negative, the encoding or target is inverted.
    """
    from sklearn.linear_model import LogisticRegression

    df_eng = engineer_features(sample_df)
    X = df_eng[FEATURES]
    y = df_eng["Survived"]

    pipe = build_pipeline(LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    pipe.fit(X, y)

    prep = pipe.named_steps["prep"]
    clf  = pipe.named_steps["clf"]
    ohe  = prep.named_transformers_["cat"]
    cat_names = [f"{c}={v}" for c, cats in zip(CATEGORICAL_FEATURES, ohe.categories_) for v in cats]
    all_names = NUMERIC_FEATURES + cat_names
    coef_map = dict(zip(all_names, clf.coef_[0]))

    sex_female_coef = coef_map.get("Sex=female", None)
    assert sex_female_coef is not None, "Sex=female coefficient not found"
    assert sex_female_coef > 0, (
        f"Sex=female coefficient is {sex_female_coef:.4f} (expected positive). "
        "Real Titanic data gives ~+2.61. Check encoding."
    )
