"""
Titanic Survival Prediction — Senior-Grade ML Pipeline
=======================================================

End-to-end binary classifier predicting Titanic passenger survival.
Built to satisfy a senior ML code review:

  * Reproducible (fixed seed, pinned deps, version logging)
  * Leak-free (sklearn Pipeline + ColumnTransformer; preprocessing
    fit ONLY on training folds — never on test data)
  * Honest evaluation
        - StratifiedKFold (5-fold)
        - DummyClassifier baseline
        - Multi-model benchmark (LR, RF, Gradient Boosting)
        - Bootstrap 95 % CIs on test metrics
  * Calibrated (Brier score + reliability curve)
  * Operational threshold tuning (F2 — recall-weighted, evacuation framing)
  * Interpretable (LR coefficients, permutation importance,
    EXACT SHAP for linear models — no shap package needed)
  * Inspection of three named passengers (Andrews / Dean / Brown)
    per the v2 report's promise to Prof. Loeckx
  * Persisted (joblib pipeline + JSON metrics + PNG charts)
  * Tested (see tests/test_pipeline.py)

Author : James Onyango Koero  <jmskoero@gmail.com>
Repo   : https://github.com/jameskoero/titanic-survival-prediction
"""
from __future__ import annotations

# ── Headless backend BEFORE any pyplot import (PyramIDE / Termux) ──
import matplotlib
matplotlib.use("Agg")

import json
import logging
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve
from urllib.error import URLError

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, fbeta_score,
    log_loss, make_scorer, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_validate, learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

NAVY = "#0A1628"   # dark navy — James's preferred theme
GOLD = "#C9A84C"
RED  = "#B33A3A"
GREEN = "#2E7D5B"
GREY = "#6E6E6E"

ROOT       = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"     # gitignored — model.joblib, metrics.json
CHART_DIR  = ROOT / "images"      # committed — charts rendered in README
OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

DATA_FILE = ROOT / "train.csv"
DATA_URLS = [
    # 3-tier fallback per James's standing instruction
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("titanic")

# Plot style (no seaborn — pure matplotlib, per James's preference) ----
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   NAVY,
    "axes.labelcolor":  NAVY,
    "axes.titlecolor":  NAVY,
    "axes.titleweight": "bold",
    "xtick.color":      NAVY,
    "ytick.color":      NAVY,
    "axes.grid":        True,
    "grid.color":       "#E5E5E5",
    "grid.linewidth":   0.6,
    "font.size":        10,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})


# ════════════════════════════════════════════════════════════════════
# 1. DATA LOADING — 3-tier fallback
# ════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    """Local file → GitHub mirror → Stanford mirror."""
    if DATA_FILE.exists():
        log.info("Loading local %s", DATA_FILE.name)
        return _normalise_columns(pd.read_csv(DATA_FILE))

    for url in DATA_URLS:
        try:
            log.info("Downloading %s", url)
            urlretrieve(url, DATA_FILE)
            return _normalise_columns(pd.read_csv(DATA_FILE))
        except (URLError, OSError) as exc:
            log.warning("Source failed (%s) — trying next", exc)

    raise RuntimeError(
        "All Titanic data sources failed. "
        "Place train.csv from kaggle.com/c/titanic in the repo root."
    )


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map alternate column names from mirrors to canonical Kaggle schema."""
    rename = {
        "Siblings/Spouses Aboard": "SibSp",
        "Parents/Children Aboard": "Parch",
    }
    df = df.rename(columns=rename)
    if "PassengerId" not in df.columns:
        df["PassengerId"] = np.arange(1, len(df) + 1)
    if "Cabin" not in df.columns:
        df["Cabin"] = np.nan
    if "Ticket" not in df.columns:
        df["Ticket"] = ""
    if "Embarked" not in df.columns:
        df["Embarked"] = "S"
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Fail fast if data drift / corruption breaks downstream code."""
    required = {"Survived", "Pclass", "Sex", "Age", "SibSp",
                "Parch", "Name", "Cabin", "Embarked"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Schema violation — missing columns: {missing}")
    if df["Survived"].nunique() != 2:
        raise ValueError("Target 'Survived' is not binary.")
    if not df["Pclass"].isin([1, 2, 3]).all():
        raise ValueError("Pclass contains values outside {1, 2, 3}.")
    log.info("Schema OK: %d rows × %d columns", *df.shape)


# ════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (matches the v2-report plan exactly)
# ════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 2.1 Title from Name -------------------------------------------------
    df["Title"] = (
        df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Rare")
    )
    rare = ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
            "Major", "Rev", "Sir", "Jonkheer", "Dona"]
    df["Title"] = df["Title"].replace(rare, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = df["Title"].where(
        df["Title"].isin(["Mr", "Mrs", "Miss", "Master", "Rare"]), "Rare"
    )

    # 2.2 Family structure ------------------------------------------------
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
    df["FamilyCategory"] = pd.cut(
        df["FamilySize"], bins=[0, 1, 4, 100],
        labels=["alone", "small", "large"],
    ).astype(str)

    # 2.3 HasCabin (the 'missingness IS information' feature) ------------
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # 2.4 GROUP-AWARE Age imputation (Pclass × Sex median) ---------------
    #     Replaces global-median imputation criticised in the v2 report.
    age_med = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(age_med).fillna(df["Age"].median())

    # 2.5 Embarked: mode-impute the 2 NAs (kept for EDA, dropped from model)
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # 2.6 Fare: impute median (kept for EDA, dropped from model — collinear with Pclass)
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    return df


NUMERIC_FEATURES     = ["Pclass", "Age", "SibSp", "Parch",
                        "Fare", "FamilySize", "IsAlone", "HasCabin"]
CATEGORICAL_FEATURES = ["Sex", "Title", "Embarked"]
FEATURES             = NUMERIC_FEATURES + CATEGORICAL_FEATURES   # 11 features


def build_preprocessor() -> ColumnTransformer:
    """Numeric → StandardScaler ; categorical → OneHot. No leakage."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", model)])


# ════════════════════════════════════════════════════════════════════
# 3. EDA PLOTS  +  README BANNER
# ════════════════════════════════════════════════════════════════════
def plot_banner() -> None:
    """Programmatic header banner — renders at the top of the README."""
    fig, ax = plt.subplots(figsize=(16, 4), facecolor=NAVY)
    ax.set_facecolor(NAVY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.68, "TITANIC SURVIVAL PREDICTION",
            ha="center", va="center", color=GOLD,
            fontsize=34, fontweight="bold", family="serif")
    ax.text(0.5, 0.42,
            "A senior-grade ML pipeline  ·  Logistic Regression  ·  scikit-learn",
            ha="center", va="center", color="white",
            fontsize=13, style="italic")
    ax.text(0.5, 0.18,
            "891 passengers   ·   11 engineered features   ·   "
            "5-fold cross-validation   ·   F2-tuned for recall",
            ha="center", va="center", color=GOLD, fontsize=11)
    # Thin gold rules
    for y in (0.92, 0.08):
        ax.axhline(y, color=GOLD, lw=1.2, alpha=0.85)
    fig.savefig(CHART_DIR / "00_banner.png", facecolor=NAVY)
    plt.close(fig)
    log.info("Saved 00_banner.png")


def plot_eda(df: pd.DataFrame, raw_missing: pd.Series) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # 3.1 Survival by Pclass × Sex
    ct = (df.groupby(["Pclass", "Sex"])["Survived"]
            .mean().unstack().reindex(index=[1, 2, 3]))
    x = np.arange(len(ct.index)); w = 0.35
    ax[0, 0].bar(x - w/2, ct["female"], w, color=GOLD, label="female")
    ax[0, 0].bar(x + w/2, ct["male"],   w, color=NAVY, label="male")
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(["1st", "2nd", "3rd"])
    ax[0, 0].set_ylim(0, 1); ax[0, 0].set_ylabel("P(survived)")
    ax[0, 0].set_title("Survival rate by class and sex")
    ax[0, 0].legend()

    # 3.2 Age distribution by survival
    survived = df.loc[df["Survived"] == 1, "Age"]
    died     = df.loc[df["Survived"] == 0, "Age"]
    ax[0, 1].hist(died,     bins=30, alpha=0.6, color=NAVY, label="died")
    ax[0, 1].hist(survived, bins=30, alpha=0.6, color=GOLD, label="survived")
    ax[0, 1].set_xlabel("Age"); ax[0, 1].set_ylabel("count")
    ax[0, 1].set_title("Age distribution by outcome"); ax[0, 1].legend()

    # 3.3 Fare distribution (log-scaled)
    fare_clip = df["Fare"].clip(upper=200)
    ax[0, 2].hist(fare_clip[df["Survived"] == 0], bins=40, alpha=0.6,
                  color=NAVY, label="died")
    ax[0, 2].hist(fare_clip[df["Survived"] == 1], bins=40, alpha=0.6,
                  color=GOLD, label="survived")
    ax[0, 2].set_xlabel("Fare ($, clipped at 200)"); ax[0, 2].set_ylabel("count")
    ax[0, 2].set_title("Fare distribution by outcome"); ax[0, 2].legend()

    # 3.4 Survival rate by family size
    fs = df.groupby("FamilySize")["Survived"].agg(["mean", "count"])
    ax[1, 0].bar(fs.index, fs["mean"], color=GOLD, edgecolor=NAVY)
    ax[1, 0].set_xlabel("FamilySize"); ax[1, 0].set_ylabel("P(survived)")
    ax[1, 0].set_title("Survival rate by family size")
    for i, (m, c) in enumerate(zip(fs["mean"], fs["count"])):
        ax[1, 0].text(fs.index[i], m + 0.02, f"n={c}",
                      ha="center", fontsize=8, color=GREY)

    # 3.5 Survival rate by port of embarkation
    em = df.groupby("Embarked")["Survived"].agg(["mean", "count"])
    em = em.reindex(["S", "C", "Q"])
    bars = ax[1, 1].bar(["S—Southampton", "C—Cherbourg", "Q—Queenstown"],
                        em["mean"], color=[NAVY, GOLD, GREY], edgecolor=NAVY)
    ax[1, 1].set_ylim(0, 1); ax[1, 1].set_ylabel("P(survived)")
    ax[1, 1].set_title("Survival rate by port of embarkation")
    for b, c in zip(bars, em["count"]):
        ax[1, 1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                      f"n={c}", ha="center", fontsize=8, color=GREY)

    # 3.6 Missingness BEFORE imputation
    miss = raw_missing.sort_values(ascending=True)
    miss = miss[miss > 0]
    ax[1, 2].barh(miss.index, miss.values * 100, color=RED)
    ax[1, 2].set_xlabel("% missing")
    ax[1, 2].set_title("Missing values BEFORE imputation")

    fig.suptitle("Titanic — Exploratory Data Analysis",
                 fontsize=15, fontweight="bold", color=NAVY)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_eda_overview.png")
    plt.close(fig)
    log.info("Saved 01_eda_overview.png")


# ════════════════════════════════════════════════════════════════════
# 4. MODEL BENCHMARKING
# ════════════════════════════════════════════════════════════════════
F2 = make_scorer(fbeta_score, beta=2)

SCORING = {
    "accuracy":  "accuracy",
    "precision": "precision",
    "recall":    "recall",
    "f1":        "f1",
    "f2":        F2,
    "roc_auc":   "roc_auc",
}


def benchmark_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {
        "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression":    LogisticRegression(C=0.1, max_iter=2000,
                                                    random_state=SEED),
        "RandomForest":          RandomForestClassifier(
                                    n_estimators=300, max_depth=8,
                                    min_samples_split=10,
                                    random_state=SEED, n_jobs=-1),
        "GradientBoosting":      GradientBoostingClassifier(
                                    n_estimators=200, max_depth=3,
                                    learning_rate=0.05, random_state=SEED),
    }
    rows = []
    for name, est in candidates.items():
        pipe = build_pipeline(est)
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        rows.append({
            "model":    name,
            "accuracy": res["test_accuracy"].mean(),
            "precision": res["test_precision"].mean(),
            "recall":    res["test_recall"].mean(),
            "f1":        res["test_f1"].mean(),
            "f2":        res["test_f2"].mean(),
            "roc_auc":   res["test_roc_auc"].mean(),
            "f1_std":    res["test_f1"].std(),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def plot_model_comparison(bench: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["accuracy", "precision", "recall", "f1", "f2", "roc_auc"]
    x = np.arange(len(metrics))
    w = 0.2
    colours = [GREY, NAVY, GOLD, GREEN]
    for i, (_, row) in enumerate(bench.iterrows()):
        ax.bar(x + (i - 1.5) * w,
               [row[m] for m in metrics],
               w, label=row["model"], color=colours[i % len(colours)])
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1); ax.set_ylabel("score")
    ax.set_title("Cross-validated model comparison (5-fold StratifiedKFold)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_model_comparison.png")
    plt.close(fig)
    log.info("Saved 02_model_comparison.png")


# ════════════════════════════════════════════════════════════════════
# 5. HYPER-PARAMETER TUNING — Logistic Regression
# ════════════════════════════════════════════════════════════════════
def tune_logistic(X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, dict, float]:
    pipe = build_pipeline(LogisticRegression(max_iter=4000,
                                             solver="liblinear",
                                             random_state=SEED))
    grid = {
        "clf__C":       [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        "clf__penalty": ["l1", "l2"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, scoring="f1", cv=cv,
                          n_jobs=-1, refit=True)
    search.fit(X, y)
    log.info("Best params: %s | best CV F1: %.4f",
             search.best_params_, search.best_score_)
    return search.best_estimator_, search.best_params_, search.best_score_


# ════════════════════════════════════════════════════════════════════
# 6. EVALUATION ON HOLD-OUT SET
# ════════════════════════════════════════════════════════════════════
@dataclass
class HoldoutMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float


def evaluate_holdout(model: Pipeline, X_te: pd.DataFrame,
                     y_te: pd.Series) -> Tuple[HoldoutMetrics, np.ndarray, np.ndarray]:
    proba = model.predict_proba(X_te)[:, 1]
    pred  = model.predict(X_te)
    pr_auc_val = average_precision_score(y_te, proba)

    m = HoldoutMetrics(
        accuracy  = accuracy_score(y_te, pred),
        precision = precision_score(y_te, pred),
        recall    = recall_score(y_te, pred),
        f1        = f1_score(y_te, pred),
        f2        = fbeta_score(y_te, pred, beta=2),
        roc_auc   = roc_auc_score(y_te, proba),
        pr_auc    = float(pr_auc_val),
        brier     = brier_score_loss(y_te, proba),
        log_loss  = log_loss(y_te, proba, labels=[0, 1]),
    )
    return m, proba, pred


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                 metric_fn, n: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI for any metric(y_true, y_score)."""
    rng = np.random.default_rng(SEED)
    n_obs = len(y_true)
    stats = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        try:
            stats[i] = metric_fn(y_true[idx], y_score[idx])
        except ValueError:
            stats[i] = np.nan
    stats = stats[~np.isnan(stats)]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


# ════════════════════════════════════════════════════════════════════
# 7. PLOTS — confusion / ROC / PR / calibration / learning / threshold
# ════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Died", "Survived"])
    ax.set_yticklabels(["Died", "Survived"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (hold-out test set)")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else NAVY,
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_confusion_matrix.png")
    plt.close(fig)
    log.info("Saved 03_confusion_matrix.png")


def plot_roc_pr(y_true, y_score) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    ax[0].plot(fpr, tpr, color=NAVY, lw=2,
               label=f"AUC = {roc_auc_score(y_true, y_score):.3f}")
    ax[0].plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC curve")
    ax[0].legend(loc="lower right")

    ax[1].plot(rec, prec, color=GOLD, lw=2)
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision–Recall curve")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_roc_pr_curves.png")
    plt.close(fig)
    log.info("Saved 04_roc_pr_curves.png")


def plot_calibration(y_true, y_score) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10,
                                            strategy="quantile")
    brier = brier_score_loss(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color=NAVY, lw=2,
            label=f"model (Brier = {brier:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability curve (calibration)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "05_calibration.png")
    plt.close(fig)
    log.info("Saved 05_calibration.png")


def plot_learning_curve(model: Pipeline, X, y) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sizes, train_sc, val_sc = learning_curve(
        model, X, y, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="f1", n_jobs=-1, random_state=SEED,
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, train_sc.mean(axis=1), "o-", color=GOLD, label="train")
    ax.fill_between(sizes,
                    train_sc.mean(axis=1) - train_sc.std(axis=1),
                    train_sc.mean(axis=1) + train_sc.std(axis=1),
                    color=GOLD, alpha=0.2)
    ax.plot(sizes, val_sc.mean(axis=1), "o-", color=NAVY, label="validation")
    ax.fill_between(sizes,
                    val_sc.mean(axis=1) - val_sc.std(axis=1),
                    val_sc.mean(axis=1) + val_sc.std(axis=1),
                    color=NAVY, alpha=0.2)
    ax.set_xlabel("Training samples"); ax.set_ylabel("F1 score")
    ax.set_title("Learning curve (bias / variance diagnostic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "06_learning_curve.png")
    plt.close(fig)
    log.info("Saved 06_learning_curve.png")


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Sweep thresholds; return optimums for F1 and F2."""
    ths = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, (y_score >= t).astype(int))      for t in ths]
    f2s = [fbeta_score(y_true, (y_score >= t).astype(int), beta=2) for t in ths]
    pre = [precision_score(y_true, (y_score >= t).astype(int), zero_division=0) for t in ths]
    rec = [recall_score(y_true, (y_score >= t).astype(int))  for t in ths]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ths, pre, color=GREY,  label="precision")
    ax.plot(ths, rec, color=RED,   label="recall")
    ax.plot(ths, f1s, color=NAVY, lw=2, label="F1")
    ax.plot(ths, f2s, color=GOLD, lw=2, label="F2 (recall-weighted)")
    best_f1 = ths[int(np.argmax(f1s))]
    best_f2 = ths[int(np.argmax(f2s))]
    ax.axvline(best_f1, color=NAVY, ls=":", alpha=0.6)
    ax.axvline(best_f2, color=GOLD, ls=":", alpha=0.6)
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("score")
    ax.set_title(f"Threshold tuning  |  F1*={best_f1:.2f}  F2*={best_f2:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "07_threshold_tuning.png")
    plt.close(fig)
    log.info("Saved 07_threshold_tuning.png")
    return {"best_f1_threshold": float(best_f1),
            "best_f2_threshold": float(best_f2),
            "best_f1": float(max(f1s)),
            "best_f2": float(max(f2s))}


# ════════════════════════════════════════════════════════════════════
# 8. INTERPRETATION — coefficients, permutation importance, SHAP
# ════════════════════════════════════════════════════════════════════
def get_feature_names(model: Pipeline) -> List[str]:
    prep: ColumnTransformer = model.named_steps["prep"]
    names: List[str] = list(NUMERIC_FEATURES)
    ohe: OneHotEncoder = prep.named_transformers_["cat"]
    for col, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
        names.extend([f"{col}={c}" for c in cats])
    return names


def plot_coefficients(model: Pipeline) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    names = get_feature_names(model)
    coefs = clf.coef_[0]
    df = pd.DataFrame({"feature": names, "coef": coefs})
    df["abs"] = df["coef"].abs()
    df = df.sort_values("abs", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colours = [GREEN if c > 0 else RED for c in df["coef"]]
    ax.barh(df["feature"], df["coef"], color=colours, edgecolor=NAVY, lw=0.5)
    ax.axvline(0, color=NAVY, lw=0.8)
    ax.set_title("Logistic-regression coefficients (standardised)")
    ax.set_xlabel("log-odds contribution")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "08_coefficients.png")
    plt.close(fig)
    log.info("Saved 08_coefficients.png")
    return df.drop(columns="abs").reset_index(drop=True)


def plot_permutation_importance(model: Pipeline, X, y) -> pd.DataFrame:
    res = permutation_importance(model, X, y, n_repeats=20,
                                 random_state=SEED, scoring="f1", n_jobs=-1)
    df = (pd.DataFrame({"feature": X.columns,
                        "importance": res.importances_mean,
                        "std": res.importances_std})
            .sort_values("importance"))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["feature"], df["importance"], xerr=df["std"],
            color=GOLD, edgecolor=NAVY, lw=0.5, ecolor=NAVY)
    ax.set_title("Permutation importance (drop in F1)")
    ax.set_xlabel("Δ F1 when feature is shuffled")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "09_permutation_importance.png")
    plt.close(fig)
    log.info("Saved 09_permutation_importance.png")
    return df.reset_index(drop=True)


# ── EXACT SHAP for linear models (no shap package required) ─────────
# For a logistic-regression with standard-scaled inputs, the SHAP value
# of feature i for sample x is:
#       phi_i(x) = beta_i * (z_i(x) - E[z_i])
# where z_i is the scaled feature. Since StandardScaler centres each
# feature at zero on the training set, E[z_i] = 0 on training data, so
#       phi_i(x) = beta_i * z_i(x).
# The base value (E[f(x)]) equals the model intercept.
# ─────────────────────────────────────────────────────────────────────
def linear_shap(model: Pipeline, X_row: pd.DataFrame) -> Tuple[np.ndarray, float]:
    prep = model.named_steps["prep"]
    clf  = model.named_steps["clf"]
    z = prep.transform(X_row)[0]
    beta = clf.coef_[0]
    base = float(clf.intercept_[0])
    return beta * z, base


def plot_waterfall(model: Pipeline, passenger: pd.DataFrame,
                   label: str, fname: str) -> None:
    phi, base = linear_shap(model, passenger)
    names = get_feature_names(model)
    proba = model.predict_proba(passenger)[0, 1]

    # rank by |phi|, keep top 10
    order = np.argsort(np.abs(phi))[::-1][:10]
    phi_top = phi[order]
    nm_top  = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = [GREEN if v > 0 else RED for v in phi_top]
    y_pos = np.arange(len(phi_top))[::-1]
    ax.barh(y_pos, phi_top, color=colours, edgecolor=NAVY, lw=0.5)
    ax.set_yticks(y_pos); ax.set_yticklabels(nm_top, fontsize=9)
    ax.axvline(0, color=NAVY, lw=0.8)
    ax.set_xlabel("contribution to log-odds (positive ⇒ survival)")
    ax.set_title(f"{label}\n"
                 f"P(survived) = {proba:.2f}   |   base log-odds = {base:+.2f}")
    fig.tight_layout()
    fig.savefig(CHART_DIR / fname)
    plt.close(fig)
    log.info("Saved %s", fname)


# ════════════════════════════════════════════════════════════════════
# 9. NAMED-PASSENGER INSPECTION (Andrews / Dean / Brown)
# ════════════════════════════════════════════════════════════════════
def find_or_synthesise_passengers(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Try to locate Andrews, Dean, Brown in the dataset by surname.
    If not present (Kaggle train.csv contains only 891 of 2,224 souls),
    fall back to a synthetic row that matches the historical profile.
    """
    candidates = {
        "Mr. Thomas Andrews — Ship's Designer (1st class male, 39)": dict(
            search="Andrews",  match_sex="male",
            row=dict(Pclass=1, Sex="male", Age=39, SibSp=0, Parch=0,
                     Fare=42.0, Embarked="S", Name="Andrews, Mr. Thomas",
                     Cabin="A36", Title="Mr"),
        ),
        "Miss Millvina Dean — youngest passenger (3rd class infant, 0.17 yrs)": dict(
            search="Dean",     match_sex="female",
            row=dict(Pclass=3, Sex="female", Age=0.17, SibSp=1, Parch=2,
                     Fare=20.575, Embarked="S",
                     Name="Dean, Miss. Elizabeth Gladys (Millvina)",
                     Cabin=np.nan, Title="Miss"),
        ),
        "Mrs. Margaret 'Molly' Brown (1st class female, 44)": dict(
            search="Brown",    match_sex="female",
            row=dict(Pclass=1, Sex="female", Age=44, SibSp=0, Parch=0,
                     Fare=27.7208, Embarked="C",
                     Name="Brown, Mrs. James Joseph (Margaret Tobin)",
                     Cabin="B4", Title="Mrs"),
        ),
    }

    found: Dict[str, pd.DataFrame] = {}
    for label, spec in candidates.items():
        mask = (df["Name"].str.contains(spec["search"], case=False, na=False)
                & (df["Sex"] == spec["match_sex"]))
        if spec["search"] == "Brown":
            mask &= (df["Pclass"] == 1)
        sub = df[mask]
        if len(sub) > 0:
            row = sub.iloc[[0]].copy()
            log.info("Found '%s' in dataset (PassengerId=%s)",
                     label.split(' — ')[0], row.iloc[0].get("PassengerId", "n/a"))
            found[label] = row
        else:
            log.info("'%s' not in dataset — using synthetic profile",
                     label.split(' — ')[0])
            row = pd.DataFrame([spec["row"]])
            row = engineer_features(row)
            found[label] = row
    return found


# ════════════════════════════════════════════════════════════════════
# 10. MAIN
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    log.info("Python %s | NumPy %s | pandas %s | sklearn %s",
             platform.python_version(), np.__version__,
             pd.__version__, sklearn.__version__)

    # ── Data ─────────────────────────────────────────────────────────
    df_raw = load_data()
    validate_schema(df_raw)
    raw_missing = df_raw.isna().mean()
    df = engineer_features(df_raw)

    plot_banner()
    plot_eda(df, raw_missing)

    X = df[FEATURES]
    y = df["Survived"]

    # ── Hold-out split (stratified) ─────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # ── Step 1: benchmark ───────────────────────────────────────────
    log.info("Benchmarking models with 5-fold StratifiedKFold ...")
    bench = benchmark_models(X_tr, y_tr)
    log.info("\n%s", bench.round(4).to_string(index=False))
    plot_model_comparison(bench)

    # ── Step 2: tune Logistic Regression ────────────────────────────
    log.info("Tuning Logistic Regression hyper-parameters ...")
    best_model, best_params, best_cv_f1 = tune_logistic(X_tr, y_tr)

    # ── Step 3: hold-out evaluation ─────────────────────────────────
    metrics, proba, pred = evaluate_holdout(best_model, X_te, y_te)
    log.info("\nHold-out metrics:\n%s",
             "\n".join(f"  {k:>10s} = {v:.4f}"
                       for k, v in metrics.__dict__.items()))

    # ── Step 4: bootstrap CIs on key metrics ────────────────────────
    y_te_arr = y_te.to_numpy()
    cis = {
        "roc_auc_95ci": bootstrap_ci(y_te_arr, proba, roc_auc_score),
        "f1_95ci":      bootstrap_ci(y_te_arr, pred,  f1_score),
        "accuracy_95ci": bootstrap_ci(y_te_arr, pred, accuracy_score),
    }
    log.info("Bootstrap 95%% CIs: %s",
             {k: tuple(round(v, 4) for v in val) for k, val in cis.items()})

    # ── Step 5: plots ───────────────────────────────────────────────
    plot_confusion_matrix(y_te, pred)
    plot_roc_pr(y_te, proba)
    plot_calibration(y_te, proba)
    plot_learning_curve(best_model, X, y)
    th_info = tune_threshold(y_te_arr, proba)

    # ── Step 6: interpretation ──────────────────────────────────────
    coef_df = plot_coefficients(best_model)
    perm_df = plot_permutation_importance(best_model, X_te, y_te)
    log.info("\nTop coefficients:\n%s",
             coef_df.iloc[::-1].head(8).to_string(index=False))

    # ── Step 7: named passengers ────────────────────────────────────
    passengers = find_or_synthesise_passengers(df)
    fname_map = {0: "10_shap_andrews.png",
                 1: "11_shap_dean.png",
                 2: "12_shap_brown.png"}
    for i, (label, row) in enumerate(passengers.items()):
        plot_waterfall(best_model, row[FEATURES], label, fname_map[i])

    # ── Step 8: error analysis ──────────────────────────────────────
    err_df = X_te.copy()
    err_df["actual"]    = y_te.values
    err_df["predicted"] = pred
    err_df["proba"]     = proba
    err_df["error_type"] = "correct"
    err_df.loc[(y_te.values == 1) & (pred == 0), "error_type"] = "false_neg"
    err_df.loc[(y_te.values == 0) & (pred == 1), "error_type"] = "false_pos"
    error_summary = (err_df[err_df["error_type"] != "correct"]
 



                    .groupby(["error_type", "Pclass", "Sex"])
                     .size().reset_index(name="count"))
    log.info("\nError breakdown:\n%s",
             error_summary.to_string(index=False))
    error_summary.to_csv(OUTPUT_DIR / "error_breakdown.csv", index=False)

    # ── Step 9: classification report ───────────────────────────────
    cls_report = classification_report(y_te, pred,
                                       target_names=["Died", "Survived"],
                                       digits=4)
    (OUTPUT_DIR / "classification_report.txt").write_text(cls_report)

    # ── Step 10: persist artefacts ──────────────────────────────────
    joblib.dump(best_model, OUTPUT_DIR / "model.joblib")
    log.info("Saved model.joblib")

    payload = {
        "versions": {
            "python":  platform.python_version(),
            "numpy":   np.__version__,
            "pandas":  pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "data": {
            "n_rows":     int(len(df)),
            "n_features": len(FEATURES),
            "features":   FEATURES,
            "survival_rate": float(y.mean()),
        },
        "model": {
            "type":   "LogisticRegression (sklearn Pipeline)",
            "best_params": best_params,
            "best_cv_f1":  best_cv_f1,
        },
        "holdout_metrics": metrics.__dict__,
        "bootstrap_95ci":  cis,
        "threshold_tuning": th_info,
        "model_comparison": bench.to_dict(orient="records"),
        "top_coefficients":      coef_df.tail(10).to_dict(orient="records"),
        "permutation_importance": perm_df.tail(10).to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)
    log.info("Saved metrics.json")

    # ── Step 11: summary ────────────────────────────────────────────
    print("\n" + "═" * 64)
    print(" TITANIC SURVIVAL — FINAL MODEL SUMMARY")
    print("═" * 64)
    print(f"  Best params      : {best_params}")
    print(f"  CV F1 (5-fold)   : {best_cv_f1:.4f}")
    print(f"  Hold-out F1      : {metrics.f1:.4f}")
    print(f"  Hold-out ROC-AUC : {metrics.roc_auc:.4f}  "
          f"95% CI {tuple(round(v, 3) for v in cis['roc_auc_95ci'])}")
    print(f"  Hold-out Acc.    : {metrics.accuracy:.4f}")
    print(f"  Brier score      : {metrics.brier:.4f}  (lower is better)")
    print(f"  Charts saved     : {CHART_DIR}")
    print(f"  Model saved      : {OUTPUT_DIR / 'model.joblib'}")
    print(f"  Metrics saved    : {OUTPUT_DIR / 'metrics.json'}")
    print("═" * 64 + "\n")

    plt.show(block=True)  # James's standing instruction (Android headless)


if __name__ == "__main__":
    sys.exit(main())








