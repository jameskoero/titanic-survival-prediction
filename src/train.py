"""Model training script for Titanic survival prediction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from preprocess import (
    DEFAULT_RAW_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_raw_dataset,
    preprocess_dataframe,
    scale_features,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "titanic_processed.csv"
DEFAULT_TRAIN_SPLIT_PATH = ROOT_DIR / "data" / "processed" / "train_split.csv"
DEFAULT_TEST_SPLIT_PATH = ROOT_DIR / "data" / "processed" / "test_split.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "outputs" / "titanic_model.joblib"
DEFAULT_SCALER_PATH = ROOT_DIR / "outputs" / "scaler.joblib"
DEFAULT_METRICS_PATH = ROOT_DIR / "outputs" / "metrics.json"


def setup_logging() -> None:
    """Configure logging for model training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def evaluate_candidate(
    model,
    x_test,
    y_test,
) -> Dict[str, float]:
    """Evaluate a fitted model on holdout data."""
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def train_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int,
    compare_models: bool,
    run_cv: bool,
    cv_folds: int,
) -> Tuple[dict, dict, StandardScaler | None]:
    """Train baseline and optional comparison models and pick best by ROC-AUC."""
    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)

    logistic_model = LogisticRegression(max_iter=1000, random_state=random_state)
    logistic_model.fit(x_train_scaled, y_train)
    logistic_metrics = evaluate_candidate(logistic_model, x_test_scaled, y_test)

    results = {
        "LogisticRegression": {
            "model": logistic_model,
            "metrics": logistic_metrics,
            "requires_scaling": True,
            "x_test": x_test_scaled,
        }
    }

    if run_cv:
        cv_scores = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=random_state),
            x_train_scaled,
            y_train,
            cv=cv_folds,
            scoring="roc_auc",
        )
        results["LogisticRegression"]["cv_roc_auc_mean"] = float(np.mean(cv_scores))

    if compare_models:
        random_forest_model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
        )
        random_forest_model.fit(x_train, y_train)
        random_forest_metrics = evaluate_candidate(random_forest_model, x_test, y_test)

        results["RandomForestClassifier"] = {
            "model": random_forest_model,
            "metrics": random_forest_metrics,
            "requires_scaling": False,
            "x_test": x_test,
        }

        if run_cv:
            rf_cv_scores = cross_val_score(
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    class_weight="balanced",
                ),
                x_train,
                y_train,
                cv=cv_folds,
                scoring="roc_auc",
            )
            results["RandomForestClassifier"]["cv_roc_auc_mean"] = float(
                np.mean(rf_cv_scores)
            )

    best_name = max(results, key=lambda name: results[name]["metrics"]["roc_auc"])
    best_entry = results[best_name]

    model_bundle = {
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "requires_scaling": best_entry["requires_scaling"],
        "model": best_entry["model"],
    }

    all_metrics = {
        "best_model": best_name,
        "models": {
            model_name: {
                key: value
                for key, value in model_info.items()
                if key not in {"model", "x_test"}
            }
            for model_name, model_info in results.items()
        },
    }

    selected_scaler = scaler if best_entry["requires_scaling"] else None
    return model_bundle, all_metrics, selected_scaler


def save_artifacts(
    model_bundle: dict,
    scaler,
    metrics: dict,
    model_path: Path,
    scaler_path: Path,
    metrics_path: Path,
) -> None:
    """Persist model, scaler, and metrics artifacts."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_bundle, model_path)
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
    elif scaler_path.exists():
        scaler_path.unlink()

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def save_splits(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    train_path: Path,
    test_path: Path,
) -> None:
    """Save train and test splits for downstream evaluation."""
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = x_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    test_df = x_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train Titanic survival models")
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--train-split-path", type=Path, default=DEFAULT_TRAIN_SPLIT_PATH)
    parser.add_argument("--test-split-path", type=Path, default=DEFAULT_TEST_SPLIT_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--compare-models", action="store_true")
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Run the model training pipeline."""
    setup_logging()
    args = parse_args()

    logging.info("Loading raw Titanic data from %s", args.raw_path)
    raw_df = load_raw_dataset(args.raw_path)

    logging.info("Preprocessing data")
    processed_df = preprocess_dataframe(raw_df)
    args.processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(args.processed_path, index=False)

    x_data = processed_df[FEATURE_COLUMNS]
    y_data = processed_df[TARGET_COLUMN]

    logging.info("Splitting data into train and test sets")
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_data,
    )

    save_splits(
        x_train,
        y_train,
        x_test,
        y_test,
        args.train_split_path,
        args.test_split_path,
    )

    logging.info("Training model(s)")
    model_bundle, metrics, scaler = train_models(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        random_state=args.random_state,
        compare_models=args.compare_models,
        run_cv=args.cross_validate,
        cv_folds=args.cv_folds,
    )

    logging.info("Saving model artifacts")
    save_artifacts(
        model_bundle=model_bundle,
        scaler=scaler,
        metrics=metrics,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        metrics_path=args.metrics_path,
    )

    logging.info("Best model: %s", metrics["best_model"])
    logging.info("Training complete")


if __name__ == "__main__":
    main()
