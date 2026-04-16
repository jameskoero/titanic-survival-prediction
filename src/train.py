"""Training script for Titanic survival prediction models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from preprocess import (
    FEATURE_COLUMNS,
    ROOT_DIR,
    build_preprocessor,
    prepare_dataset,
    run_preprocessing,
)

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure training logger settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model training."""
    parser = argparse.ArgumentParser(description="Train Titanic survival prediction models.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=ROOT_DIR / "data" / "raw" / "train.csv",
        help="Path to raw training CSV file.",
    )
    parser.add_argument(
        "--processed-data",
        type=Path,
        default=ROOT_DIR / "data" / "processed" / "processed_train.csv",
        help="Path to save processed CSV file.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=ROOT_DIR / "outputs" / "model.joblib",
        help="Path to save trained model artifact.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT_DIR / "outputs" / "metrics.json",
        help="Path to save metrics JSON.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds.")
    return parser.parse_args()


def evaluate_model(model, x_test, y_test) -> Tuple[float, float]:
    """Evaluate a fitted model and return key metrics.

    Args:
        model: Fitted model instance.
        x_test: Transformed test features.
        y_test: Test labels.

    Returns:
        Tuple containing (accuracy, roc_auc).
    """
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    return accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_prob)


def train_models(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    """Train candidate models, compare metrics, and save best artifact.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Dictionary with model performance metrics.
    """
    LOGGER.info("Starting preprocessing")
    processed_df = run_preprocessing(args.raw_data, args.processed_data)
    x, y = prepare_dataset(processed_df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=args.random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=args.random_state,
            n_jobs=-1,
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    fitted_models = {}

    for model_name, model in candidates.items():
        LOGGER.info("Training %s", model_name)
        fitted_model = model.fit(x_train_t, y_train)
        accuracy, roc_auc = evaluate_model(fitted_model, x_test_t, y_test)

        cv_pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("model", clone(model))]
        )
        cv_scores = cross_val_score(
            cv_pipeline,
            x,
            y,
            cv=args.cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
        )

        results[model_name] = {
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "cv_roc_auc_mean": float(np.mean(cv_scores)),
            "cv_roc_auc_std": float(np.std(cv_scores)),
        }
        fitted_models[model_name] = fitted_model

    best_model_name = max(results, key=lambda key: results[key]["roc_auc"])
    best_model = fitted_models[best_model_name]

    artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_columns": FEATURE_COLUMNS,
        "train_config": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "cv_folds": args.cv_folds,
        },
        "metrics": results,
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_output)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2)

    LOGGER.info("Best model: %s", best_model_name)
    LOGGER.info("Model artifact saved to %s", args.model_output)
    LOGGER.info("Metrics saved to %s", args.metrics_output)

    return results


def main() -> None:
    """Run end-to-end training pipeline from CLI."""
    configure_logging()
    args = parse_args()
    metrics = train_models(args)
    LOGGER.info("Training complete. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
