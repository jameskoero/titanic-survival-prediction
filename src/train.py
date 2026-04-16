from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Titanic survival model")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/train_processed.csv"),
        help="Path to processed CSV",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("outputs/model.joblib"),
        help="Path to save trained model",
    )
    parser.add_argument(
        "--scaler-output",
        type=Path,
        default=Path("outputs/scaler.joblib"),
        help="Path to save fitted scaler",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("outputs/train_metrics.json"),
        help="Path to save training metrics",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proportion of dataset to use for testing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for train/test split and model training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=args.random_state)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    joblib.dump(scaler, args.scaler_output)

    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {args.model_output}")
    print(f"Scaler saved to {args.scaler_output}")
    print(f"Training metrics: {metrics}")


if __name__ == "__main__":
    main()
