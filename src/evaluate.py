from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Titanic survival model")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/train_processed.csv"),
        help="Path to processed CSV",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("outputs/model.joblib"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=Path("outputs/scaler.joblib"),
        help="Path to fitted scaler",
    )
    parser.add_argument(
        "--confusion-matrix-output",
        type=Path,
        default=Path("outputs/confusion_matrix.png"),
        help="Path to save confusion matrix image",
    )
    parser.add_argument(
        "--roc-output",
        type=Path,
        default=Path("outputs/roc_curve.png"),
        help="Path to save ROC curve image",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("outputs/evaluation_metrics.json"),
        help="Path to save evaluation metrics",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proportion of dataset used for testing during training",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for train/test split during training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    model = joblib.load(args.model_path)
    scaler = joblib.load(args.scaler_path)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_value = auc(fpr, tpr)

    args.confusion_matrix_output.parent.mkdir(parents=True, exist_ok=True)

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("Titanic Survival Confusion Matrix")
    fig_cm.tight_layout()
    fig_cm.savefig(args.confusion_matrix_output, dpi=150)
    plt.close(fig_cm)

    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_value:.3f})", linewidth=2)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Titanic Survival ROC Curve")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(args.roc_output, dpi=150)
    plt.close(fig_roc)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Confusion matrix saved to {args.confusion_matrix_output}")
    print(f"ROC curve saved to {args.roc_output}")
    print(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()
