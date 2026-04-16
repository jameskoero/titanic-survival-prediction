"""Model evaluation and reporting for Titanic survival prediction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from preprocess import ROOT_DIR, add_engineered_features, load_raw_data, prepare_dataset

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logger for evaluation script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Titanic survival model.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=ROOT_DIR / "data" / "raw" / "train.csv",
        help="Path to raw Titanic CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT_DIR / "outputs" / "model.joblib",
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--metrics-input",
        type=Path,
        default=ROOT_DIR / "outputs" / "metrics.json",
        help="Path to training metrics JSON file.",
    )
    parser.add_argument(
        "--confusion-matrix-output",
        type=Path,
        default=ROOT_DIR / "outputs" / "confusion_matrix.png",
        help="Path to save confusion matrix plot.",
    )
    parser.add_argument(
        "--roc-output",
        type=Path,
        default=ROOT_DIR / "outputs" / "roc_curve.png",
        help="Path to save ROC curve plot.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT_DIR / "report" / "titanic_report.pdf",
        help="Path to save PDF report.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_artifacts(model_path: Path, metrics_input: Path) -> Tuple[Dict, Dict]:
    """Load saved model artifact and training metrics.

    Args:
        model_path: Path to model artifact.
        metrics_input: Path to metrics JSON.

    Returns:
        Tuple of (artifact dictionary, metrics dictionary).
    """
    artifact = joblib.load(model_path)
    with metrics_input.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    return artifact, metrics


def save_confusion_matrix_plot(cm: np.ndarray, output_path: Path) -> None:
    """Create and save confusion matrix visualization.

    Args:
        cm: Confusion matrix array.
        output_path: Destination image path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    labels = ["Did Not Survive", "Survived"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > cm.max() / 2 else "black"
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color=color)

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve_plot(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_path: Path) -> None:
    """Create and save ROC curve visualization.

    Args:
        fpr: False positive rate values.
        tpr: True positive rate values.
        roc_auc: Area under ROC curve.
        output_path: Destination image path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
    ax.set_title("Receiver Operating Characteristic")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_report_pdf(
    report_output: Path,
    metrics_summary: Dict[str, float],
    training_metrics: Dict,
    dataset_insights: Dict[str, float],
    confusion_matrix_output: Path,
    roc_output: Path,
) -> None:
    """Generate concise professional PDF report.

    Args:
        report_output: Destination report path.
        metrics_summary: Primary evaluation metrics.
        training_metrics: Training metrics for model comparison.
        dataset_insights: EDA-style summary insights.
        confusion_matrix_output: Path to confusion matrix image.
        roc_output: Path to ROC image.
    """
    report_output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(report_output) as pdf:
        fig1 = plt.figure(figsize=(8.27, 11.69))
        fig1.text(0.08, 0.95, "Titanic Survival Prediction Report", fontsize=18, weight="bold")
        fig1.text(0.08, 0.90, "1. Introduction", fontsize=13, weight="bold")
        fig1.text(
            0.08,
            0.84,
            "This report summarizes data analysis and predictive modeling for the Kaggle Titanic dataset,\n"
            "aimed at estimating passenger survival probability.",
            fontsize=10,
        )
        fig1.text(0.08, 0.76, "2. Methodology", fontsize=13, weight="bold")
        fig1.text(
            0.08,
            0.66,
            "Feature engineering: Title extraction, FamilySize, IsAlone\n"
            "Preprocessing: median/mode imputation, one-hot encoding, standard scaling\n"
            "Models: Logistic Regression and Random Forest with ROC-AUC comparison\n"
            "Validation: train/test split and cross-validation",
            fontsize=10,
        )
        fig1.text(0.08, 0.55, "3. EDA Insights", fontsize=13, weight="bold")
        fig1.text(
            0.08,
            0.45,
            f"Total passengers: {dataset_insights['passenger_count']:.0f}\n"
            f"Overall survival rate: {dataset_insights['survival_rate']:.2%}\n"
            f"Female survival rate: {dataset_insights['female_survival_rate']:.2%}\n"
            f"Male survival rate: {dataset_insights['male_survival_rate']:.2%}\n"
            f"Average age: {dataset_insights['average_age']:.2f}",
            fontsize=10,
        )
        fig1.text(0.08, 0.34, "4. Conclusion", fontsize=13, weight="bold")
        fig1.text(
            0.08,
            0.27,
            "The selected model provides strong discrimination performance and a practical baseline for\n"
            "binary survival prediction, with opportunities for feature enrichment and ensemble tuning.",
            fontsize=10,
        )
        fig1.axis("off")
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.text(0.08, 0.95, "Model Performance", fontsize=16, weight="bold")
        fig2.text(
            0.08,
            0.86,
            f"Accuracy: {metrics_summary['accuracy']:.4f}\n"
            f"ROC-AUC: {metrics_summary['roc_auc']:.4f}\n"
            f"Selected model: {metrics_summary['model_name']}",
            fontsize=11,
        )
        fig2.text(0.08, 0.76, "Training Comparison Metrics", fontsize=13, weight="bold")
        fig2.text(0.08, 0.62, json.dumps(training_metrics, indent=2), fontsize=9, family="monospace")
        fig2.axis("off")
        pdf.savefig(fig2)
        plt.close(fig2)

        image_fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
        image_fig.suptitle("Evaluation Visuals", fontsize=16, weight="bold")
        axes[0].imshow(plt.imread(confusion_matrix_output))
        axes[0].axis("off")
        axes[0].set_title("Confusion Matrix")

        axes[1].imshow(plt.imread(roc_output))
        axes[1].axis("off")
        axes[1].set_title("ROC Curve")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(image_fig)
        plt.close(image_fig)


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    """Evaluate model performance and generate visual/report artifacts.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Dictionary containing summary metrics.
    """
    artifact, training_metrics = load_artifacts(args.model_path, args.metrics_input)

    raw_df = load_raw_data(args.raw_data)
    processed_df = add_engineered_features(raw_df)
    x, y = prepare_dataset(processed_df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    preprocessor = artifact["preprocessor"]
    model = artifact["model"]
    x_test_t = preprocessor.transform(x_test)

    y_pred = model.predict(x_test_t)
    y_prob = model.predict_proba(x_test_t)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    save_confusion_matrix_plot(cm, args.confusion_matrix_output)
    save_roc_curve_plot(fpr, tpr, roc_auc, args.roc_output)

    dataset_insights = {
        "passenger_count": float(len(processed_df)),
        "survival_rate": float(processed_df["Survived"].mean()),
        "female_survival_rate": float(
            processed_df.loc[processed_df["Sex"] == "female", "Survived"].mean()
        ),
        "male_survival_rate": float(
            processed_df.loc[processed_df["Sex"] == "male", "Survived"].mean()
        ),
        "average_age": float(processed_df["Age"].mean(skipna=True)),
    }

    summary = {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "model_name": artifact["model_name"],
    }

    generate_report_pdf(
        args.report_output,
        summary,
        training_metrics,
        dataset_insights,
        args.confusion_matrix_output,
        args.roc_output,
    )

    LOGGER.info("Evaluation complete. Accuracy=%.4f ROC-AUC=%.4f", accuracy, roc_auc)
    LOGGER.info("Confusion matrix saved to %s", args.confusion_matrix_output)
    LOGGER.info("ROC curve saved to %s", args.roc_output)
    LOGGER.info("PDF report saved to %s", args.report_output)
    return summary


def main() -> None:
    """Entry point for evaluation script."""
    configure_logging()
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
