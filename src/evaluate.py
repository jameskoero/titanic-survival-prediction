"""Model evaluation and reporting for Titanic survival prediction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from preprocess import TARGET_COLUMN

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEST_SPLIT_PATH = ROOT_DIR / "data" / "processed" / "test_split.csv"
DEFAULT_PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "titanic_processed.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "outputs" / "titanic_model.joblib"
DEFAULT_SCALER_PATH = ROOT_DIR / "outputs" / "scaler.joblib"
DEFAULT_METRICS_PATH = ROOT_DIR / "outputs" / "metrics.json"
DEFAULT_CM_PATH = ROOT_DIR / "outputs" / "confusion_matrix.png"
DEFAULT_ROC_PATH = ROOT_DIR / "outputs" / "roc_curve.png"
DEFAULT_REPORT_PATH = ROOT_DIR / "report" / "titanic_report.pdf"


def setup_logging() -> None:
    """Configure logging for evaluation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_artifacts(model_path: Path, scaler_path: Path) -> Tuple[dict, object | None]:
    """Load trained model and optional scaler."""
    model_bundle = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model_bundle, scaler


def evaluate_model(
    model_bundle: dict,
    scaler,
    test_df: pd.DataFrame,
) -> Dict[str, object]:
    """Run model evaluation and return metrics payload."""
    x_test = test_df[model_bundle["feature_columns"]]
    y_test = test_df[TARGET_COLUMN]

    if model_bundle["requires_scaling"]:
        if scaler is None:
            raise ValueError("Scaler artifact is missing for a scaling-required model.")
        x_test_input = scaler.transform(x_test)
    else:
        x_test_input = x_test

    model = model_bundle["model"]
    y_pred = model.predict(x_test_input)
    y_prob = model.predict_proba(x_test_input)[:, 1]

    metrics = {
        "model_name": model_bundle["model_name"],
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return metrics


def save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    """Generate and save confusion matrix visualization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Titanic Survival - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_roc_curve(y_true, y_prob, output_path: Path) -> None:
    """Generate and save ROC curve plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.set_style("whitegrid")
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Titanic Survival - ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def summarize_eda_insights(processed_df: pd.DataFrame) -> Dict[str, float]:
    """Compute compact EDA insights for the report."""
    insights = {
        "rows": int(len(processed_df)),
        "survival_rate": float(processed_df[TARGET_COLUMN].mean()),
        "female_survival_rate": float(
            processed_df.loc[processed_df["Sex"] == 1, TARGET_COLUMN].mean()
        ),
        "male_survival_rate": float(
            processed_df.loc[processed_df["Sex"] == 0, TARGET_COLUMN].mean()
        ),
        "first_class_survival_rate": float(
            processed_df.loc[processed_df["Pclass"] == 1, TARGET_COLUMN].mean()
        ),
    }
    return insights


def generate_pdf_report(
    report_path: Path,
    evaluation_metrics: Dict[str, object],
    insights: Dict[str, float],
) -> None:
    """Generate concise multi-page PDF report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(report_path) as pdf:
        fig1, ax1 = plt.subplots(figsize=(8.27, 11.69))
        ax1.axis("off")
        intro_text = (
            "Titanic Survival Prediction Report\n\n"
            "1. Introduction\n"
            "This project predicts passenger survival on the Titanic using supervised "
            "machine learning on Kaggle's Titanic dataset.\n\n"
            "2. Methodology\n"
            "- Data preprocessing: missing-value handling, categorical encoding, "
            "feature engineering\n"
            "- Models: Logistic Regression (baseline) with optional Random Forest\n"
            "- Evaluation: stratified holdout validation with Accuracy and ROC-AUC\n"
        )
        ax1.text(0.02, 0.98, intro_text, va="top", fontsize=11)
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(8.27, 11.69))
        ax2.axis("off")
        eda_text = (
            "3. EDA Insights\n"
            f"- Total records analyzed: {insights['rows']}\n"
            f"- Overall survival rate: {insights['survival_rate']:.3f}\n"
            f"- Female survival rate: {insights['female_survival_rate']:.3f}\n"
            f"- Male survival rate: {insights['male_survival_rate']:.3f}\n"
            f"- First-class survival rate: {insights['first_class_survival_rate']:.3f}\n\n"
            "4. Model Performance\n"
            f"- Selected model: {evaluation_metrics['model_name']}\n"
            f"- Accuracy: {evaluation_metrics['accuracy']:.3f}\n"
            f"- ROC-AUC: {evaluation_metrics['roc_auc']:.3f}\n"
        )
        ax2.text(0.02, 0.98, eda_text, va="top", fontsize=11)
        pdf.savefig(fig2)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(8.27, 11.69))
        ax3.axis("off")
        conclusion_text = (
            "5. Conclusion\n"
            "The pipeline delivers an interpretable baseline with strong discrimination "
            "for binary survival prediction.\n\n"
            "Next steps:\n"
            "- Hyperparameter optimization\n"
            "- Calibrated probability estimation\n"
            "- Automated experiment tracking and CI model checks\n"
        )
        ax3.text(0.02, 0.98, conclusion_text, va="top", fontsize=11)
        pdf.savefig(fig3)
        plt.close(fig3)


def persist_metrics(metrics_payload: Dict[str, object], metrics_path: Path) -> None:
    """Save evaluation metrics to JSON without raw arrays."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_to_store = {
        "model_name": metrics_payload["model_name"],
        "accuracy": metrics_payload["accuracy"],
        "roc_auc": metrics_payload["roc_auc"],
        "classification_report": metrics_payload["classification_report"],
    }
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_to_store, metrics_file, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation stage."""
    parser = argparse.ArgumentParser(description="Evaluate Titanic survival model")
    parser.add_argument("--test-split-path", type=Path, default=DEFAULT_TEST_SPLIT_PATH)
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--confusion-matrix-path", type=Path, default=DEFAULT_CM_PATH)
    parser.add_argument("--roc-curve-path", type=Path, default=DEFAULT_ROC_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run evaluation, plots generation, and report export."""
    setup_logging()
    args = parse_args()

    logging.info("Loading artifacts")
    model_bundle, scaler = load_artifacts(args.model_path, args.scaler_path)

    logging.info("Loading test split from %s", args.test_split_path)
    test_df = pd.read_csv(args.test_split_path)

    logging.info("Evaluating model")
    metrics_payload = evaluate_model(model_bundle, scaler, test_df)

    logging.info("Saving confusion matrix to %s", args.confusion_matrix_path)
    save_confusion_matrix(
        metrics_payload["y_true"],
        metrics_payload["y_pred"],
        args.confusion_matrix_path,
    )

    logging.info("Saving ROC curve to %s", args.roc_curve_path)
    save_roc_curve(metrics_payload["y_true"], metrics_payload["y_prob"], args.roc_curve_path)

    logging.info("Loading processed data for EDA insights")
    processed_df = pd.read_csv(args.processed_path)
    insights = summarize_eda_insights(processed_df)

    logging.info("Generating PDF report at %s", args.report_path)
    generate_pdf_report(args.report_path, metrics_payload, insights)

    logging.info("Persisting evaluation metrics to %s", args.metrics_path)
    persist_metrics(metrics_payload, args.metrics_path)

    logging.info(
        "Evaluation complete | Accuracy: %.4f | ROC-AUC: %.4f",
        metrics_payload["accuracy"],
        metrics_payload["roc_auc"],
    )


if __name__ == "__main__":
    main()
