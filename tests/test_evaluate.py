from __future__ import annotations

import argparse

from src.evaluate import evaluate
from src.train import train_models


def test_evaluate_metrics_and_end_to_end_outputs(raw_data_csv, tmp_path):
    model_output = tmp_path / "model.joblib"
    metrics_output = tmp_path / "metrics.json"

    train_args = argparse.Namespace(
        raw_data=raw_data_csv,
        processed_data=tmp_path / "processed_train.csv",
        model_output=model_output,
        metrics_output=metrics_output,
        test_size=0.25,
        random_state=42,
        cv_folds=2,
    )
    train_models(train_args)

    eval_args = argparse.Namespace(
        raw_data=raw_data_csv,
        model_path=model_output,
        metrics_input=metrics_output,
        confusion_matrix_output=tmp_path / "confusion_matrix.png",
        roc_output=tmp_path / "roc_curve.png",
        report_output=tmp_path / "titanic_report.pdf",
        test_size=0.25,
        random_state=42,
    )

    summary = evaluate(eval_args)

    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["roc_auc"] <= 1.0
    assert summary["model_name"] in {"logistic_regression", "random_forest"}

    for artifact_path in (
        eval_args.confusion_matrix_output,
        eval_args.roc_output,
        eval_args.report_output,
    ):
        assert artifact_path.exists()
        assert artifact_path.stat().st_size > 0
