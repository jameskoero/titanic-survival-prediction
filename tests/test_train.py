from __future__ import annotations

import argparse
import json

import joblib

from src.train import train_models


def test_train_models_saves_artifacts(raw_data_csv, tmp_path):
    model_output = tmp_path / "model.joblib"
    metrics_output = tmp_path / "metrics.json"

    args = argparse.Namespace(
        raw_data=raw_data_csv,
        processed_data=tmp_path / "processed_train.csv",
        model_output=model_output,
        metrics_output=metrics_output,
        test_size=0.25,
        random_state=42,
        cv_folds=2,
    )

    results = train_models(args)

    assert set(results.keys()) == {"logistic_regression", "random_forest"}
    assert model_output.exists()
    assert metrics_output.exists()

    artifact = joblib.load(model_output)
    assert {"model_name", "model", "preprocessor", "feature_columns", "train_config", "metrics"}.issubset(
        artifact.keys()
    )

    with metrics_output.open("r", encoding="utf-8") as fh:
        persisted_metrics = json.load(fh)

    assert set(persisted_metrics.keys()) == set(results.keys())
    for model_metrics in persisted_metrics.values():
        assert 0.0 <= model_metrics["accuracy"] <= 1.0
        assert 0.0 <= model_metrics["roc_auc"] <= 1.0
        assert model_metrics["cv_roc_auc_std"] >= 0.0
