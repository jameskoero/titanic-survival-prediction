import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import titanic_model as tm  # noqa: E402


def build_sample_df(rows=30):
    data = []
    titles = ["Mr", "Mrs", "Miss", "Master", "Dr", "Ms", "Mme", "Mlle"]
    embarked_values = ["S", "C", "Q"]

    for i in range(rows):
        title = titles[i % len(titles)]
        data.append(
            {
                "PassengerId": i + 1,
                "Survived": i % 2,
                "Pclass": (i % 3) + 1,
                "Name": f"Doe, {title}. Passenger{i}",
                "Sex": "female" if i % 2 else "male",
                "Age": float(20 + (i % 30)),
                "SibSp": i % 3,
                "Parch": (i + 1) % 2,
                "Ticket": f"TKT-{i}",
                "Fare": float(10 + i),
                "Cabin": None,
                "Embarked": embarked_values[i % 3],
            }
        )

    return pd.DataFrame(data)


def test_engineer_features_creates_expected_columns():
    df = build_sample_df(6)
    engineered = tm.engineer_features(df)

    assert {"Title", "FamilySize", "IsAlone"}.issubset(engineered.columns)
    assert engineered.loc[0, "Title"] == "Mr"
    assert engineered.loc[4, "Title"] == "Rare"
    assert engineered.loc[5, "Title"] == "Miss"
    assert engineered.loc[0, "FamilySize"] == df.loc[0, "SibSp"] + df.loc[0, "Parch"] + 1


def test_fill_missing_values_imputes_age_embarked_fare():
    df = build_sample_df(10)
    df.loc[0, "Age"] = np.nan
    df.loc[1, "Embarked"] = np.nan
    df.loc[2, "Fare"] = np.nan

    filled = tm.fill_missing_values(df)

    assert filled["Age"].isna().sum() == 0
    assert filled["Embarked"].isna().sum() == 0
    assert filled["Fare"].isna().sum() == 0


def test_encode_categorical_maps_values_to_numeric():
    df = tm.engineer_features(build_sample_df(8))
    encoded = tm.encode_categorical(df)

    assert set(encoded["Sex"].unique()) <= {0, 1}
    assert set(encoded["Embarked"].unique()) <= {0, 1, 2}
    assert encoded["Title"].dtype.kind in "if"


def test_split_and_scale_outputs_expected_shapes():
    df = tm.encode_categorical(tm.fill_missing_values(tm.engineer_features(build_sample_df(30))))
    X, y = tm.prepare_features_target(df)

    X_train_sc, X_test_sc, y_train, y_test, _ = tm.split_and_scale(X, y)

    assert X_train_sc.shape[1] == len(tm.FEATURES)
    assert X_test_sc.shape[1] == len(tm.FEATURES)
    assert len(y_train) + len(y_test) == len(y)
    assert np.allclose(X_train_sc.mean(axis=0), 0, atol=1e-7)


def test_train_and_evaluate_model_returns_metrics():
    df = tm.encode_categorical(tm.fill_missing_values(tm.engineer_features(build_sample_df(30))))
    X, y = tm.prepare_features_target(df)
    X_train_sc, X_test_sc, y_train, y_test, _ = tm.split_and_scale(X, y)

    model = tm.train_model(X_train_sc, y_train)
    metrics = tm.evaluate_model(model, X_test_sc, y_test)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert metrics["confusion_matrix"].shape == (2, 2)
    assert isinstance(metrics["classification_report"], str)


def test_create_results_plot_saves_output_file(tmp_path):
    confusion = np.array([[4, 1], [2, 3]])
    coeffs = np.array([0.1, -0.2, 0.3, 0.4, -0.1, 0.5, -0.6, 0.2])
    output = tmp_path / "plot.png"

    tm.create_results_plot(confusion, tm.FEATURES, coeffs, output_path=str(output), show=False)

    assert output.exists()
    assert output.stat().st_size > 0


def test_run_pipeline_executes_end_to_end(tmp_path):
    csv_path = tmp_path / "train.csv"
    output = tmp_path / "results.png"
    build_sample_df(40).to_csv(csv_path, index=False)

    metrics = tm.run_pipeline(data_path=str(csv_path), output_plot=str(output), show_plot=False)

    assert output.exists()
    assert metrics["confusion_matrix"].shape == (2, 2)
    assert metrics["y_pred"].ndim == 1
