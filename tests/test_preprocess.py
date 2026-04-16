from __future__ import annotations

import unittest

import numpy as np

from src.preprocess import FEATURE_COLUMNS, add_engineered_features, build_preprocessor, prepare_dataset


class TestFeatureEngineering(unittest.TestCase):
    def test_title_familysize_and_isalone(self) -> None:
        import pandas as pd

        input_df = pd.DataFrame(
            {
                "Name": ["A, Mr. One", "B, Mrs. Two", "C, Mlle. Three", "D, Dr. Four"],
                "SibSp": [0, 1, 0, 0],
                "Parch": [0, 1, 0, 0],
            }
        )

        processed = add_engineered_features(input_df)

        self.assertEqual(processed["Title"].tolist(), ["Mr", "Mrs", "Miss", "Rare"])
        self.assertEqual(processed["FamilySize"].tolist(), [1, 3, 1, 1])
        self.assertEqual(processed["IsAlone"].tolist(), [1, 0, 1, 1])


def test_prepare_dataset_columns(synthetic_titanic_df):
    processed = add_engineered_features(synthetic_titanic_df)
    x, y = prepare_dataset(processed)

    assert list(x.columns) == FEATURE_COLUMNS
    assert y.name == "Survived"
    assert len(x) == len(y)


def test_preprocessor_imputation_and_encoding(synthetic_titanic_df):
    processed = add_engineered_features(synthetic_titanic_df)
    x, _ = prepare_dataset(processed)
    transformer = build_preprocessor()

    transformed = transformer.fit_transform(x)
    transformed_array = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert np.isfinite(transformed_array).all()
    assert transformed_array.shape[1] > len(FEATURE_COLUMNS)
