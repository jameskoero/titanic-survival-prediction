"""Data preprocessing utilities for Titanic survival prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "train.csv"
DEFAULT_PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_train.csv"

NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load Titanic raw dataset from disk.

    Args:
        data_path: Path to the raw CSV file. Defaults to data/raw/train.csv.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the dataset file is missing.
    """
    csv_path = Path(data_path) if data_path else DEFAULT_RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download Kaggle Titanic train.csv and place it in data/raw/."
        )
    LOGGER.info("Loading raw data from %s", csv_path)
    return pd.read_csv(csv_path)


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features used by the predictive model.

    Args:
        dataframe: Input Titanic DataFrame.

    Returns:
        DataFrame with engineered feature columns.
    """
    df = dataframe.copy()
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def prepare_dataset(
    dataframe: pd.DataFrame, target_column: str = "Survived"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare model features and target arrays.

    Args:
        dataframe: Input DataFrame containing features and target.
        target_column: Name of target column.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    feature_frame = dataframe[FEATURE_COLUMNS].copy()
    target = dataframe[target_column].copy()
    return feature_frame, target


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn preprocessing transformer for model training.

    Returns:
        A fitted-ready ColumnTransformer.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def save_processed_data(
    dataframe: pd.DataFrame, processed_path: Optional[Path] = None
) -> Path:
    """Persist processed DataFrame to data/processed directory.

    Args:
        dataframe: DataFrame to persist.
        processed_path: Optional output CSV path.

    Returns:
        Path where CSV was saved.
    """
    output_path = Path(processed_path) if processed_path else DEFAULT_PROCESSED_DATA_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    LOGGER.info("Processed data saved to %s", output_path)
    return output_path


def run_preprocessing(
    raw_data_path: Optional[Path] = None, processed_path: Optional[Path] = None
) -> pd.DataFrame:
    """Execute preprocessing and persist processed dataset.

    Args:
        raw_data_path: Optional path to raw CSV.
        processed_path: Optional destination for processed CSV.

    Returns:
        Processed DataFrame with engineered features.
    """
    raw_df = load_raw_data(raw_data_path)
    processed_df = add_engineered_features(raw_df)
    save_processed_data(processed_df, processed_path)
    return processed_df
