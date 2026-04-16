"""Data preprocessing utilities for Titanic survival prediction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = ROOT_DIR / "data" / "raw" / "train.csv"
DEFAULT_PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "titanic_processed.csv"

FEATURE_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "Title",
]
TARGET_COLUMN = "Survived"

TITLE_MAPPING = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
EMBARKED_MAPPING = {"S": 0, "C": 1, "Q": 2}
SEX_MAPPING = {"male": 0, "female": 1}



def setup_logging() -> None:
    """Configure logging for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )



def load_raw_dataset(raw_path: Path) -> pd.DataFrame:
    """Load the Titanic raw dataset from disk.

    Args:
        raw_path: Path to Kaggle Titanic train.csv.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError: If dataset file is not found.
    """
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. "
            "Download Kaggle Titanic train.csv and place it at data/raw/train.csv."
        )
    return pd.read_csv(raw_path)



def extract_title(name: str) -> str:
    """Extract and normalize title from passenger name."""
    title = pd.Series([name]).str.extract(r" ([A-Za-z]+)\\.", expand=False).iloc[0]

    if title in {
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
    }:
        return "Rare"
    if title in {"Mlle", "Ms"}:
        return "Miss"
    if title == "Mme":
        return "Mrs"
    return title



def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using robust defaults."""
    cleaned_df = df.copy()
    cleaned_df["Age"] = cleaned_df["Age"].fillna(cleaned_df["Age"].median())
    cleaned_df["Embarked"] = cleaned_df["Embarked"].fillna(cleaned_df["Embarked"].mode()[0])
    cleaned_df["Fare"] = cleaned_df["Fare"].fillna(cleaned_df["Fare"].median())
    return cleaned_df



def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns to numeric values."""
    encoded_df = df.copy()
    encoded_df["Sex"] = encoded_df["Sex"].map(SEX_MAPPING)
    encoded_df["Embarked"] = encoded_df["Embarked"].map(EMBARKED_MAPPING)
    encoded_df["Title"] = encoded_df["Title"].map(TITLE_MAPPING).fillna(0)
    return encoded_df



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features for the Titanic dataset."""
    featured_df = df.copy()
    featured_df["Title"] = featured_df["Name"].apply(extract_title)
    featured_df["FamilySize"] = featured_df["SibSp"] + featured_df["Parch"] + 1
    featured_df["IsAlone"] = (featured_df["FamilySize"] == 1).astype(int)
    return featured_df



def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run full preprocessing and return train-ready dataframe.

    Returns:
        Dataframe with selected feature columns and target column.
    """
    featured_df = engineer_features(df)
    cleaned_df = handle_missing_values(featured_df)
    encoded_df = encode_categorical_variables(cleaned_df)

    final_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    return encoded_df[final_columns].copy()



def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split processed dataframe into features and target."""
    features = df[FEATURE_COLUMNS].copy()
    target = df[TARGET_COLUMN].copy()
    return features, target



def scale_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale features for models sensitive to feature magnitude."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler



def save_processed_dataset(processed_df: pd.DataFrame, output_path: Path) -> None:
    """Save processed dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)



def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for preprocessing script."""
    parser = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Path to raw Titanic train.csv",
    )
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
        help="Path to save processed dataset",
    )
    return parser.parse_args()



def main() -> None:
    """Run preprocessing pipeline from CLI."""
    setup_logging()
    args = parse_args()

    logging.info("Loading raw data from %s", args.raw_path)
    raw_df = load_raw_dataset(args.raw_path)

    logging.info("Preprocessing dataframe")
    processed_df = preprocess_dataframe(raw_df)

    logging.info("Saving processed data to %s", args.processed_path)
    save_processed_dataset(processed_df, args.processed_path)
    logging.info("Preprocessing complete. Rows: %d", len(processed_df))


if __name__ == "__main__":
    main()
