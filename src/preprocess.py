from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
from constants import FEATURE_COLUMNS, TARGET_COLUMN

TITLE_MAP = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
EMBARKED_MAP = {"S": 0, "C": 1, "Q": 2}
SEX_MAP = {"male": 0, "female": 1}


def _extract_title(name: str) -> str:
    title = pd.Series([name]).str.extract(r" ([A-Za-z]+)\.", expand=False).iloc[0]
    if pd.isna(title):
        return "Rare"

    replacements: Dict[str, str] = {
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    return replacements.get(title, title)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()

    processed["Title"] = processed["Name"].fillna("").map(_extract_title)
    processed["FamilySize"] = processed["SibSp"].fillna(0) + processed["Parch"].fillna(0) + 1
    processed["IsAlone"] = (processed["FamilySize"] == 1).astype(int)

    processed["Age"] = processed["Age"].fillna(processed["Age"].median())
    processed["Embarked"] = processed["Embarked"].fillna(processed["Embarked"].mode().iloc[0])
    processed["Fare"] = processed["Fare"].fillna(processed["Fare"].median())

    processed["Sex"] = processed["Sex"].map(SEX_MAP).fillna(0).astype(int)
    processed["Embarked"] = processed["Embarked"].map(EMBARKED_MAP).fillna(0).astype(int)
    processed["Title"] = processed["Title"].map(TITLE_MAP).fillna(0).astype(int)

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [c for c in required_cols if c not in processed.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after preprocessing: {missing_cols}")

    return processed[required_cols]


def preprocess_file(input_csv: Path, output_csv: Path) -> Path:
    df = pd.read_csv(input_csv)
    processed = preprocess_dataframe(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_csv, index=False)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    parser.add_argument("--input", type=Path, default=Path("data/raw/train.csv"), help="Path to raw CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/train_processed.csv"),
        help="Path to processed CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = preprocess_file(args.input, args.output)
    print(f"Processed dataset written to {output_path}")


if __name__ == "__main__":
    main()
