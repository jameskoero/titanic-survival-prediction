from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

matplotlib.use("Agg")


@pytest.fixture
def synthetic_titanic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": list(range(1, 13)),
            "Survived": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "Pclass": [3, 1, 3, 1, 2, 3, 1, 3, 2, 3, 1, 2],
            "Name": [
                "Smith, Mr. John",
                "Doe, Mrs. Jane",
                "Taylor, Miss. Anna",
                "Brown, Master. Tom",
                "White, Dr. Alice",
                "Black, Mr. Sam",
                "Green, Mme. Eve",
                "Gray, Mlle. Sara",
                "King, Ms. Rita",
                "Hill, Rev. Mark",
                "Cole, Mr. Dan",
                "Ward, Mrs. Beth",
            ],
            "Sex": [
                "male",
                "female",
                "female",
                "male",
                "female",
                "male",
                "female",
                "female",
                "female",
                "male",
                "male",
                "female",
            ],
            "Age": [22, 38, np.nan, 4, 35, 28, 30, np.nan, 19, 45, 40, 27],
            "SibSp": [1, 1, 0, 3, 1, 0, 1, 0, 0, 0, 0, 1],
            "Parch": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            "Ticket": ["A/5 21171"] * 12,
            "Fare": [7.25, 71.28, 7.92, 53.1, np.nan, 8.05, 52.0, 11.13, 30.0, 13.0, 15.0, 41.57],
            "Cabin": [np.nan] * 12,
            "Embarked": ["S", "C", "S", "S", "C", "Q", np.nan, "S", "Q", "S", "S", "C"],
        }
    )


@pytest.fixture
def raw_data_csv(tmp_path: Path, synthetic_titanic_df: pd.DataFrame) -> Path:
    raw_path = tmp_path / "train.csv"
    synthetic_titanic_df.to_csv(raw_path, index=False)
    return raw_path
