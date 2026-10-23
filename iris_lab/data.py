"""Data loading and validation for the Iris dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
COLUMNS = FEATURES + ["species"]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "iris.csv"


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"missing required column(s): {missing_columns}")

    if frame.empty:
        raise ValueError("dataset is empty")

    for column in FEATURES:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"column {column!r} must be numeric")

    if frame[COLUMNS].isna().any().any():
        missing = frame[COLUMNS].isna().sum()
        raise ValueError(
            "missing values found: "
            + ", ".join(f"{column}={int(count)}" for column, count in missing.items() if count)
        )

    unknown = sorted(set(frame["species"]) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"unknown species label(s): {unknown}")

    return frame


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.drop(columns=[column for column in frame.columns if column.startswith("Unnamed:")])
    validate_data(frame)
    return frame[COLUMNS].reset_index(drop=True)


def iqr_outlier_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in FEATURES:
        q1, q3 = frame[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        counts[column] = int(((frame[column] < lower) | (frame[column] > upper)).sum())
    return counts


def data_quality_report(frame: pd.DataFrame) -> dict:
    validate_data(frame)
    return {
        "rows": int(len(frame)),
        "columns": int(len(COLUMNS)),
        "missing_values": int(frame[COLUMNS].isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "class_counts": {
            name: int(count)
            for name, count in frame["species"].value_counts().reindex(CLASS_NAMES).items()
        },
        "outliers": iqr_outlier_counts(frame),
    }
