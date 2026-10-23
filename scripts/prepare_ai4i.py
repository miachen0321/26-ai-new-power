"""Prepare the UCI AI4I 2020 Predictive Maintenance dataset.

Outputs:
    data/ai4i2020.csv  Clean dataset, BOM stripped, validated columns.

Run from the project root:
    python3 scripts/prepare_ai4i.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "ai4i2020.zip"
CSV_PATH = DATA_DIR / "ai4i2020.csv"

NUMERIC_COLUMNS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]
FLAG_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]
COLUMNS = [
    "udi",
    "product_id",
    "type",
    *NUMERIC_COLUMNS,
    "machine_failure",
    *FLAG_COLUMNS,
]


def download(url: str, destination: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
        print(f"Downloaded {url} -> {destination}")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Warning: could not download {url}: {exc}", file=sys.stderr)
        raise


def load_frame(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            "data/ai4i2020.csv is missing; run the download step first"
        )
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    frame.columns = COLUMNS
    frame["type"] = frame["type"].str.strip()
    return frame


def quality_report(frame: pd.DataFrame) -> dict:
    valid_product_ids = frame["product_id"].astype(str).str.fullmatch(r"[LMH]\d{5}")
    return {
        "rows": len(frame),
        "columns": frame.shape[1],
        "missing_values": int(frame[NUMERIC_COLUMNS + FLAG_COLUMNS + ["machine_failure"]].isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated(subset=COLUMNS).sum()),
        "invalid_product_ids": int((~valid_product_ids).sum()),
    }


def validate(frame: pd.DataFrame) -> None:
    if not set(COLUMNS).issubset(frame.columns):
        raise ValueError("Unexpected columns in ai4i2020.csv")
    if frame[NUMERIC_COLUMNS + FLAG_COLUMNS + ["machine_failure"]].isna().any().any():
        raise ValueError("Missing values found in ai4i2020 numeric/flag columns")
    if not frame["machine_failure"].isin([0, 1]).all():
        raise ValueError("machine_failure must be binary")
    if not frame["type"].isin(["L", "M", "H"]).all():
        raise ValueError("type must be one of L, M, H")
    if not frame["product_id"].astype(str).str.fullmatch(r"[LMH]\d{5}").all():
        raise ValueError("product_id must follow the [LMH]XXXXX format")


def consistency_flags(frame: pd.DataFrame) -> pd.Series:
    flags = frame[FLAG_COLUMNS].sum(axis=1)
    return (flags > 0).astype(int) != frame["machine_failure"].astype(int)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        download(DATA_URL, ZIP_PATH)
        import zipfile

        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(DATA_DIR)

    frame = load_frame(CSV_PATH)
    validate(frame)
    report = quality_report(frame)
    print(f"Quality report: {report}")
    inconsistent = consistency_flags(frame)
    inconsistent.to_csv(DATA_DIR / "ai4i2020_flag_inconsistency.csv", index=False, header=["inconsistent"])
    frame.to_csv(CSV_PATH, index=False)

    print(f"Saved {len(frame)} rows x {frame.shape[1]} columns to {CSV_PATH}")
    print(frame["machine_failure"].value_counts().to_string())
    print(f"Flag/machine_failure inconsistencies: {int(inconsistent.sum())}")


if __name__ == "__main__":
    main()
