"""Download and prepare the UCI Iris dataset.

Outputs:
    data/iris.csv   Clean dataset with English column names.
    data/iris.names Original UCI metadata (when available).

The script uses the UCI repository first and falls back to scikit-learn's
bundled copy if the network is unavailable.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "iris.data"
NAMES_PATH = DATA_DIR / "iris.names"
CSV_PATH = DATA_DIR / "iris.csv"

COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
SPECIES_MAP = {
    "Iris-setosa": "setosa",
    "Iris-versicolor": "versicolor",
    "Iris-virginica": "virginica",
}


def download(url: str, destination: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            destination.write_bytes(response.read())
        print(f"Downloaded {url} -> {destination}")
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Warning: could not download {url}: {exc}", file=sys.stderr)
        return False


def load_from_uci() -> pd.DataFrame:
    if not RAW_PATH.exists():
        if not download(DATA_URL, RAW_PATH):
            raise FileNotFoundError("UCI iris.data is unavailable")

    frame = pd.read_csv(RAW_PATH, header=None, names=COLUMNS)
    frame = frame.dropna(how="all").reset_index(drop=True)
    frame = frame[frame["species"].notna()].copy()
    frame["species"] = frame["species"].str.strip().map(SPECIES_MAP)
    if frame["species"].isna().any():
        unknown = sorted(set(frame.loc[frame["species"].isna(), "species"]))
        raise ValueError(f"Unexpected species labels: {unknown}")
    return frame


def load_from_sklearn() -> pd.DataFrame:
    from sklearn.datasets import load_iris

    bunch = load_iris(as_frame=True)
    frame = bunch.data.copy()
    frame.columns = COLUMNS[:4]
    frame["species"] = bunch.target.map(dict(enumerate(bunch.target_names)))
    print("Used scikit-learn's bundled Iris copy as a fallback.")
    return frame


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not NAMES_PATH.exists():
        download(NAMES_URL, NAMES_PATH)

    try:
        frame = load_from_uci()
    except Exception as exc:
        print(f"Falling back to scikit-learn: {exc}", file=sys.stderr)
        frame = load_from_sklearn()

    frame = frame[COLUMNS]
    frame.to_csv(CSV_PATH, index=False)

    print(f"Saved {len(frame)} rows x {frame.shape[1]} columns to {CSV_PATH}")
    print(frame["species"].value_counts().to_string())


if __name__ == "__main__":
    main()
