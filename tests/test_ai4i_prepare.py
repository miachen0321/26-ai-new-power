from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_ai4i import (
    CSV_PATH,
    COLUMNS,
    FLAG_COLUMNS,
    consistency_flags,
    load_frame,
    quality_report,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]

VALID_ROW = {
    "udi": 1,
    "product_id": "M14860",
    "type": "L",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0,
    "machine_failure": 0,
    **{flag: 0 for flag in FLAG_COLUMNS},
}


def make_row(**overrides) -> dict:
    return {**VALID_ROW, **overrides}


def sample_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_load_frame_strips_bom_and_names_columns(tmp_path: Path) -> None:
    frame = sample_frame([make_row()])
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")

    loaded = load_frame(path)

    assert list(loaded.columns) == COLUMNS
    assert loaded["type"].tolist() == ["L"]


def test_load_frame_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_frame(tmp_path / "not_exists.csv")


def test_load_real_csv_matches_width() -> None:
    if not CSV_PATH.exists():
        pytest.skip("data/ai4i2020.csv not prepared yet")
    loaded = load_frame(CSV_PATH)
    assert list(loaded.columns) == COLUMNS
    assert len(loaded) == 10000


def test_validate_accepts_valid_frame() -> None:
    validate(sample_frame([make_row()]))


def test_validate_rejects_nan() -> None:
    frame = sample_frame([make_row(air_temperature=float("nan"))])
    with pytest.raises(ValueError, match="Missing values"):
        validate(frame)


def test_validate_rejects_non_binary_failure() -> None:
    frame = sample_frame([make_row(machine_failure=2)])
    with pytest.raises(ValueError, match="binary"):
        validate(frame)


def test_validate_rejects_bad_type() -> None:
    frame = sample_frame([make_row(type="X")])
    with pytest.raises(ValueError, match="one of L, M, H"):
        validate(frame)


def test_validate_rejects_bad_product_id() -> None:
    frame = sample_frame([make_row(product_id="XX")])
    with pytest.raises(ValueError, match="product_id"):
        validate(frame)


def test_quality_report_counts() -> None:
    frame = sample_frame([make_row(), make_row(udi=2), make_row(udi=2)])
    frame.loc[0, "torque"] = np.nan

    report = quality_report(frame)

    assert report["rows"] == 3
    assert report["missing_values"] == 1
    assert report["duplicate_rows"] == 1
    assert report["invalid_product_ids"] == 0


def test_quality_report_invalid_product_id() -> None:
    frame = sample_frame([make_row(), make_row(product_id="BADID")])
    assert quality_report(frame)["invalid_product_ids"] == 1


def test_consistency_flags_detects_mismatch() -> None:
    frame = sample_frame([make_row(), make_row(twf=1)])

    flags = consistency_flags(frame)

    assert flags.tolist() == [False, True]


def test_consistency_flags_real_data_matches_saved_record() -> None:
    if not CSV_PATH.exists():
        pytest.skip("data/ai4i2020.csv not prepared yet")
    loaded = load_frame(CSV_PATH)
    inconsistency_path = ROOT / "data" / "ai4i2020_flag_inconsistency.csv"
    if not inconsistency_path.exists():
        pytest.skip("inconsistency record not saved yet")
    saved = pd.read_csv(inconsistency_path)

    assert len(saved) == len(loaded)
    assert int(consistency_flags(loaded).sum()) == int(saved["inconsistent"].sum())
