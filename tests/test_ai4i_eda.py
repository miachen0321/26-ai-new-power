"""TDD tests for AI4I 2020 EDA (issue #5): perform_eda contract and artifacts.

Synthetic tests run fully offline inside tmp_path sandboxes; the real-data
tests use data/ai4i2020.csv (local file, no network) and regenerate the
committed EDA artifacts under figures/ and results/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_ai4i_analysis as eda_module
from scripts.run_ai4i_analysis import (
    DATA_PATH,
    engineer_features,
    load_data,
    perform_eda,
)

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "ai4i_eda_summary.md"
EDA_FIGURES = [
    "ai4i_class_balance.png",
    "ai4i_correlation_heatmap.png",
    "ai4i_class_means.png",
    "ai4i_top_feature_boxplots.png",
]
BASE_NUMERIC = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]
FLAG_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]
SUMMARY_TITLE = "AI4I 2020 資料探索摘要"


def synthetic_frame(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    label = np.zeros(rows, dtype=int)
    label[rows // 2 :] = 1
    frame = pd.DataFrame(
        {
            "udi": np.arange(1, rows + 1),
            "product_id": [f"M{i}" for i in range(rows)],
            "type": rng.choice(["L", "M", "H"], size=rows),
            "air_temperature": rng.normal(298.0, 1.0, rows),
            "process_temperature": rng.normal(309.0, 1.0, rows),
            "rotational_speed": np.where(label == 1, 1200.0, 1600.0),
            "torque": np.where(label == 1, 60.0, 30.0),
            "tool_wear": rng.integers(0, 250, rows).astype(float),
            "machine_failure": label,
            **{flag: np.zeros(rows, dtype=int) for flag in FLAG_COLUMNS},
        }
    )
    frame.loc[frame.index[rows // 2 :], "twf"] = 1
    return frame


def class_mean_diff(enriched: pd.DataFrame) -> pd.Series:
    grouped = enriched.groupby("machine_failure")[BASE_NUMERIC].mean().loc[[0, 1]]
    return grouped.loc[1] - grouped.loc[0]


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(eda_module, "FIG_DIR", tmp_path)
    monkeypatch.setattr(eda_module, "RESULT_DIR", tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def real_summary_source() -> pd.DataFrame:
    if not DATA_PATH.exists():
        pytest.skip("data/ai4i2020.csv not prepared yet")
    return engineer_features(load_data(DATA_PATH))


def test_perform_eda_returns_all_summary_keys(sandbox: Path) -> None:
    summary = perform_eda(engineer_features(synthetic_frame()))

    required = {
        "shape",
        "missing_values",
        "duplicate_rows",
        "class_counts",
        "flag_counts",
        "correlation",
        "top_discriminating_features",
    }
    assert required.issubset(summary.keys())


def test_perform_eda_shape_covers_engineered_frame(sandbox: Path) -> None:
    enriched = engineer_features(synthetic_frame())
    summary = perform_eda(enriched)

    assert summary["shape"] == list(enriched.shape)
    assert summary["shape"] == [60, 16]


def test_perform_eda_missing_values_all_zero_for_clean_data(sandbox: Path) -> None:
    enriched = engineer_features(synthetic_frame())
    summary = perform_eda(enriched)

    assert set(summary["missing_values"].keys()) == set(enriched.columns)
    assert summary["missing_values"] == {column: 0 for column in enriched.columns}


def test_perform_eda_missing_values_counts_nan_cells(sandbox: Path) -> None:
    enriched = engineer_features(synthetic_frame())
    enriched.loc[enriched.index[3], "torque"] = np.nan
    summary = perform_eda(enriched)

    assert summary["missing_values"]["torque"] == 1
    assert sum(summary["missing_values"].values()) == 1


def test_perform_eda_duplicate_rows(sandbox: Path) -> None:
    base = synthetic_frame(rows=4)
    enriched = engineer_features(pd.concat([base, base.iloc[[0]]], ignore_index=True))
    assert enriched.duplicated().sum() == 1

    summary = perform_eda(enriched)

    assert summary["duplicate_rows"] == 1


def test_perform_eda_class_counts(sandbox: Path) -> None:
    summary = perform_eda(engineer_features(synthetic_frame()))

    assert summary["class_counts"] == {0: 30, 1: 30}
    assert all(isinstance(k, int) for k in summary["class_counts"].keys())
    assert sum(summary["class_counts"].values()) == 60


def test_perform_eda_flag_counts(sandbox: Path) -> None:
    enriched = engineer_features(synthetic_frame())
    summary = perform_eda(enriched)

    assert set(summary["flag_counts"].keys()) == set(FLAG_COLUMNS)
    assert summary["flag_counts"] == {
        flag: int(enriched[flag].sum()) for flag in FLAG_COLUMNS
    }
    assert summary["flag_counts"]["twf"] == 30


def test_perform_eda_correlation_structure_and_values(sandbox: Path) -> None:
    enriched = engineer_features(synthetic_frame(rows=200))
    summary = perform_eda(enriched)

    expected_keys = set(BASE_NUMERIC + ["machine_failure"])
    corr = summary["correlation"]
    assert set(corr.keys()) == expected_keys
    for column in corr:
        assert set(corr[column].keys()) == expected_keys
    assert corr["machine_failure"]["machine_failure"] == 1.0
    for i in expected_keys:
        for j in expected_keys:
            assert corr[i][j] == pytest.approx(corr[j][i], abs=1e-6)
            assert -1.0 <= corr[i][j] <= 1.0

    torque_pearson = np.corrcoef(
        enriched["torque"].to_numpy(), enriched["machine_failure"].to_numpy()
    )[0, 1]
    assert corr["machine_failure"]["torque"] == pytest.approx(torque_pearson, abs=1e-6)
    positive = {
        key: corr["machine_failure"][key]
        for key in BASE_NUMERIC
        if corr["machine_failure"][key] > 0
    }
    assert max(positive, key=positive.get) == "torque"


def test_perform_eda_top_discriminating_features_order(sandbox: Path) -> None:
    enriched = engineer_features(synthetic_frame())
    summary = perform_eda(enriched)

    diff = class_mean_diff(enriched)
    expected = diff.abs().sort_values(ascending=False).index.tolist()
    assert summary["top_discriminating_features"] == expected
    assert set(summary["top_discriminating_features"]) == set(BASE_NUMERIC)
    assert summary["top_discriminating_features"][0] == "rotational_speed"


def test_perform_eda_writes_figures_and_summary_in_sandbox(sandbox: Path) -> None:
    perform_eda(engineer_features(synthetic_frame()))

    for name in EDA_FIGURES:
        assert (sandbox / name).is_file(), name
    summary_text = (sandbox / "ai4i_eda_summary.md").read_text(encoding="utf-8")
    assert SUMMARY_TITLE in summary_text
    assert "各失效模式出現次數" in summary_text


def test_real_data_shape_and_cleanliness(real_summary_source: pd.DataFrame) -> None:
    summary = perform_eda(real_summary_source)

    assert summary["shape"] == [10000, 16]
    assert sum(summary["missing_values"].values()) == 0
    assert summary["duplicate_rows"] == 0


def test_real_data_class_counts_and_flag_counts(real_summary_source: pd.DataFrame) -> None:
    summary = perform_eda(real_summary_source)

    assert summary["class_counts"] == {0: 9661, 1: 339}
    assert summary["flag_counts"] == {
        "twf": 46,
        "hdf": 115,
        "pwf": 95,
        "osf": 98,
        "rnf": 19,
    }


def test_real_data_top_features_include_known_signals(
    real_summary_source: pd.DataFrame,
) -> None:
    summary = perform_eda(real_summary_source)

    diff = class_mean_diff(real_summary_source)
    expected = diff.abs().sort_values(ascending=False).index.tolist()
    assert summary["top_discriminating_features"] == expected
    for feature in ["torque", "tool_wear", "rotational_speed"]:
        assert feature in summary["top_discriminating_features"]


def test_results_summary_md_exists_and_contains_title(real_summary_source: pd.DataFrame) -> None:
    summary = perform_eda(real_summary_source)
    assert summary["shape"] == [10000, 16]

    text = SUMMARY_PATH.read_text(encoding="utf-8")
    assert SUMMARY_TITLE in text
    assert "10000" in text
    assert "各失效模式出現次數" in text


def test_real_eda_figures_exist_under_figures(real_summary_source: pd.DataFrame) -> None:
    perform_eda(real_summary_source)

    for name in EDA_FIGURES:
        assert (ROOT / "figures" / name).is_file(), name
