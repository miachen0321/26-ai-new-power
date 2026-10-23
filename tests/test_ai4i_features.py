"""TDD tests for AI4I 2020 feature engineering and preprocessing (issue #6)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_ai4i_analysis import (
    CATEGORICAL_FEATURES,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    build_model_frame,
    engineer_features,
    load_data,
    make_preprocessor,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
VALID_TYPES = {"L", "M", "H"}
EXPECTED_ONE_HOT = {"type_M", "type_H"}


@pytest.fixture(scope="module")
def enriched() -> pd.DataFrame:
    return engineer_features(load_data(DATA_PATH))


def test_model_frame_excludes_every_leakage_column(enriched: pd.DataFrame) -> None:
    x, _ = build_model_frame(enriched)
    for column in LEAKAGE_COLUMNS:
        assert column not in x.columns
    assert set(x.columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_preprocessor_output_never_leaks_columns(enriched: pd.DataFrame) -> None:
    x, _ = build_model_frame(enriched)
    names = make_preprocessor().fit(x).get_feature_names_out().tolist()
    for column in LEAKAGE_COLUMNS:
        assert all(column not in name for name in names)


def test_model_frame_shape_and_layout(enriched: pd.DataFrame) -> None:
    x, y = build_model_frame(enriched)
    assert list(x.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert x.shape == (len(enriched), len(NUMERIC_FEATURES) + 1)
    assert y.name == "machine_failure"
    assert set(y.unique()) == {0, 1}


def test_preprocessor_type_onehot_width_and_names(enriched: pd.DataFrame) -> None:
    x, _ = build_model_frame(enriched)
    preprocessor = make_preprocessor().fit(x)
    names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]

    one_hot = [name for name in names if name.startswith("type_")]
    assert len(one_hot) == 2
    assert set(one_hot) == EXPECTED_ONE_HOT

    numeric = [name for name in names if name in NUMERIC_FEATURES]
    assert len(numeric) == len(NUMERIC_FEATURES)


def test_derived_features_exact_formulas_on_synthetic_frame() -> None:
    raw = pd.DataFrame(
        {
            "udi": [1, 2],
            "product_id": ["M14860", "L47181"],
            "type": ["M", "L"],
            "air_temperature": [298.1, 298.2],
            "process_temperature": [308.6, 309.1],
            "rotational_speed": [1551, 1408],
            "torque": [42.8, 46.3],
            "tool_wear": [0, 3],
            "machine_failure": [0, 0],
            "twf": [0, 0],
            "hdf": [0, 0],
            "pwf": [0, 0],
            "osf": [0, 0],
            "rnf": [0, 0],
        }
    )
    out = engineer_features(raw)
    np.testing.assert_allclose(out["torque_speed_product"], out["torque"] * out["rotational_speed"])
    np.testing.assert_allclose(
        out["temperature_difference"],
        out["process_temperature"] - out["air_temperature"],
    )
    assert out["torque_speed_product"].iloc[0] == pytest.approx(42.8 * 1551)
    assert out["temperature_difference"].iloc[1] == pytest.approx(309.1 - 298.2)
    assert len(out) == len(raw)


def test_derived_features_match_formulas_on_full_data(enriched: pd.DataFrame) -> None:
    np.testing.assert_allclose(
        enriched["torque_speed_product"],
        enriched["torque"] * enriched["rotational_speed"],
    )
    np.testing.assert_allclose(
        enriched["temperature_difference"],
        enriched["process_temperature"] - enriched["air_temperature"],
    )


def test_feature_matrix_and_encoded_matrix_have_no_nan(enriched: pd.DataFrame) -> None:
    x, _ = build_model_frame(enriched)
    assert not x.isna().any().any()

    matrix = make_preprocessor().fit_transform(x)
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    assert np.isfinite(dense).all()


def test_type_values_are_valid_globally(enriched: pd.DataFrame) -> None:
    assert set(enriched["type"].unique()) == VALID_TYPES
    assert enriched["type"].map(lambda value: value in VALID_TYPES).all()
