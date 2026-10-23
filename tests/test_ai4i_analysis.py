import numpy as np
import pandas as pd
import pytest

from scripts.run_ai4i_analysis import (
    engineer_features,
    build_model_frame,
    build_models,
    load_data,
    make_preprocessor,
)


ROOT_PATH = "data/ai4i2020.csv"


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return load_data(__import__("pathlib").Path(__file__).resolve().parents[1] / ROOT_PATH)


@pytest.fixture(scope="module")
def enriched(frame: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(frame)


def test_load_data_has_expected_shape_and_columns() -> None:
    expected = [
        "udi",
        "product_id",
        "type",
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
        "machine_failure",
        "twf",
        "hdf",
        "pwf",
        "osf",
        "rnf",
    ]
    frame = load_data(__import__("pathlib").Path(__file__).resolve().parents[1] / ROOT_PATH)
    assert list(frame.columns) == expected
    assert len(frame) == 10000
    assert not frame.isna().any().any()


def test_engineer_features_adds_derived_columns(enriched: pd.DataFrame) -> None:
    np.testing.assert_allclose(
        enriched["torque_speed_product"],
        enriched["torque"] * enriched["rotational_speed"],
    )
    np.testing.assert_allclose(
        enriched["temperature_difference"],
        enriched["process_temperature"] - enriched["air_temperature"],
    )


def test_build_model_frame_excludes_leakage_columns(enriched: pd.DataFrame) -> None:
    x, y = build_model_frame(enriched)
    for column in ["machine_failure", "twf", "hdf", "pwf", "osf", "rnf", "udi", "product_id"]:
        assert column not in x.columns
    assert set(y.unique()) == {0, 1}
    assert len(x) == len(enriched)


def test_model_frame_is_not_empty(enriched: pd.DataFrame) -> None:
    x, y = build_model_frame(enriched)
    assert len(x) == len(enriched)
    assert set(y.unique()) == {0, 1}


def test_make_preprocessor_outputs_expected_width(enriched: pd.DataFrame) -> None:
    x, _ = build_model_frame(enriched)
    matrix = make_preprocessor().fit_transform(x)
    assert matrix.shape[0] == len(x)


def test_all_models_fit_and_predict(enriched: pd.DataFrame) -> None:
    x, y = build_model_frame(enriched)
    sample_x = x.iloc[:500]
    sample_y = y.iloc[:500]
    for name, pipeline in build_models().items():
        pipeline.fit(sample_x, sample_y)
        probability = pipeline.predict_proba(sample_x)[:, 1]
        assert len(probability) == len(sample_x)
        assert np.all((probability >= 0) & (probability <= 1))
        assert name in {"logistic_regression", "decision_tree", "random_forest"}
