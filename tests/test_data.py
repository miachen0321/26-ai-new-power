import pytest

from iris_lab.data import (
    CLASS_NAMES,
    FEATURES,
    data_quality_report,
    load_data,
    validate_data,
)


def test_load_project_dataset() -> None:
    frame = load_data()

    assert frame.shape == (150, 5)
    assert list(frame.columns) == FEATURES + ["species"]
    assert frame["species"].value_counts().to_dict() == {
        name: 50 for name in CLASS_NAMES
    }
    assert not frame[FEATURES].isna().any().any()


def test_data_quality_report_reports_known_issues() -> None:
    report = data_quality_report(load_data())

    assert report["rows"] == 150
    assert report["columns"] == 5
    assert report["missing_values"] == 0
    assert report["duplicate_rows"] == 3
    assert report["outliers"]["sepal_width"] == 4
    assert report["outliers"]["petal_width"] == 0


def test_validate_rejects_missing_required_column() -> None:
    frame = load_data().drop(columns=["petal_width"])

    with pytest.raises(ValueError, match="missing required column"):
        validate_data(frame)


def test_validate_rejects_unknown_species() -> None:
    frame = load_data().copy()
    frame.loc[0, "species"] = "unknown"

    with pytest.raises(ValueError, match="unknown species"):
        validate_data(frame)


def test_validate_rejects_missing_values() -> None:
    frame = load_data().copy()
    frame.loc[0, "sepal_length"] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_data(frame)


def test_validate_accepts_dataframe_with_extra_column() -> None:
    frame = load_data()
    frame.insert(0, "row_id", range(len(frame)))

    validated = validate_data(frame)

    assert len(validated) == 150
