"""Issue #7：AI4I 基準模型測試（建模成本考量，使用前 500 筆抽樣資料）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT / "scripts"

import sys  # noqa: E402

sys.path.insert(0, str(sys_path))

from run_ai4i_analysis import (  # noqa: E402
    build_model_frame,
    build_models,
    engineer_features,
    make_preprocessor,
)

EXPECTED_MODELS = ["logistic_regression", "decision_tree", "random_forest"]


@pytest.fixture(scope="module")
def sampled_xy() -> tuple[pd.DataFrame, pd.Series]:
    frame = engineer_features(pd.read_csv(ROOT / "data" / "ai4i2020.csv").head(500))
    return build_model_frame(frame)


@pytest.fixture(scope="module")
def comparison_csv() -> pd.DataFrame:
    path = ROOT / "results" / "ai4i_model_comparison.csv"
    if not path.exists():
        pytest.skip("results/ai4i_model_comparison.csv 尚未由研究腳本產生")
    return pd.read_csv(path)


def test_build_models_returns_three_named_pipelines() -> None:
    models = build_models()
    assert list(models.keys()) == EXPECTED_MODELS
    for name in EXPECTED_MODELS:
        pipeline = models[name]
        assert isinstance(pipeline, Pipeline)
        assert ["preprocess", "model"] == [step for step, _ in pipeline.steps]
        assert isinstance(pipeline.named_steps["preprocess"], ColumnTransformer)
        assert pipeline.named_steps["preprocess"] is not make_preprocessor()


@pytest.mark.parametrize(
    "name,estimator_cls,weight",
    [
        ("logistic_regression", LogisticRegression, "balanced"),
        ("decision_tree", DecisionTreeClassifier, "balanced"),
        ("random_forest", RandomForestClassifier, "balanced_subsample"),
    ],
)
def test_class_weight_imbalance_handling(name: str, estimator_cls: type, weight: str) -> None:
    model = build_models()[name].named_steps["model"]
    assert isinstance(model, estimator_cls)
    assert model.class_weight == weight


@pytest.mark.parametrize(
    "name,estimator_cls",
    [
        ("logistic_regression", LogisticRegression),
        ("decision_tree", DecisionTreeClassifier),
        ("random_forest", RandomForestClassifier),
    ],
)
def test_predict_proba_on_sample(sampled_xy: tuple[pd.DataFrame, pd.Series], name: str, estimator_cls: type) -> None:
    x, y = sampled_xy
    pipeline = build_models()[name]
    assert isinstance(pipeline.named_steps["model"], estimator_cls)
    pipeline.fit(x, y)
    proba = pipeline.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


@pytest.mark.parametrize("name", EXPECTED_MODELS)
def test_reproducibility(sampled_xy: tuple[pd.DataFrame, pd.Series], name: str) -> None:
    x, y = sampled_xy
    proba_a = build_models()[name].fit(x, y).predict_proba(x)
    proba_b = build_models()[name].fit(x, y).predict_proba(x)
    np.testing.assert_allclose(proba_a, proba_b)


def test_model_comparison_csv(comparison_csv: pd.DataFrame) -> None:
    assert set(["model", "recall", "f1", "pr_auc", "roc_auc"]).issubset(comparison_csv.columns)
    assert comparison_csv.shape[0] == 3
    assert set(comparison_csv["model"].tolist()) == set(EXPECTED_MODELS)
    assert {"recall", "f1", "pr_auc", "roc_auc"}.issubset(comparison_csv.columns)
