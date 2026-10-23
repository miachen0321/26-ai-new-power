"""Conclusion-preserving tests for the experiment's stated conclusions.

These tests assert the experiment's conclusions (not individual functions):
1. The model feature list contains no failure-mode columns nor the target.
2. Models do not predict the majority class only (recall > 0, non-uniform predictions).
3. Every number in docs/results.json is consistent with the reports in reports/.
4. Saved preprocessing parameters (mean/std) reproduce identical golden predictions.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ai4i_analysis import build_model_frame, build_models, engineer_features

RESULTS_JSON = ROOT / "docs" / "results.json"
MODEL_PARAMS = ROOT / "docs" / "model_params.json"
GOLDEN = ROOT / "tests" / "golden.json"
BASELINES = ROOT / "report" / "ai4i_baselines.md"
IRIS_PAPER = ROOT / "report" / "iris_neural_network_paper.md"

BANNED_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf", "machine_failure"]


def test_feature_list_excludes_failure_flags_and_target() -> None:
    """結論 1：特徵清單不含 TWF/HDF/PWF/OSF/RNF，也不含 Machine failure。"""
    ai4i = engineer_features(pd.read_csv(ROOT / "data" / "ai4i2020.csv"))
    x, _ = build_model_frame(ai4i)

    leaked = sorted(set(x.columns) & set(BANNED_COLUMNS))
    assert not leaked, f"特徵矩陣混入了失效旗標／目標欄位：{leaked}"

    parameters = json.loads(MODEL_PARAMS.read_text(encoding="utf-8"))
    feature_order = parameters["ai4i_logreg"]["feature_order"]
    leaked_saved = [c for c in feature_order if c in BANNED_COLUMNS]
    assert not leaked_saved, f"model_params.json feature_order 含禁用欄位：{leaked_saved}"
    expanded = (set(x.columns) - {"type"}) | {"type_M", "type_H"}
    assert set(feature_order) == set(expanded), (
        f"feature_order {feature_order} 與訓練特徵 one-hot 展開後 {sorted(expanded)} 不一致"
    )

    saved = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    leaked_importance = [n for n in saved["ai4i"]["rf_feature_importance"] if n in BANNED_COLUMNS]
    assert not leaked_importance, f"特徵重要度表含禁用欄位：{leaked_importance}"


def test_models_do_not_predict_majority_only() -> None:
    """結論 2：模型不是只預測多數類——recall > 0，且預測不是全部同一類。"""
    ai4i = engineer_features(pd.read_csv(ROOT / "data" / "ai4i2020.csv"))
    x, y = build_model_frame(ai4i)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=42
    )
    assert int((y_test == 1).sum()) > 0, "測試切分本身無正類，檢驗無意義"

    for name, pipeline in build_models().items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        true_positives = int(((predictions == 1) & (y_test == 1)).sum())
        recall = true_positives / int((y_test == 1).sum())
        assert recall > 0, f"{name} recall = 0（等於只會說『不會故障』）"
        assert sorted(set(predictions.tolist())) != [0], (
            f"{name} 的預測輸出全部為 0（單一類別）"
        )

    saved = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    for name, metrics in saved["ai4i"]["models"].items():
        assert metrics["recall"] > 0, f"results.json {name} recall = 0"
        assert metrics["confusion_matrix"][1][1] > 0, (
            f"results.json {name} 混淆矩陣 TP = 0（存檔的模型就是只預測多數類）"
        )


def test_ai4i_baseline_report_numbers_match_results_json() -> None:
    """結論 3（AI4I）：report/ai4i_baselines.md 表內每個數字與 docs/results.json 一致。"""
    text = BASELINES.read_text(encoding="utf-8")
    saved = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))["ai4i"]["models"]
    pattern = re.compile(
        r"\| (?P<model>\w[\w_]*)\s*\| (?P<recall>[0-9.]+)\s*\| (?P<f1>[0-9.]+)\s*\| "
        r"(?P<pr_auc>[0-9.]+)\s*\|\s*(?P<roc_auc>[0-9.]+)\s*\|"
    )
    rows = {m.group("model"): m.groupdict() for m in pattern.finditer(text)}
    assert set(rows) == set(saved), f"報告模型清單不一致：{sorted(rows)} vs {sorted(saved)}"
    for model, metrics in saved.items():
        row = rows[model]
        for metric in ("recall", "f1", "pr_auc", "roc_auc"):
            report_value = float(row[metric])
            assert abs(report_value - metrics[metric]) < 5e-5, (
                f"{model}.{metric}：報告 {report_value} vs results.json {metrics[metric]:.6f}"
            )


def test_iris_report_numbers_match_results_json() -> None:
    """結論 3（Iris）：論文中的數字與 docs/results.json/訓練結果一致。"""
    text = IRIS_PAPER.read_text(encoding="utf-8")
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))["iris"]
    expected = {
        "test_accuracy": f"{data['test_accuracy'] * 100:.2f}",
        "test_loss": f"{data['test_loss']:.4f}",
        "macro_auc": f"{data['macro_auc']:.3f}",
        "cv_mean_accuracy": f"{data['cv_mean_accuracy'] * 100:.2f}",
    }
    for key, rendered in expected.items():
        assert rendered in text, f"results.json iris.{key} = {rendered} 在論文中找不到"


def test_saved_preprocessing_params_reproduce_golden_predictions() -> None:
    """結論 4：mean/std 有存檔，且用存檔參數重算可重現相同預測。"""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    parameters = json.loads(MODEL_PARAMS.read_text(encoding="utf-8"))
    ai = parameters["ai4i_logreg"]

    mean = np.asarray(ai["scaler_mean"])
    scale = np.asarray(ai["scaler_scale"])
    coef = np.asarray(ai["coef"])
    intercept = ai["intercept"]

    for index, case in enumerate(golden["ai4i"]):
        row = case["inputs"]
        nums = np.array([
            row["air_temperature"], row["process_temperature"], row["rotational_speed"],
            row["torque"], row["tool_wear"], row["torque"] * row["rotational_speed"],
            row["process_temperature"] - row["air_temperature"],
        ])
        z = np.concatenate([(nums - mean) / scale, [float(row["type"] == "M"), float(row["type"] == "H")]])
        probability = 1.0 / (1.0 + np.exp(-(float(np.dot(z, coef)) + intercept)))
        expected = case["probabilities"]["failure"]
        assert probability == pytest.approx(expected, abs=1e-6), (
            f"ai4i golden[{index}]：以存檔 mean/std/coef 重算得 {probability:.6f}"
            + f" ≠ 黃金值 {expected:.6f}（前處理參數無法重現相同預測）"
        )
