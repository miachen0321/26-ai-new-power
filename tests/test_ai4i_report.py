import csv
import json
import pathlib

from scripts.run_ai4i_analysis import write_report

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "ai4i_metrics.json"
PERM_PATH = ROOT / "results" / "ai4i_permutation_importance.csv"
REPORT_PATH = ROOT / "report" / "ai4i_baselines.md"


def test_report_contains_title_and_table_columns() -> None:
    model_record = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    eda = {
        "shape": (10000, 16),
        "class_counts": {0: 9661, 1: 339},
        "flag_counts": {"twf": 46, "hdf": 115, "pwf": 95, "osf": 98, "rnf": 19},
        "top_discriminating_features": ["rotational_speed", "tool_wear", "torque"],
    }
    write_report(eda, model_record)
    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "基準模型比較" in report
    for col in ["recall", "f1", "pr_auc", "roc_auc"]:
        assert col in report


def test_metrics_json_has_required_keys() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    for key in ["models", "best_model_by_f1", "rf_feature_importance", "permutation_importance"]:
        assert key in metrics


def test_permutation_csv_sorted_descending() -> None:
    with open(PERM_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        for col in ["feature", "importance_mean", "importance_std"]:
            assert col in reader.fieldnames
        rows = list(reader)
    means = [float(r["importance_mean"]) for r in rows]
    assert means == sorted(means, reverse=True)


def test_confusion_matrices_sum_to_test_rows() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    test_rows = metrics["test_rows"]
    for name, model in metrics["models"].items():
        cm = model["confusion_matrix"]
        assert sum(sum(row) for row in cm) == test_rows, f"{name}: {cm} != test_rows {test_rows}"
