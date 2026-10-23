import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "data/iris.csv",
    "data/iris.data",
    "data/iris.names",
    "figures/pca_3d.png",
    "figures/pca_explained_variance.png",
    "figures/training_curves.png",
    "figures/confusion_matrix.png",
    "figures/per_class_metrics.png",
    "figures/roc_curves.png",
    "figures/cross_validation.png",
    "models/iris_mlp.keras",
    "results/metrics.json",
    "results/classification_report.txt",
    "results/eda_summary.md",
    "report/iris_neural_network_paper.md",
    "report/iris_neural_network_paper.html",
]


@pytest.mark.parametrize("relative_path", REQUIRED_FILES)
def test_required_artifact_exists(relative_path: str) -> None:
    path = ROOT / relative_path

    assert path.is_file()
    assert path.stat().st_size > 0


def test_model_metrics_meet_acceptance_thresholds() -> None:
    metrics = json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))

    assert metrics["test_accuracy"] >= 0.90
    assert metrics["macro_auc"] >= 0.95
    assert metrics["cv_mean_accuracy"] >= 0.90
    assert metrics["train_size"] == 105
    assert metrics["test_size"] == 45


def test_report_covers_all_issue_topics() -> None:
    report = (ROOT / "report/iris_neural_network_paper.md").read_text(encoding="utf-8")

    for term in ["欄位", "PCA", "類神經網路", "混淆矩陣", "交叉驗證", "小論文"]:
        assert term in report

    assert report.count("![") >= 8
    assert "../figures/pca_3d.png" in report
    assert "../figures/training_curves.png" in report
    assert "../figures/confusion_matrix.png" in report
