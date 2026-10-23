"""AI4I 2020 predictive maintenance baseline model comparison.

Binary classification of `machine_failure` with scikit-learn baselines.

Run from the project root:
    python3 scripts/run_ai4i_analysis.py

All figures are written to figures/ and all numerical results to results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"

NUMERIC_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "torque_speed_product",
    "temperature_difference",
]
CATEGORICAL_FEATURES = ["type"]
LEAKAGE_COLUMNS = ["udi", "product_id", "machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]
SEED = 42
TEST_SIZE = 0.30

sns.set_theme(style="whitegrid", context="notebook")


def save_figure(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"Saved figure: {shown}")


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.isna().any().any():
        raise ValueError("Missing values in ai4i2020.csv")
    return frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["torque_speed_product"] = (
        enriched["torque"] * enriched["rotational_speed"]
    )
    enriched["temperature_difference"] = (
        enriched["process_temperature"] - enriched["air_temperature"]
    )
    return enriched


def build_model_frame(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = enriched[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = enriched["machine_failure"].astype(int)
    return x, y


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", categories=[["L", "M", "H"]]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=6, class_weight="balanced", random_state=SEED
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=10,
                        class_weight="balanced_subsample",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def perform_eda(enriched: pd.DataFrame) -> dict:
    numeric = NUMERIC_FEATURES[:5]
    class_counts = enriched["machine_failure"].value_counts().sort_index().to_dict()
    subset = enriched[numeric + ["machine_failure"]]
    corr = subset.corr().round(4)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = enriched["machine_failure"].value_counts().sort_index()
    bars = ax.bar(["OK (0)", "Failure (1)"], counts.values, color=["#55A868", "#C44E52"], alpha=0.85)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, f"{value}", ha="center", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title(f"Binary class balance (failure rate {counts[1] / counts.sum() * 100:.2f}%)")
    save_figure(fig, "ai4i_class_balance.png")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, linewidths=0.6, ax=ax)
    ax.set_title("Pearson correlation of raw features with machine failure")
    save_figure(fig, "ai4i_correlation_heatmap.png")

    grouped = enriched.groupby("machine_failure")[numeric].mean().loc[[0, 1]]
    diff = grouped.loc[1] - grouped.loc[0]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    positions = np.arange(len(numeric))
    width = 0.35
    ax.bar(positions - width / 2, grouped.loc[0].values, width, label="No failure", color="#55A868", alpha=0.85)
    ax.bar(positions + width / 2, grouped.loc[1].values, width, label="Failure", color="#C44E52", alpha=0.85)
    ax.set_xticks(positions, numeric, rotation=12)
    ax.set_ylabel("Mean by class")
    ax.set_title("Feature class means: failure vs no failure")
    ax.legend()
    save_figure(fig, "ai4i_class_means.png")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    strongest = diff.abs().sort_values(ascending=False).index[:3].tolist()
    for ax, column in zip(axes, strongest):
        sns.boxplot(data=enriched, x="machine_failure", y=column, hue="machine_failure",
                    palette=["#55A868", "#C44E52"], legend=False, fliersize=1.5, ax=ax)
        ax.set_xlabel("machine_failure")
        ax.set_title(column)
    fig.tight_layout()
    save_figure(fig, "ai4i_top_feature_boxplots.png")

    summary = {
        "shape": list(enriched.shape),
        "missing_values": {c: int(v) for c, v in enriched.isna().sum().items()},
        "duplicate_rows": int(enriched.duplicated().sum()),
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
        "flag_counts": {c: int(enriched[c].sum()) for c in ["twf", "hdf", "pwf", "osf", "rnf"]},
        "describe": enriched[numeric].describe().round(4).to_dict(),
        "correlation": corr.to_dict(),
        "class_means": grouped.round(4).to_dict(),
        "top_discriminating_features": diff.abs().sort_values(ascending=False).index.tolist(),
    }

    lines = [
        "# AI4I 2020 資料探索摘要",
        "",
        f"- 資料筆數：{summary['shape'][0]}",
        f"- 欄位數：{summary['shape'][1]}",
        f"- 缺失值：{sum(summary['missing_values'].values())}",
        f"- 重複列：{summary['duplicate_rows']}",
        f"- 類別分布：正常 {class_counts.get(0, 0)}、失效 {class_counts.get(1, 1)}",
        f"- 失效比例：{class_counts.get(1, 1) / summary['shape'][0] * 100:.2f}%",
        "## 各失效模式出現次數",
        "",
        "```text",
        *(f"{c}: {summary['flag_counts'][c]}" for c in ["twf", "hdf", "pwf", "osf", "rnf"]),
        "```",
        "## 特徵與失效的相關係數",
        "",
        corr["machine_failure"].to_markdown(),
        "",
        "## 各類別特徵平均",
        "",
        grouped.round(3).to_markdown(),
        "",
        "## 區辨力最高的特徵（依類別平均差）",
        "",
        "```text",
        *(f"- {c}: {diff[c]:.4f}" for c in summary["top_discriminating_features"]),
        "```",
    ]
    (RESULT_DIR / "ai4i_eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def train_and_compare(x: pd.DataFrame, y: pd.Series, enriched: pd.DataFrame) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"Train: {len(x_train)} rows, test: {len(x_test)} rows")

    metrics = {}
    frames = {}
    for name, pipeline in build_models().items():
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        precision_all, recall_all, f1_all, _ = precision_recall_fscore_support(
            y_test, prediction, labels=[0, 1], zero_division=0
        )
        positive = precision_recall_fscore_support(
            y_test, prediction, labels=[1], zero_division=0
        )
        metrics[name] = {
            "precision": float(positive[0][0]),
            "recall": float(positive[1][0]),
            "f1": float(positive[2][0]),
            "class_0_precision": float(precision_all[0]),
            "class_0_recall": float(recall_all[0]),
            "class_0_f1": float(f1_all[0]),
            "pr_auc": float(average_precision_score(y_test, probability)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
            "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
        }
        frames[name] = pd.DataFrame({"probability": probability, "sub_index": np.arange(len(y_test))})
        print(
            f"{name}: recall={metrics[name]['recall']:.3f} f1={metrics[name]['f1']:.3f} "
            f"pr_auc={metrics[name]['pr_auc']:.3f} roc_auc={metrics[name]['roc_auc']:.3f}"
        )

    comparison = pd.DataFrame(metrics).T[
        ["recall", "f1", "pr_auc", "roc_auc"]
    ].reset_index().rename(columns={"index": "model"}).round(4)
    comparison.to_csv(RESULT_DIR / "ai4i_model_comparison.csv", index=False)

    comparison_metrics = pd.DataFrame(metrics).T.drop(columns=["confusion_matrix"]).round(4)
    comparison_metrics.to_csv(RESULT_DIR / "ai4i_full_metrics.csv")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = comparison.plot(
        kind="bar",
        x="model",
        ax=ax,
        color=["#DD8452", "#55A868", "#4C72B0", "#8172B2"],
        rot=0,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Test-set baseline comparison (recall / F1 / PR-AUC / ROC-AUC)")
    ax.set_xlabel("")
    ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=8, padding=1)
    ax.bar_label(ax.containers[1], fmt="%.3f", fontsize=8, padding=1)
    ax.bar_label(ax.containers[2], fmt="%.3f", fontsize=8, padding=1)
    ax.bar_label(ax.containers[3], fmt="%.3f", fontsize=8, padding=1)
    ax.legend(ncol=2)
    save_figure(fig, "ai4i_model_comparison.png")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, color in zip(metrics, ["#4C72B0", "#55A868", "#C44E52"]):
        pipeline = build_models()[name]
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, probability)
        baseline = y_test.mean()
        ax.plot(
            recall_curve,
            precision_curve,
            color=color,
            linewidth=2,
            label=f"{name} (AP = {metrics[name]['pr_auc']:.3f})",
        )
    ax.axhline(baseline, color="gray", linestyle=":", linewidth=1.2, label=f"Chance ({baseline:.3f})")
    ax.set_xlabel("Recall (true positive rate)")
    ax.set_ylabel("Precision")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Precision-Recall curves on the test set")
    ax.legend(loc="upper right")
    save_figure(fig, "ai4i_precision_recall_curves.png")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, color in zip(metrics, ["#4C72B0", "#55A868", "#C44E52"]):
        pipeline = build_models()[name]
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probability)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {metrics[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance (0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right")
    save_figure(fig, "ai4i_roc_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, name in zip(axes, metrics):
        matrix = np.asarray(metrics[name]["confusion_matrix"])
        row_norm = matrix / matrix.sum(axis=1, keepdims=True)
        sns.heatmap(
            row_norm,
            annot=np.array([[f"{matrix[i, j]:d}\n({row_norm[i, j]:.1%})" for j in range(2)] for i in range(2)]),
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=["OK", "Failure"],
            yticklabels=["OK", "Failure"],
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    save_figure(fig, "ai4i_confusion_matrices.png")

    best_name = max(metrics, key=lambda key: metrics[key]["f1"])
    forest = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=10,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    forest.fit(x_train, y_train)
    names = forest.named_steps["preprocess"].get_feature_names_out()
    names = [feature.split("__", 1)[-1] for feature in names]
    importances = forest.named_steps["model"].feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(np.array(names)[order], importances[order], color="#4C72B0", alpha=0.85)
    ax.set_title("Random Forest feature importances")
    ax.set_xlabel("Importance")
    save_figure(fig, "ai4i_rf_feature_importance.png")

    perm = permutation_importance(forest, x_test, y_test, n_repeats=10, random_state=SEED, n_jobs=-1)
    perm_frame = pd.DataFrame(
        {
            "feature": x.columns.tolist(),
            "importance_mean": perm.importances_mean.round(5),
            "importance_std": perm.importances_std.round(5),
        }
    ).sort_values("importance_mean", ascending=False)
    perm_frame.to_csv(RESULT_DIR / "ai4i_permutation_importance.csv", index=False)

    metrics_record = {
        "random_seed": SEED,
        "test_size": TEST_SIZE,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "positive_class_rate_train": float(y_train.mean()),
        "positive_class_rate_test": float(y_test.mean()),
        "models": metrics,
        "best_model_by_f1": best_name,
        "rf_feature_importance": dict(zip(names, importances.round(4))),
        "permutation_importance": dict(zip(perm_frame["feature"], perm_frame["importance_mean"])),
    }
    (RESULT_DIR / "ai4i_metrics.json").write_text(
        json.dumps(metrics_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics_record


def write_report(eda: dict, model_record: dict) -> None:
    comparison = pd.DataFrame(model_record["models"]).T.drop(columns=["confusion_matrix"]).round(4)
    comparison_table = comparison[["recall", "f1", "pr_auc", "roc_auc"]].to_markdown()
    confusion_lines = (
        f"- {name}: TP={metrics['confusion_matrix'][1][1]}, FN={metrics['confusion_matrix'][1][0]}, "
        f"FP={metrics['confusion_matrix'][0][1]}, TN={metrics['confusion_matrix'][0][0]}"
        for name, metrics in model_record["models"].items()
    )
    feature_lines = (
        f"- {key}: {value}"
        for key, value in sorted(
            model_record["rf_feature_importance"].items(), key=lambda item: item[1], reverse=True
        )[:8]
    )
    report = f"""# 基準模型比較：AI4I 2020 預測性維護

- 資料筆數：{eda['shape'][0]}，欄位數：{eda['shape'][1]}
- 失效比例：{(eda['class_counts'].get(1, 1) / eda['shape'][0]) * 100:.2f}%（類別嚴重不平衡，不採用 Accuracy 做主指標）
- 切分：分層 70/30（seed = {SEED}），以 recall / F1 / PR-AUC / ROC-AUC 評估
- 失效模式分布：{', '.join(f"{k} {v}" for k, v in eda["flag_counts"].items())}

## 特徵工程

- `type` one-hot 編碼（drop first）
- 派生特徵：`torque_speed_product`（扭矩 × 轉速）、`temperature_difference`（製程溫度 − 環境溫度）
- 資料外洩防護：移除 `udi`、`product_id`，並且**不使用** 5 個失效模式欄位作為特徵
- 不平衡處理：Logistic Regression / Decision Tree 使用 `class_weight=balanced`，Random Forest 使用 `balanced_subsample`

## Top 5 區辨特徵

```text
{chr(10).join(f"- {c}" for c in eda['top_discriminating_features'][:5])}
```

## 模型比較（測試集）

{comparison_table}

## 混淆矩陣摘要

```text
{chr(10).join(confusion_lines)}
```

## 特徵重要度（Random Forest）

```text
{chr(10).join(feature_lines)}
```

## 結論

- 三組基準模型中，以 F1 最優者為 `{model_record['best_model_by_f1']}`。
- 由於類別不平衡，PR-AUC 與 Recall 是本任務較具參考價值的指標；Precision-Recall 曲線見 `figures/ai4i_precision_recall_curves.png`。
- 後續可再比較：閾值調校、SMOTE 過採樣、以及 Gradient Boosting 系列模型。
"""
    (ROOT / "report" / "ai4i_baselines.md").write_text(report, encoding="utf-8")
    print("Saved report: report/ai4i_baselines.md")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_data(DATA_PATH)
    print(f"Loaded {frame.shape[0]} rows and {frame.shape[1]} columns")
    enriched = engineer_features(frame)
    eda = perform_eda(enriched)
    x, y = build_model_frame(enriched)
    unused = [column for column in LEAKAGE_COLUMNS if column in enriched.columns and column not in x.columns]
    print(f"Excluded leakage columns: {unused}")
    model_record = train_and_compare(x, y, enriched)
    write_report(eda, model_record)
    print(f"\nBest baseline by F1: {model_record['best_model_by_f1']}")


if __name__ == "__main__":
    main()
