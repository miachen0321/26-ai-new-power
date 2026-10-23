"""Iris exploratory analysis, PCA visualization and neural-network classification.

Run from the project root:
    python3 scripts/run_analysis.py

All figures are written to figures/ and all numerical results to results/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize

import keras
import tensorflow as tf

from iris_lab.data import CLASS_NAMES, FEATURES, data_quality_report, load_data
from iris_lab.features import compute_pca
from iris_lab.modeling import build_model, evaluate_predictions, split_and_scale

DATA_PATH = ROOT / "data" / "iris.csv"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

COLORS = {"setosa": "#4C72B0", "versicolor": "#DD8452", "virginica": "#55A868"}
SEED = 42

tf.get_logger().setLevel("ERROR")
keras.utils.set_random_seed(SEED)
sns.set_theme(style="whitegrid", context="notebook")


def save_figure(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path.relative_to(ROOT)}")


def perform_eda(frame: pd.DataFrame) -> dict:
    quality = data_quality_report(frame)
    summary = {
        "shape": list(frame.shape),
        "columns": list(frame.columns),
        "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
        "missing_values": {c: int(v) for c, v in frame.isna().sum().items()},
        "duplicate_rows": quality["duplicate_rows"],
        "class_counts": quality["class_counts"],
        "describe": frame[FEATURES].describe().round(4).to_dict(),
        "class_means": frame.groupby("species")[FEATURES].mean().round(4).to_dict(),
        "correlation": frame[FEATURES].corr().round(4).to_dict(),
        "iqr_outliers": quality["outliers"],
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = CLASS_NAMES
    sns.countplot(data=frame, x="species", order=order, hue="species", palette=COLORS, legend=False, ax=ax)
    for patch in ax.patches:
        ax.annotate(
            f"{int(patch.get_height())}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_title("Class distribution (n = 150)")
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    save_figure(fig, "class_distribution.png")

    pair = sns.pairplot(
        frame,
        hue="species",
        hue_order=CLASS_NAMES,
        palette=COLORS,
        diag_kind="kde",
        plot_kws={"s": 28, "alpha": 0.75, "edgecolor": "none"},
        corner=False,
    )
    pair.fig.suptitle("Iris feature pair plot", y=1.02, fontsize=15)
    pair.savefig(FIG_DIR / "feature_pairplot.png", dpi=180, bbox_inches="tight")
    plt.close(pair.fig)
    print("Saved figure: figures/feature_pairplot.png")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    corr = frame[FEATURES].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.6,
        ax=ax,
    )
    ax.set_title("Pearson correlation between features")
    save_figure(fig, "correlation_heatmap.png")

    lines = [
        "# 資料探索摘要",
        "",
        f"- 資料筆數：{summary['shape'][0]}",
        f"- 欄位數：{summary['shape'][1]}",
        f"- 缺失值：{sum(summary['missing_values'].values())}",
        f"- 重複列：{summary['duplicate_rows']}",
        "- 類別分布：" + "、".join(f"{k} {v}" for k, v in summary["class_counts"].items()),
        "",
        "## 數值欄位描述統計",
        "",
        frame[FEATURES].describe().round(3).to_markdown(),
        "",
        "## 各類別特徵平均",
        "",
        frame.groupby("species")[FEATURES].mean().round(3).to_markdown(),
        "",
        "## 相關係數矩陣",
        "",
        corr.round(3).to_markdown(),
        "",
        "## IQR 離群值數量（1.5 × IQR 準則）",
        "",
        pd.Series(summary["iqr_outliers"], name="outliers").to_frame().to_markdown(),
        "",
    ]
    (RESULT_DIR / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run_pca(frame: pd.DataFrame) -> dict:
    projection = compute_pca(frame, n_components=4)
    components = projection.coordinates
    explained = projection.explained_variance_ratio
    cumulative = projection.cumulative_variance_ratio

    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    for species in CLASS_NAMES:
        mask = frame["species"].to_numpy() == species
        ax.scatter(
            components[mask, 0],
            components[mask, 1],
            components[mask, 2],
            s=45,
            alpha=0.8,
            color=COLORS[species],
            label=species,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)", labelpad=8)
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)", labelpad=8)
    ax.set_zlabel(f"PC3 ({explained[2] * 100:.1f}%)", labelpad=8)
    ax.set_title("3D PCA feature distribution of the Iris dataset")
    ax.legend(title="Species", loc="upper left")
    ax.view_init(elev=22, azim=45)
    save_figure(fig, "pca_3d.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    indices = np.arange(1, 5)
    bars = ax.bar(indices, explained * 100, color="#4C72B0", alpha=0.85, label="Individual")
    ax.plot(indices, cumulative * 100, marker="o", color="#C44E52", label="Cumulative")
    for bar, value in zip(bars, explained * 100):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{value:.1f}%",
            ha="center",
            fontsize=9,
        )
    for index, value in zip(indices, cumulative * 100):
        ax.text(index + 0.08, value + 1.5, f"{value:.1f}%", color="#C44E52", fontsize=9)
    ax.set_xticks(indices, [f"PC{i}" for i in indices])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA explained variance")
    ax.legend()
    save_figure(fig, "pca_explained_variance.png")

    return {
        "explained_variance_ratio": explained.tolist(),
        "cumulative_variance_ratio": cumulative.tolist(),
        "components": projection.components.tolist(),
    }


def train_and_evaluate(frame: pd.DataFrame) -> dict:
    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame["species"].map({name: index for index, name in enumerate(CLASS_NAMES)}).to_numpy()

    split = split_and_scale(frame, test_size=0.30, random_state=SEED)
    x_train, x_test = split.x_train, split.x_test
    y_train, y_test = split.y_train, split.y_test
    x_train_scaled, x_test_scaled = x_train, x_test

    model = build_model()
    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line), expand_nested=True)
    (RESULT_DIR / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=12, min_lr=1e-5, verbose=0
        ),
    ]
    history = model.fit(
        x_train_scaled,
        y_train,
        validation_split=0.20,
        epochs=600,
        batch_size=16,
        callbacks=callbacks,
        shuffle=True,
        verbose=0,
    )
    history_frame = pd.DataFrame(history.history)
    history_frame.index.name = "epoch"
    history_frame.to_csv(RESULT_DIR / "training_history.csv")

    test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test, verbose=0)
    probabilities = model.predict(x_test_scaled, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    evaluation = evaluate_predictions(y_test, y_pred, probabilities)
    report_text = evaluation["classification_report_text"]
    report_dict = evaluation["classification_report"]
    (RESULT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

    cm = np.asarray(evaluation["confusion_matrix"])
    cm_frame = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_frame.to_csv(RESULT_DIR / "confusion_matrix.csv")

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_data = {}
    for index, species in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, index], probabilities[:, index])
        roc_data[species] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": evaluation["per_class_auc"][species],
        }
    macro_auc = evaluation["macro_auc"]

    best_epoch = int(np.argmin(history_frame["val_loss"].to_numpy())) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history_frame.index, history_frame["loss"], label="Train", color="#4C72B0")
    axes[0].plot(
        history_frame.index, history_frame["val_loss"], label="Validation", color="#C44E52"
    )
    axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
    axes[0].set_title("Loss during training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Categorical cross-entropy")
    axes[0].legend()

    axes[1].plot(
        history_frame.index, history_frame["accuracy"], label="Train", color="#4C72B0"
    )
    axes[1].plot(
        history_frame.index,
        history_frame["val_accuracy"],
        label="Validation",
        color="#C44E52",
    )
    axes[1].axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
    axes[1].set_title("Accuracy during training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.6, 1.02)
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, "training_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, matrix, title, fmt in [
        (axes[0], cm, "Confusion matrix (counts)", "d"),
        (axes[1], cm / cm.sum(axis=1, keepdims=True), "Confusion matrix (row-normalized)", ".2f"),
    ]:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            cbar=False,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    fig.tight_layout()
    save_figure(fig, "confusion_matrix.png")

    metrics_frame = pd.DataFrame(
        {name: report_dict[name] for name in CLASS_NAMES}
    ).T[["precision", "recall", "f1-score"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics_frame.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"], rot=0)
    ax.set_ylim(0, 1.12)
    ax.set_title("Test-set precision, recall and F1-score by class")
    ax.set_ylabel("Score")
    ax.set_xlabel("Species")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    ax.legend(loc="lower right", ncol=3)
    save_figure(fig, "per_class_metrics.png")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for index, species in enumerate(CLASS_NAMES):
        ax.plot(
            roc_data[species]["fpr"],
            roc_data[species]["tpr"],
            color=list(COLORS.values())[index],
            label=f"{species} (AUC = {roc_data[species]['auc']:.3f})",
            linewidth=2,
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves on the test set")
    ax.legend(loc="lower right")
    save_figure(fig, "roc_curves.png")

    cv_results = cross_validate(x, y)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    folds = [f"Fold {i}" for i in cv_results["fold"]]
    bars = ax.bar(folds, cv_results["accuracy"], color="#4C72B0", alpha=0.85)
    mean_accuracy = float(np.mean(cv_results["accuracy"]))
    std_accuracy = float(np.std(cv_results["accuracy"], ddof=1))
    ax.axhline(
        mean_accuracy,
        color="#C44E52",
        linestyle="--",
        label=f"Mean = {mean_accuracy:.3f} ± {std_accuracy:.3f}",
    )
    ax.set_ylim(0.8, 1.03)
    ax.set_ylabel("Accuracy")
    ax.set_title("5-fold stratified cross-validation accuracy")
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.legend(loc="lower right")
    save_figure(fig, "cross_validation.png")

    metrics = {
        "random_seed": SEED,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "epochs_trained": int(len(history_frame)),
        "best_epoch": best_epoch,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_auc": float(macro_auc),
        "per_class_auc": {k: v["auc"] for k, v in roc_data.items()},
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "cv_accuracy": [float(v) for v in cv_results["accuracy"]],
        "cv_mean_accuracy": mean_accuracy,
        "cv_std_accuracy": std_accuracy,
        "misclassified_indices_in_test": [
            int(i) for i in np.where(y_test != y_pred)[0]
        ],
        "test_true": y_test.tolist(),
        "test_pred": y_pred.tolist(),
        "test_probabilities": probabilities.tolist(),
    }
    if metrics["misclassified_indices_in_test"]:
        x_test_raw = split.scaler.inverse_transform(x_test)
        x_test_frame = pd.DataFrame(x_test_raw, columns=FEATURES)
        x_test_frame["true"] = [CLASS_NAMES[i] for i in y_test]
        x_test_frame["predicted"] = [CLASS_NAMES[i] for i in y_pred]
        x_test_frame["confidence"] = probabilities.max(axis=1).round(4)
        misclassified = x_test_frame.iloc[metrics["misclassified_indices_in_test"]]
        misclassified.to_csv(RESULT_DIR / "misclassified_samples.csv", index=False)

    model.save(MODEL_DIR / "iris_mlp.keras")
    (RESULT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def cross_validate(x: np.ndarray, y: np.ndarray) -> dict:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accuracies = []
    for fold, (train_index, valid_index) in enumerate(splitter.split(x, y), start=1):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_index])
        x_valid = scaler.transform(x[valid_index])
        model = build_model()
        model.fit(
            x_train,
            y[train_index],
            validation_split=0.15,
            epochs=400,
            batch_size=16,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=25, restore_best_weights=True, verbose=0
                )
            ],
            shuffle=True,
            verbose=0,
        )
        predictions = model.predict(x_valid, verbose=0).argmax(axis=1)
        accuracy = accuracy_score(y[valid_index], predictions)
        accuracies.append(float(accuracy))
        print(f"Cross-validation fold {fold}: accuracy = {accuracy:.4f}")
    frame = pd.DataFrame({"fold": list(range(1, 6)), "accuracy": accuracies})
    frame.to_csv(RESULT_DIR / "cross_validation.csv", index=False)
    return {"fold": frame["fold"].tolist(), "accuracy": accuracies}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_data(DATA_PATH)
    print(f"Loaded {frame.shape[0]} rows and {frame.shape[1]} columns")

    eda_summary = perform_eda(frame)
    pca_summary = run_pca(frame)
    metrics = train_and_evaluate(frame)

    combined = {
        "eda": eda_summary,
        "pca": pca_summary,
        "model": metrics,
    }
    (RESULT_DIR / "all_results.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Final results ===")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Macro one-vs-rest AUC: {metrics['macro_auc']:.4f}")
    print(
        f"5-fold CV accuracy: {metrics['cv_mean_accuracy']:.4f} "
        f"± {metrics['cv_std_accuracy']:.4f}"
    )
    print(f"Best epoch: {metrics['best_epoch']} (trained {metrics['epochs_trained']} epochs)")
    print(f"Misclassified test samples: {len(metrics['misclassified_indices_in_test'])}")


if __name__ == "__main__":
    main()
