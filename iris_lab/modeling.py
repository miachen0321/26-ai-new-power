"""Neural-network modelling utilities for Iris classification."""

from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

from iris_lab.data import CLASS_NAMES, FEATURES, validate_data


@dataclass(frozen=True)
class DataSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def build_model(input_dim: int = 4, num_classes: int = 3) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,), name="features"),
            layers.Dense(16, activation="relu", name="hidden_1"),
            layers.Dense(8, activation="relu", name="hidden_2"),
            layers.Dense(num_classes, activation="softmax", name="class_probabilities"),
        ],
        name="iris_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def split_and_scale(
    frame: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 42,
) -> DataSplit:
    validate_data(frame)
    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame["species"].map({name: index for index, name in enumerate(CLASS_NAMES)}).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return DataSplit(
        x_train=x_train_scaled,
        x_test=x_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    names = class_names or CLASS_NAMES
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities)

    report_text = classification_report(
        y_true, y_pred, target_names=names, digits=4, zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        target_names=names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    y_true_binary = label_binarize(y_true, classes=list(range(len(names))))
    per_class_auc = {}
    for index, name in enumerate(names):
        if y_true_binary[:, index].sum() == 0:
            per_class_auc[name] = float("nan")
        else:
            per_class_auc[name] = float(
                roc_auc_score(y_true_binary[:, index], probabilities[:, index])
            )

    macro_auc = float(
        roc_auc_score(
            y_true_binary,
            probabilities,
            average="macro",
            multi_class="ovr",
        )
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "classification_report": report,
        "classification_report_text": report_text,
        "confusion_matrix": matrix.tolist(),
    }
