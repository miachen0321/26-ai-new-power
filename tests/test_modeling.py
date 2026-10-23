import keras
import numpy as np
import pytest

from iris_lab.data import load_data
from iris_lab.modeling import build_model, evaluate_predictions, split_and_scale


def test_model_architecture_matches_issue_requirements() -> None:
    model = build_model()

    assert model.input_shape == (None, 4)
    assert model.output_shape == (None, 3)
    assert model.count_params() == 243
    assert model.loss == "sparse_categorical_crossentropy"


def test_model_initial_weights_are_reproducible() -> None:
    keras.utils.set_random_seed(42)
    first = build_model()
    first_weights = [np.array(weight, copy=True) for weight in first.get_weights()]

    keras.utils.set_random_seed(42)
    second = build_model()
    second_weights = [np.array(weight, copy=True) for weight in second.get_weights()]

    assert len(first_weights) == len(second_weights)
    for left, right in zip(first_weights, second_weights):
        assert np.allclose(left, right)


def test_split_is_stratified_and_scaler_fits_training_data_only() -> None:
    split = split_and_scale(load_data())

    assert split.x_train.shape == (105, 4)
    assert split.x_test.shape == (45, 4)
    assert split.y_train.shape == (105,)
    assert split.y_test.shape == (45,)
    assert np.bincount(split.y_test).tolist() == [15, 15, 15]
    assert np.allclose(split.x_train.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(split.x_train.std(axis=0), 1.0, atol=1e-7)


def test_evaluate_perfect_predictions() -> None:
    y_true = np.array([0, 1, 2] * 3)
    probabilities = np.eye(3)[y_true]
    y_pred = probabilities.argmax(axis=1)

    metrics = evaluate_predictions(y_true, y_pred, probabilities)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_auc"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    assert metrics["per_class_auc"]["setosa"] == pytest.approx(1.0)


def test_evaluate_detects_misclassification() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 2, 1])
    probabilities = np.eye(3)[y_pred]

    metrics = evaluate_predictions(y_true, y_pred, probabilities)

    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    assert metrics["macro_auc"] == pytest.approx(0.5)
