"""Generate frontend golden cases from the trained Python models.

Fixed 3 raw-argument inputs per model are run through the real training
artifacts (models/iris_mlp.keras and the deterministic AI4I Logistic
Regression pipeline, seed 42) and expectations are saved to tests/golden.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from sklearn.model_selection import train_test_split

import keras

from iris_lab.data import CLASS_NAMES, load_data
from iris_lab.modeling import build_model, split_and_scale
from scripts.run_ai4i_analysis import (
    build_model_frame,
    build_models,
    engineer_features,
)

OUT_PATH = ROOT / "tests" / "golden.json"

IRIS_INPUTS = [
    [5.1, 3.5, 1.4, 0.2],
    [6.1, 2.9, 4.7, 1.4],
    [6.5, 3.0, 5.2, 2.0],
]
AI4I_INPUTS = [
    {"air_temperature": 298.1, "process_temperature": 308.6, "rotational_speed": 1551, "torque": 42.8, "tool_wear": 0, "type": "L"},
    {"air_temperature": 298.6, "process_temperature": 308.7, "rotational_speed": 1408, "torque": 46.3, "tool_wear": 0, "type": "M"},
    {"air_temperature": 299.2, "process_temperature": 310.8, "rotational_speed": 1360, "torque": 65.7, "tool_wear": 200, "type": "L"},
]


def iris_golden() -> list[dict]:
    keras.utils.set_random_seed(42)
    frame = load_data(Path("data/iris.csv"))
    split = split_and_scale(frame, test_size=0.30, random_state=42)
    model = build_model()
    model.load_weights("models/iris_mlp.keras")
    scaled = split.scaler.transform(np.asarray(IRIS_INPUTS, dtype=float))
    probabilities = model.predict(scaled, verbose=0)
    cases = []
    for raw, probs in zip(IRIS_INPUTS, probabilities):
        index = int(np.argmax(probs))
        cases.append(
            {
                "inputs": dict(zip(["sepal_length", "sepal_width", "petal_length", "petal_width"], raw)),
                "expected_class": CLASS_NAMES[index],
                "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
            }
        )
    return cases


def ai4i_golden() -> list[dict]:
    enriched = engineer_features(pd.read_csv("data/ai4i2020.csv"))
    x, y = build_model_frame(enriched)
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.30, stratify=y, random_state=42)
    pipeline = build_models()["logistic_regression"]
    pipeline.fit(x_train, y_train)
    cases = []
    for row in AI4I_INPUTS:
        probe = engineer_features(pd.DataFrame([row]))
        probability = float(pipeline.predict_proba(probe)[0, 1])
        cases.append(
            {
                "inputs": row,
                "expected_failure": int(probability >= 0.5),
                "probabilities": {"failure": probability, "ok": 1.0 - probability},
            }
        )
    return cases


def main() -> None:
    golden = {
        "description": "3 fixed inputs per model; expectations from real Python artifacts (seed 42)",
        "tolerance": 1e-6,
        "iris": iris_golden(),
        "ai4i": ai4i_golden(),
    }
    OUT_PATH.write_text(json.dumps(golden, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    for name, cases in [("iris", golden["iris"]), ("ai4i", golden["ai4i"])]:
        for index, case in enumerate(cases):
            print(f"{name}[{index}]: {case['expected_class' if name == 'iris' else 'expected_failure']} {[round(v, 6) for v in case['probabilities'].values()]}")


if __name__ == "__main__":
    main()
