"""Feature transformations and PCA for the Iris dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from iris_lab.data import FEATURES, validate_data


@dataclass(frozen=True)
class PCAProjection:
    coordinates: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    components: np.ndarray


def compute_pca(frame: pd.DataFrame, n_components: int = 4) -> PCAProjection:
    validate_data(frame)
    if not 1 <= n_components <= len(FEATURES):
        raise ValueError(f"n_components must be between 1 and {len(FEATURES)}")

    scaled = StandardScaler().fit_transform(frame[FEATURES])
    pca = PCA(n_components=n_components, random_state=42)
    coordinates = pca.fit_transform(scaled)

    return PCAProjection(
        coordinates=coordinates,
        explained_variance_ratio=pca.explained_variance_ratio_,
        cumulative_variance_ratio=np.cumsum(pca.explained_variance_ratio_),
        components=pca.components_,
    )
