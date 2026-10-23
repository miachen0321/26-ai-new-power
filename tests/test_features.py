import numpy as np

from iris_lab.data import load_data
from iris_lab.features import compute_pca


def test_pca_returns_expected_shapes_and_variance() -> None:
    result = compute_pca(load_data())

    assert result.coordinates.shape == (150, 4)
    assert result.components.shape == (4, 4)
    assert np.isclose(result.explained_variance_ratio.sum(), 1.0)
    assert result.explained_variance_ratio[:3].sum() > 0.99
    assert 0.72 < result.explained_variance_ratio[0] < 0.73
    assert np.allclose(result.cumulative_variance_ratio[-1], 1.0)


def test_pca_supports_three_components() -> None:
    result = compute_pca(load_data(), n_components=3)

    assert result.coordinates.shape == (150, 3)
    assert result.components.shape == (3, 4)
    assert result.explained_variance_ratio.shape == (3,)
    assert result.cumulative_variance_ratio[-1] > 0.99


def test_pca_components_are_orthonormal() -> None:
    result = compute_pca(load_data())

    product = result.components @ result.components.T

    assert np.allclose(product, np.eye(4), atol=1e-7)
