import pytest
import numpy as np

from pyreduce.estimate_background_scatter import estimate_background_scatter


def test_scatter(flat, orders):
    img, _ = flat
    orders, column_range = orders

    scatter = estimate_background_scatter(
        img, orders, scatter_degree=4, column_range=column_range
    )

    assert isinstance(scatter, np.ndarray)
    assert scatter.ndim == 2
    assert scatter.shape[0] == 5
    assert scatter.shape[1] == 5


def test_simple():
    img = np.full((100, 100), 10.)
    orders = np.array([[0, 25], [0, 50], [0, 75]])

    scatter = estimate_background_scatter(img, orders, scatter_degree=0, plot=False)

    assert isinstance(scatter, np.ndarray)
    assert scatter.ndim == 2
    assert scatter.shape[0] == 1
    assert scatter.shape[1] == 1

    assert np.allclose(scatter[0, 0], 10.)


def test_scatter_degree():
    img = np.full((100, 100), 10.)
    orders = np.full((2, 2), 1.)

    estimate_background_scatter(img, orders, scatter_degree=0)

    with pytest.raises(ValueError):
        estimate_background_scatter(img, orders, scatter_degree=-1)

    with pytest.raises(TypeError):
        estimate_background_scatter(img, orders, scatter_degree=2.5)

    estimate_background_scatter(img, orders, scatter_degree=(2, 2))

    with pytest.raises(ValueError):
        estimate_background_scatter(img, orders, scatter_degree=(1,))

    with pytest.raises(ValueError):
        estimate_background_scatter(img, orders, scatter_degree=(3, 2, 1))

    with pytest.raises(ValueError):
        estimate_background_scatter(img, orders, scatter_degree=(2, -1))

    with pytest.raises(TypeError):
        estimate_background_scatter(img, orders, scatter_degree=(2, 2.5))
