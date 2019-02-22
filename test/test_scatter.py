import pytest
import numpy as np

from pyreduce.estimate_background_scatter import estimate_background_scatter


def test_scatter(flat, orders):
    img, _ = flat
    orders, column_range = orders

    back, yback = estimate_background_scatter(
        img, orders, scatter_degree=4, column_range=column_range
    )

    assert isinstance(back, np.ndarray)
    assert back.ndim == 2
    assert back.shape[0] == len(orders) + 1
    assert back.shape[1] == img.shape[1]

    assert isinstance(yback, np.ndarray)
    assert yback.ndim == 2
    assert yback.shape[0] == len(orders) + 1
    assert yback.shape[1] == img.shape[1]


def test_simple():
    img = np.full((100, 100), 10.)
    orders = np.array([[0, 25], [0, 50], [0, 75]])

    back, yback = estimate_background_scatter(img, orders, scatter_degree=4, plot=False)

    assert np.allclose(back, 10)
    assert np.allclose(yback[0], 12)
    assert np.allclose(yback[1], 37)
    assert np.allclose(yback[2], 62)
    assert np.allclose(yback[3], 87)

def test_scatter_degree():
    img = np.full((100, 100), 10.)
    orders = np.full((2, 2), 1.)

    estimate_background_scatter(img, orders, scatter_degree=3)

    with pytest.raises(ValueError):
        estimate_background_scatter(img, orders, scatter_degree=0)

    with pytest.raises(TypeError):
        estimate_background_scatter(img, orders, scatter_degree=2.5)

    estimate_background_scatter(img, orders, scatter_degree=(2, 2))

    with pytest.raises(ValueError):
        estimate_background_scatter(img, orders, scatter_degree=(3, 2, 1))

    with pytest.raises(ValueError):
        estimate_background_scatter(img, orders, scatter_degree=(2, 0))

    with pytest.raises(TypeError):
        estimate_background_scatter(img, orders, scatter_degree=(2, 2.5))
