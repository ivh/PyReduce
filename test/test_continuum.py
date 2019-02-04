import pytest
import numpy as np

from pyreduce.continuum_normalization import splice_orders


def test_continuum(spec, orders, wave, normflat, order_range):
    orders, column_range = orders
    spec, sigma = spec
    norm, blaze = normflat

    # fix column ranges
    for i in range(spec.shape[0]):
        column_range[i] = np.where(spec[i] != 0)[0][[0, -1]] + [0, 1]

    spec, wave, blaze, sigma = splice_orders(
        spec,
        wave,
        blaze,
        sigma,
        column_range=column_range,
        order_range=order_range,
        scaling=True,
        plot=False,
    )

    assert isinstance(spec, np.ma.masked_array)
    assert isinstance(wave, np.ma.masked_array)
    assert isinstance(blaze, np.ma.masked_array)
    assert isinstance(sigma, np.ma.masked_array)

    assert spec.ndim == wave.ndim == blaze.ndim == sigma.ndim == 2
    assert (
        spec.shape[0]
        == wave.shape[0]
        == blaze.shape[0]
        == sigma.shape[0]
        == order_range[1] - order_range[0] + 1
    )
    assert (
        spec.shape[1]
        == wave.shape[1]
        == blaze.shape[1]
        == sigma.shape[1]
        == norm.shape[1]
    )

