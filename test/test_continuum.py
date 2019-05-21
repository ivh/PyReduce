import pytest
import numpy as np

from pyreduce.continuum_normalization import splice_orders


def test_continuum(spec, wave, normflat, order_range):
    spec, sigma = spec
    norm, blaze = normflat
    wave, thar = wave

    spec, wave, blaze, sigma = splice_orders(
        spec, wave, blaze, sigma, scaling=True, plot=False
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
        == order_range[1] - order_range[0]
    )
    assert (
        spec.shape[1]
        == wave.shape[1]
        == blaze.shape[1]
        == sigma.shape[1]
        == norm.shape[1]
    )

