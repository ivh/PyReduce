# -*- coding: utf-8 -*-
import numpy as np
import pytest

from pyreduce.continuum_normalization import continuum_normalize, splice_orders


@pytest.fixture
def spliced(spec, wave, normflat):
    spec, sigma = spec
    _, blaze = normflat
    wave = wave

    if wave is None:
        return None, None, None, None

    spec, wave, blaze, sigma = splice_orders(
        spec, wave, blaze, sigma, scaling=True, plot=False
    )
    return spec, wave, blaze, sigma


def test_splice(spec, wave, normflat, order_range):
    spec, sigma = spec
    norm, blaze = normflat
    wave = wave

    if wave is None:
        pytest.skip("Need wavecal for splice")

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


def test_continuum(spliced):
    spec, wave, cont, sigm = spliced

    if wave is None:
        pytest.skip("Need wavecal for continuum")

    new = continuum_normalize(
        spec,
        wave,
        cont,
        sigm,
        iterations=1,
        plot=False,
    )

    assert isinstance(new, np.ndarray)
    assert new.ndim == 2
    assert new.shape[0] == spec.shape[0]
    assert new.shape[1] == spec.shape[1]
