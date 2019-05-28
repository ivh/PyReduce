import pytest
import numpy as np

from pyreduce.normalize_flat import normalize_flat


def test_normflat(flat, orders, settings, order_range):
    flat, fhead = flat
    orders, column_range = orders
    settings = settings["norm_flat"]

    norm, blaze = normalize_flat(
        flat,
        orders,
        gain=fhead["e_gain"],
        readnoise=fhead["e_readn"],
        dark=fhead["e_drk"],
        column_range=column_range,
        order_range=order_range,
        extraction_width=settings["extraction_width"],
        degree=settings["scatter_degree"],
        threshold=settings["threshold"],
        lambda_sf=settings["smooth_slitfunction"],
        lambda_sp=settings["smooth_spectrum"],
        swath_width=settings["swath_width"],
        plot=False,
    )

    assert isinstance(norm, np.ndarray)
    assert norm.ndim == flat.ndim
    assert norm.shape[0] == flat.shape[0]
    assert norm.shape[1] == flat.shape[1]
    assert norm.dtype == flat.dtype
    assert np.ma.min(norm) > 0
    assert not np.any(np.isnan(norm))

    assert isinstance(blaze, np.ndarray)
    assert blaze.ndim == 2
    assert blaze.shape[0] == order_range[1] - order_range[0]
    assert blaze.shape[1] == flat.shape[1]
    assert np.issubdtype(blaze.dtype, np.floating)
    assert not np.any(np.isnan(blaze))

    for i, j in enumerate(range(order_range[0], order_range[1])):
        cr = column_range[j]
        assert np.all(blaze[i, : cr[0]].mask == True)
        assert np.all(blaze[i, cr[1] :].mask == True)
