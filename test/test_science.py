# -*- coding: utf-8 -*-
import numpy as np
import pytest

from pyreduce import util
from pyreduce.combine_frames import combine_calibrate
from pyreduce.extract import extract


def test_science(
    files,
    instr,
    instrument,
    mode,
    mask,
    bias,
    normflat,
    orders,
    settings,
    order_range,
):
    if len(files["science"]) == 0:
        pytest.skip(f"No science files found for instrument {instrument}")

    flat, blaze = normflat
    bias, bhead = bias
    orders, column_range = orders
    settings = settings["science"]

    # Fix column ranges
    for i in range(blaze.shape[0]):
        column_range[i] = np.where(blaze[i] != 0)[0][[0, -1]]

    f = files["science"][0]

    im, head = combine_calibrate(
        [f],
        instr,
        mode,
        mask=mask,
        bias=bias,
        bhead=bhead,
        norm=flat,
        bias_scaling=settings["bias_scaling"],
    )

    # Optimally extract science spectrum
    spec, sigma, _, _ = extract(
        im,
        orders,
        gain=head["e_gain"],
        readnoise=head["e_readn"],
        dark=head["e_drk"],
        column_range=column_range,
        order_range=order_range,
        extraction_type=settings["extraction_method"],
        extraction_width=settings["extraction_width"],
        lambda_sf=settings["smooth_slitfunction"],
        lambda_sp=settings["smooth_spectrum"],
        osample=settings["oversampling"],
        swath_width=settings["swath_width"],
        plot=False,
    )

    assert isinstance(spec, np.ma.masked_array)
    assert spec.ndim == 2
    assert spec.shape[0] == order_range[1] - order_range[0]
    assert spec.shape[1] == im.shape[1]
    assert np.issubdtype(spec.dtype, np.floating)
    assert not np.any(np.isnan(spec))
    assert not np.all(np.all(spec.mask, axis=0))

    assert isinstance(sigma, np.ma.masked_array)
    assert sigma.ndim == 2
    assert sigma.shape[0] == order_range[1] - order_range[0]
    assert sigma.shape[1] == im.shape[1]
    assert np.issubdtype(sigma.dtype, np.floating)
    assert not np.any(np.isnan(sigma))
    assert not np.all(np.all(sigma.mask, axis=0))
