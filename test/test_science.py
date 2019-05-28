import pytest
import numpy as np

from pyreduce import util
from pyreduce.extract import extract


def test_science(
    files,
    instrument,
    mode,
    mask,
    extension,
    bias,
    normflat,
    orders,
    settings,
    order_range,
):
    flat, blaze = normflat
    bias, _ = bias
    orders, column_range = orders
    settings = settings["science"]

    # Fix column ranges
    for i in range(blaze.shape[0]):
        column_range[i] = np.where(blaze[i] != 0)[0][[0, -1]]

    f = files["science"][0]

    im, head = util.load_fits(
        f, instrument, mode, extension, mask=mask, dtype=np.float32
    )
    # Correct for bias and flat field
    im -= bias
    im /= flat

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
    assert spec.shape[1] == bias.shape[1]
    assert np.issubdtype(spec.dtype, np.floating)
    assert not np.any(np.isnan(spec))
    assert not np.all(np.all(spec.mask, axis=0))

    assert isinstance(sigma, np.ma.masked_array)
    assert sigma.ndim == 2
    assert sigma.shape[0] == order_range[1] - order_range[0]
    assert sigma.shape[1] == bias.shape[1]
    assert np.issubdtype(sigma.dtype, np.floating)
    assert not np.any(np.isnan(sigma))
    assert not np.all(np.all(sigma.mask, axis=0))
