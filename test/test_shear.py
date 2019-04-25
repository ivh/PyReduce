import pytest
import numpy as np

from pyreduce.util import load_fits
from pyreduce.make_shear import make_shear


def test_shear(files, wave, orders, instrument, mode, extension, mask, order_range):
    _, extracted = wave
    orders, column_range = orders

    files = files["wave"][0]
    original, thead = load_fits(files, instrument, mode, extension, mask=mask)

    tilt, shear = make_shear(
        extracted,
        original,
        orders,
        column_range=column_range,
        order_range=order_range,
        plot=False,
    )

    assert isinstance(tilt, np.ndarray)
    assert tilt.ndim == 2
    assert tilt.shape[0] == order_range[1] - order_range[0]
    assert tilt.shape[1] == extracted.shape[1]

    assert isinstance(shear, np.ndarray)
    assert shear.ndim == 2
    assert shear.shape[0] == order_range[1] - order_range[0]
    assert shear.shape[1] == extracted.shape[1]

