import pytest
import numpy as np

from pyreduce.util import load_fits
from pyreduce.make_shear import Curvature as CurvatureModule


def test_shear(files, wave, orders, instrument, mode, extension, mask, order_range):
    _, extracted = wave
    orders, column_range = orders

    files = files["curvature"][0]
    original, thead = load_fits(files, instrument, mode, extension, mask=mask)

    module = CurvatureModule(
        orders, column_range=column_range, order_range=order_range, plot=False
    )
    tilt, shear = module.execute(extracted, original)

    assert isinstance(tilt, np.ndarray)
    assert tilt.ndim == 2
    assert tilt.shape[0] == order_range[1] - order_range[0]
    assert tilt.shape[1] == extracted.shape[1]

    assert isinstance(shear, np.ndarray)
    assert shear.ndim == 2
    assert shear.shape[0] == order_range[1] - order_range[0]
    assert shear.shape[1] == extracted.shape[1]
