import pytest
import numpy as np

from pyreduce.util import load_fits
from pyreduce.make_shear import make_shear


def test_shear(files, wave, orders, instrument, mode, extension, mask):
    _, extracted = wave
    orders, column_range = orders

    files = files["wave"][0]
    original, thead = load_fits(files, instrument, mode, extension, mask=mask)

    shear = make_shear(extracted, original, orders, column_range=column_range, plot=False)

    assert isinstance(shear, np.ndarray)
    assert shear.ndim == 2
    assert shear.shape[0] == len(orders)
    assert shear.shape[1] == extracted.shape[1]


