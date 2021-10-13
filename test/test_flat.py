# -*- coding: utf-8 -*-
from os.path import dirname, join

import numpy as np
import pytest
from astropy.io import fits

from pyreduce import instruments
from pyreduce.combine_frames import combine_calibrate


def test_flat(instrument, mode, files, mask):
    if len(files["flat"]) == 0:
        pytest.skip(f"No flat files for instrument {instrument}")

    flat, fhead = combine_calibrate(
        files["flat"], instrument, mode, mask=mask, window=50
    )

    assert isinstance(flat, np.ma.masked_array)
    assert isinstance(fhead, fits.Header)

    assert flat.ndim == 2
    assert flat.shape[0] > 1
    assert flat.shape[1] > 1


def test_flat_with_bias(instrument, mode, files, mask, bias):
    if len(files["flat"]) == 0:
        pytest.skip(f"No flat files for instrument {instrument}")

    window = 50
    bias, bhead = bias
    flat, fhead = combine_calibrate(
        files["flat"],
        instrument,
        mode,
        window=window,
        bias=bias,
        bhead=bhead,
        mask=mask,
    )

    assert isinstance(flat, np.ma.masked_array)
    assert isinstance(fhead, fits.Header)

    assert flat.ndim == 2
    assert flat.shape[0] > 1
    assert flat.shape[1] > 1


def test_simple(tempfiles):
    n = 2
    files = tempfiles[:n]

    for i in range(n):
        data = np.full((100, 100), 5, dtype=float)
        fits.writeto(files[i], data)

    flat, fhead = combine_calibrate(files, "common", "", extension=0)

    assert isinstance(flat, np.ndarray)
    assert flat.shape[0] == 100
    assert flat.shape[1] == 100
    assert np.all(flat == 5 * n)

    n = 6
    files = tempfiles[:n]

    for i in range(n):
        data = np.full((200, 100), 5, dtype=float)
        fits.writeto(files[i], data, overwrite=True)

    flat, fhead = combine_calibrate(files, "common", "", extension=0, window=50)

    assert isinstance(flat, np.ndarray)
    assert flat.shape[0] == 200
    assert flat.shape[1] == 100
    assert np.all(flat == 5 * n)
