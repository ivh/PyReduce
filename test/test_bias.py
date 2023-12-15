# -*- coding: utf-8 -*-
from os.path import dirname, join

import numpy as np
import pytest
from astropy.io import fits

from pyreduce import instruments
from pyreduce.combine_frames import combine_bias


def test_bias(instrument, mode, files, mask):
    if len(files["bias"]) == 0:
        pytest.skip(f"No bias files for instrument {instrument}")

    bias, bhead = combine_bias(files["bias"], instrument, mode, window=50, mask=mask)

    assert isinstance(bias, np.ma.masked_array)
    assert isinstance(bhead, fits.Header)

    assert bias.ndim == 2
    assert bias.shape[0] == mask.shape[0]
    assert bias.shape[1] == mask.shape[1]


def test_only_one_file(instrument, mode, files, mask):
    if len(files["bias"]) == 0:
        pytest.skip(f"No bias files for instrument {instrument}")

    files = [files["bias"][0]]
    bias, bhead = combine_bias(files, instrument, mode, window=50, mask=mask)

    assert isinstance(bias, np.ma.masked_array)
    assert isinstance(bhead, fits.Header)

    assert bias.ndim == 2
    assert bias.shape[0] == mask.shape[0]
    assert bias.shape[1] == mask.shape[1]


def test_no_data_files():
    with pytest.raises(FileNotFoundError):
        combine_bias([], "", "")


def test_wrong_data_type():
    with pytest.raises(TypeError):
        combine_bias(None, None, "")

    with pytest.raises(ValueError):
        combine_bias([None], None, "")


def test_simple_input(tempfiles):
    n = 2
    files = tempfiles[:n]

    for i in range(n):
        data = np.full((100, 100), i, dtype=float)
        fits.writeto(files[i], data)

    bias, bhead = combine_bias(files, "common", "", extension=0)

    assert isinstance(bias, np.ndarray)
    assert bias.shape[0] == 100
    assert bias.shape[1] == 100
    assert np.all(bias == sum(range(n)) / n)
