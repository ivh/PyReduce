import pytest
from os.path import dirname, join

import numpy as np
from astropy.io import fits

from pyreduce import instruments
from pyreduce.combine_frames import combine_bias


def test_bias(instrument, mode, files, extension, mask):
    bias, bhead = combine_bias(
        files["bias"], instrument, mode, extension=extension, window=50, mask=mask
    )

    assert isinstance(bias, np.ma.masked_array)
    assert isinstance(bhead, fits.Header)

    assert bias.ndim == bhead["NAXIS"]
    assert bias.shape[0] == bhead["NAXIS1"] - 100  # remove window from both sides
    assert bias.shape[1] == bhead["NAXIS2"]


def test_only_one_file(instrument, mode, files, extension):
    files = [files["bias"][0]]
    bias, bhead = combine_bias(files, instrument, mode, extension=extension, window=50)

    assert isinstance(bias, np.ma.masked_array)
    assert isinstance(bhead, fits.Header)

    assert bias.ndim == bhead["NAXIS"]
    assert bias.shape[0] == bhead["NAXIS1"] - 100  # remove window from both sides
    assert bias.shape[1] == bhead["NAXIS2"]


def test_no_data_files():
    with pytest.raises(ValueError):
        combine_bias([], "", "")


def test_wrong_data_type():
    with pytest.raises(TypeError):
        combine_bias(None, "", "")

    with pytest.raises(ValueError):
        combine_bias([None], "", "")
