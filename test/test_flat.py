import pytest
from os.path import dirname, join

import numpy as np
from astropy.io import fits

from pyreduce import instruments
from pyreduce.combine_frames import combine_flat


def test_flat(instrument, mode, files, extension, mask):
    flat, fhead = combine_flat(
        files["flat"],
        instrument,
        mode,
        extension=extension,
        window=50,
        mask=mask,
    )

    assert isinstance(flat, np.ma.masked_array)
    assert isinstance(fhead, fits.Header)

    assert flat.ndim == fhead["NAXIS"]
    assert flat.shape[0] == fhead["NAXIS1"] - 100  # remove window from both sides
    assert flat.shape[1] == fhead["NAXIS2"]


def test_simple(tempfiles):
    n = 2
    files = tempfiles[:n]

    for i in range(n):
        data = np.full((100, 100), 5, dtype=float)
        fits.writeto(files[i], data)

    flat, fhead = combine_flat(files, "common", "", extension=0)

    assert isinstance(flat, np.ndarray)
    assert flat.shape[0] == 100
    assert flat.shape[1] == 100
    assert np.all(flat == 5 * n)

    n = 6
    files = tempfiles[:n]

    for i in range(n):
        data = np.full((200, 100), 5, dtype=float)
        fits.writeto(files[i], data, overwrite=True)

    flat, fhead = combine_flat(files, "common", "", extension=0, window=50)

    assert isinstance(flat, np.ndarray)
    assert flat.shape[0] == 200
    assert flat.shape[1] == 100
    assert np.all(flat == 5 * n)
