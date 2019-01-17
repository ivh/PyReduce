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
        bias=0,
        window=50,
        mask=mask,
    )

    assert isinstance(flat, np.ma.masked_array)
    assert isinstance(fhead, fits.Header)

    assert flat.ndim == fhead["NAXIS"]
    assert flat.shape[0] == fhead["NAXIS1"] - 100  # remove window from both sides
    assert flat.shape[1] == fhead["NAXIS2"]
