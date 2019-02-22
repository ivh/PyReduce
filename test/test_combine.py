
import pytest

import astropy.io.fits as fits
import numpy as np
import tempfile
import os

from pyreduce import combine_frames


@pytest.fixture
def instrument():
    return "UVES"


@pytest.fixture
def mode():
    return "middle"


@pytest.fixture
def files():
    n = 3
    files = [
        tempfile.NamedTemporaryFile(suffix=".fits", delete=False) for _ in range(n)
    ]
    files = [f.name for f in files]
    yield files
    # Tear down code
    for f in files:
        os.remove(f)


def test_running_median():
    arr = np.array([np.arange(10), np.arange(1, 11), np.arange(2, 12)])
    size = 3

    result = combine_frames.running_median(arr, size)
    compare = np.array([np.arange(1, 9), np.arange(2, 10), np.arange(3, 11)])

    assert np.array_equal(result, compare)


def test_running_mean():
    arr = np.array([np.arange(10), np.arange(1, 11), np.arange(2, 12)])
    size = 3

    result = combine_frames.running_median(arr, size)
    compare = np.array([np.arange(1, 9), np.arange(2, 10), np.arange(3, 11)])

    assert np.array_equal(result, compare)


def test_combine_frames(files):
    img = np.full((100, 100), 10)
    ovscx = 5
    for f in files:
        data = img + np.random.randint(0, 20, size=img.shape)
        head = fits.Header(
            cards={
                "ESO DET OUT1 PRSCX": 0,
                "ESO DET OUT1 OVSCX": ovscx,
                "ESO DET OUT1 CONAD": 1,
                "ESO DET OUT1 RON": 0,
                "EXPTIME": 10,
                "RA": 100,
                "DEC": 51,
                "MJD-OBS": 12030,
            }
        )
        fits.writeto(f, data=data, header=head)

    combine, chead = combine_frames.combine_frames(files, "UVES", "middle", 0, window=5)

    assert combine.shape[0] == img.shape[0] - ovscx
    assert combine.shape[1] == img.shape[1]

    assert chead["exptime"] == 10 * len(files)

