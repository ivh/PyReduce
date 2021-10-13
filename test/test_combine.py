# -*- coding: utf-8 -*-

import astropy.io.fits as fits
import numpy as np
import pytest

from pyreduce import combine_frames


@pytest.fixture
def buffer():
    return np.array([np.arange(10), np.arange(1, 11), np.arange(2, 12)])


@pytest.fixture
def size():
    return 3


def create_file(file, nx=100, ny=100, ovscx=5):
    img = np.full((ny, nx), 10)
    data = img + np.random.randint(0, 20, size=img.shape)
    head = fits.Header(
        cards={
            "ESO DET OUT1 PRSCX": 0,
            "ESO DET OUT1 OVSCX": ovscx,
            "ESO DET OUT1 CONAD": 1,
            "ESO DET OUT1 RON": 0,
            "EXPTIME": 1,
            "RA": 100,
            "DEC": 51,
            "DATE-OBS": "2011-07-17T21:09:37.7545",
        }
    )
    fits.writeto(file, data=data, header=head)
    return data, head


def test_running_median(buffer, size):
    result = combine_frames.running_median(buffer, size)
    compare = np.array([np.arange(1, 9), np.arange(2, 10), np.arange(3, 11)])

    assert np.array_equal(result, compare)


def test_running_mean(buffer, size):
    result = combine_frames.running_median(buffer, size)
    compare = np.array([np.arange(1, 9), np.arange(2, 10), np.arange(3, 11)])

    assert np.array_equal(result, compare)


def test_calculate_probability(buffer):
    window = 1
    compare = np.array(
        [np.arange(1, 9), np.arange(2, 10), np.arange(3, 11)], dtype=float
    )

    result = combine_frames.calculate_probability(buffer, window, "sum")
    weights = np.sum(compare, axis=0)
    assert np.allclose(result, compare / weights)

    result = combine_frames.calculate_probability(buffer, window, "median")
    weights = np.mean(compare, axis=0)
    assert np.allclose(result, compare / weights)


def test_combine_frames(tempfiles):
    for f in tempfiles:
        create_file(f, 100, 100, 5)

    combine, chead = combine_frames.combine_frames(
        tempfiles, "UVES", "middle", 0, window=5
    )

    assert combine.shape[0] == 100 - 5
    assert combine.shape[1] == 100

    assert chead["exptime"] == len(tempfiles)


def test_nofiles():
    files = []
    with pytest.raises(ValueError):
        combine_frames.combine_frames(files, "UVES", "middle", 0, window=5)


def test_onefile(tempfiles):
    tempfiles = tempfiles[:1]
    nx, ny = 110, 100
    compare, head = create_file(tempfiles[0], nx, ny, 5)
    compare = np.rot90(compare, -1)[:-5]

    combine, chead = combine_frames.combine_frames(
        tempfiles, "UVES", "middle", 0, window=5
    )

    assert combine.shape[0] == nx - 5
    assert combine.shape[1] == ny
    assert np.allclose(combine, compare)

    for key, value in head.items():
        assert chead[key] == value


def test_twofiles(tempfiles):
    tempfiles = tempfiles[:2]

    for f in tempfiles:
        create_file(f, 100, 100, 5)

    combine, chead = combine_frames.combine_frames(
        tempfiles, "UVES", "middle", 0, window=5
    )

    assert combine.shape[0] == 100 - 5
    assert combine.shape[1] == 100

    assert chead["exptime"] == len(tempfiles)


def test_bad_window_size(tempfiles):
    for f in tempfiles:
        create_file(f, 100, 100, 5)

    combine, chead = combine_frames.combine_frames(
        tempfiles, "UVES", "middle", 0, window=80
    )
    assert combine.shape[0] == 100 - 5
    assert combine.shape[1] == 100

    assert chead["exptime"] == len(tempfiles)


def test_normal_orientation(tempfiles):
    for f in tempfiles:
        create_file(f, 100, 100, 0)

    combine, chead = combine_frames.combine_frames(
        tempfiles, "CRIRES_PLUS", "J1228_OPEN_det1", 0, window=10
    )
    assert combine.shape[0] == 100 - 10  # there is a 5 pixel cutoff on each side
    assert combine.shape[1] == 100 - 10  # there is a 5 pixel cutoff on each side

    assert chead["exptime"] == len(tempfiles)
