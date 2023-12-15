# -*- coding: utf-8 -*-
import numpy as np
import pytest

from pyreduce.clipnflip import clipnflip

# clipnflip(image, header, xrange=None, yrange=None, orientation=None)


@pytest.fixture
def image():
    image = np.arange(0, 200).reshape((20, 10))
    return image


@pytest.fixture
def header(image):
    # Default header is set up to change nothing
    # Note that image indices are y and x (and not x and y)
    header = {}
    header["e_amp"] = 1
    header["e_xlo"], header["e_xhi"] = 0, image.shape[1]
    header["e_ylo"], header["e_yhi"] = 0, image.shape[0]
    header["e_orient"] = 0
    header["e_transpose"] = False
    return header


def test_nochange(image, header):
    # This should change nothing
    flipped = clipnflip(image, header)

    assert isinstance(flipped, np.ndarray)
    assert flipped.shape[0] == image.shape[0]
    assert flipped.shape[1] == image.shape[1]
    assert np.all(flipped == image)


def test_only_rotate(image, header):
    for orient in [0, 1, 2, 3]:
        flipped = clipnflip(image, header, orientation=orient)

        compare = np.rot90(image, -1 * orient)

        assert isinstance(flipped, np.ndarray)
        assert flipped.shape[0] == compare.shape[0]
        assert flipped.shape[1] == compare.shape[1]
        assert np.all(flipped == compare)


def test_only_clip(image, header):
    # clip x direction
    flipped = clipnflip(image, header, xrange=(5, 8))

    assert isinstance(flipped, np.ndarray)
    assert flipped.shape[0] == 20
    assert flipped.shape[1] == 3
    assert np.all(flipped == image[:, 5:8])

    # clip y direction
    flipped = clipnflip(image, header, yrange=(10, 15))

    assert isinstance(flipped, np.ndarray)
    assert flipped.shape[0] == 5
    assert flipped.shape[1] == 10
    assert np.all(flipped == image[10:15, :])


def test_bad_clipping_range(image, header):
    # Case 1: x completely out of range
    with pytest.raises(IndexError):
        flipped = clipnflip(image, header, xrange=(100, 150))

    # Case 2: x upper limit out of range
    with pytest.raises(IndexError):
        flipped = clipnflip(image, header, xrange=(5, 150))

    # Case 3: x limits inverse
    with pytest.raises(IndexError):
        flipped = clipnflip(image, header, xrange=(10, 0))

    # Case 4: x limits only one column
    with pytest.raises(IndexError):
        flipped = clipnflip(image, header, xrange=(1, 1))

    # Case 1: y completely out of range
    with pytest.raises(IndexError):
        flipped = clipnflip(image, header, yrange=(100, 150))

    # Case 2: y upper limit out of range
    with pytest.raises(IndexError):
        flipped = clipnflip(image, header, yrange=(5, 150))

    # Case 3: y limits inverse
    with pytest.raises(IndexError):
        flipped = clipnflip(image, header, yrange=(10, 0))

    # Case 4: y limits only one column
    with pytest.raises(IndexError):
        flipped = clipnflip(image, header, yrange=(1, 1))


def test_multidimensional(image, header):
    image = image[None, None, ...]
    flipped = clipnflip(image, header)

    assert isinstance(flipped, np.ndarray)
    assert flipped.shape[0] == image.shape[-2]
    assert flipped.shape[1] == image.shape[-1]
    assert np.all(flipped == image[0, 0])
