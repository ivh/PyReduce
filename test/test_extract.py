import pytest

from os.path import dirname, join
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian

# from skimage import transform as tf

# from pyreduce.clib.build_extract import build

# build()

from pyreduce import extract
from pyreduce import util
from pyreduce.cwrappers import slitfunc


# @pytest.fixture
# def testdata1():
#     folder = dirname(__file__)
#     fname = "test.dat"
#     return join(folder, fname)


# @pytest.fixture
# def testdata2():
#     folder = dirname(__file__)
#     fname = "test2.dat"
#     return join(folder, fname)


# @pytest.fixture
# def testdata3():
#     folder = dirname(__file__)
#     fname = "test3.dat"
#     return join(folder, fname)


# @pytest.fixture
# def testdata2_after():
#     folder = dirname(__file__)
#     fname = "test2_after.dat"
#     return join(folder, fname)


@pytest.fixture
def width():
    return 100


@pytest.fixture
def height():
    return 50


@pytest.fixture(params=["linear", "sine"])
def spec(request, width):
    name = request.param

    if name == "linear":
        return 5 + np.linspace(0, 5, width)
    if name == "sine":
        return 5 + np.sin(np.linspace(0, 20 * np.pi, width))


@pytest.fixture(params=["gaussian"])
def slitf(request, height, oversample):
    name = request.param

    if name == "gaussian":
        y = gaussian(height * oversample, height / 8 * oversample)
        return y / np.sum(y)
    if name == "rectangular":
        x = np.arange(height * oversample)
        y = np.where(
            (x > height * oversample / 4) & (x < height * oversample * 3 / 4), 1, 0
        )
        return y / np.sum(y)


@pytest.fixture(params=[1, 2, 10], ids=["os=1", "os=2", "os=10"])
def oversample(request):
    return request.param


@pytest.fixture(params=[0], ids=["noise=0"])
def noise(request):
    return request.param


@pytest.fixture(params=["straight"], ids=["ycen=straight"])
def ycen(request, width, height):
    name = request.param
    if name == "straight":
        return np.full(width, (height - 1) / 2)

    if name == "linear":
        delta = 2
        return np.linspace(height / 2 - delta / 2, height / 2 + delta / 2, num=width)


@pytest.fixture
def orders(width, ycen):
    fit = np.polyfit(np.arange(width), ycen, deg=5)
    return np.atleast_2d(fit)
    # return np.array([[(height - 1) / 2]])


@pytest.fixture
def sample_data(width, height, spec, slitf, oversample, noise, ycen, tilt=0):
    img = spec[None, :] * slitf[:, None]
    img += noise * np.random.randn(*img.shape)

    # TODO more sophisticated sample data creation

    # afine_tf = tf.AffineTransform(shear=-shear)
    # img = tf.warp(img, inverse_map=afine_tf)

    # # apply order curvature
    # big_height = (int(np.ceil(np.max(ycen))) + height) * oversample
    # big_height += oversample - (big_height % oversample)
    # big_img = np.zeros((big_height, width))
    # index = util.make_index(
    #     (oversample * ycen).astype(int) - height // 2 * oversample,
    #     (ycen * oversample).astype(int) + height // 2 * oversample - 1,
    #     0,
    #     width,
    # )
    # big_img[index] = img

    # # downsample oversampling
    # img = big_img[::oversample]
    # for i in range(1, oversample):
    #     img += big_img[i::oversample]
    # img /= oversample

    return img, spec, slitf


def test_extend_orders():
    # Test normal case
    orders = np.array([[0.1, 5], [0.1, 7]])
    extended = extract.extend_orders(orders, 10)

    assert np.array_equal(orders, extended[1:-1])
    assert np.array_equal(extended[0], [0.1, 3])
    assert np.array_equal(extended[-1], [0.1, 9])

    # Test just one order
    orders = np.array([0.1, 5], ndmin=2)
    extended = extract.extend_orders(orders, 10)

    assert np.array_equal(orders, extended[1:-1])
    assert np.array_equal(extended[0], [0, 0])
    assert np.array_equal(extended[-1], [0, 10])


def test_fix_column_range():
    # Some orders will be shortened
    img = np.zeros((50, 1000))
    orders = np.array([[0.2, 3], [0.2, 5], [0.2, 7], [0.2, 9]])
    ew = np.array([[10, 10], [10, 10], [10, 10], [10, 10]])
    cr = np.array([[0, 1000], [0, 1000], [0, 1000], [0, 1000]])

    fixed = extract.fix_column_range(img, orders, ew, cr)

    assert np.array_equal(fixed[1], [25, 175])
    assert np.array_equal(fixed[2], [15, 165])
    assert np.array_equal(fixed[0], fixed[1])
    assert np.array_equal(fixed[-1], fixed[-1])

    # Nothing should change here
    orders = np.array([[20], [20], [20]])
    ew = np.array([[10, 10], [10, 10], [10, 10]])
    cr = np.array([[0, 1000], [0, 1000], [0, 1000]])

    fixed = extract.fix_column_range(img, orders, ew, np.copy(cr))
    assert np.array_equal(fixed, cr)


def test_arc_extraction(sample_data, orders, width, noise, oversample):
    img, spec, slitf = sample_data

    # orders = np.array([orders[0], orders[0], orders[0]])
    extraction_width = np.array([[10, 10]])
    column_range = np.array([[0, width]])
    # column_range = extract.fix_column_range(img, orders, extraction_width, column_range)

    spec_out, unc_out = extract.arc_extraction(
        img, orders, extraction_width, column_range
    )

    assert isinstance(spec_out, np.ndarray)
    assert spec_out.ndim == 2
    assert spec_out.shape[0] == 1
    assert spec_out.shape[1] == width

    assert isinstance(unc_out, np.ndarray)
    assert unc_out.ndim == 2
    assert unc_out.shape[0] == 1
    assert unc_out.shape[1] == width

    assert np.abs(np.diff(spec_out / spec)).max() < noise * 10 + 1e-8
    assert np.abs(np.diff(unc_out / spec_out)).max() < oversample / 5 + 1e-1


def test_vertical_extraction(sample_data, orders, width, height, noise, oversample):
    img, spec, slitf = sample_data

    spec_vert, sunc_vert, slitf_vert, _ = extract.extract(img, orders)

    assert isinstance(spec_vert, np.ma.masked_array)
    assert spec_vert.ndim == 2
    assert spec_vert.shape[0] == orders.shape[0]
    assert spec_vert.shape[1] == width

    assert isinstance(sunc_vert, np.ma.masked_array)
    assert sunc_vert.ndim == 2
    assert sunc_vert.shape[0] == orders.shape[0]
    assert sunc_vert.shape[1] == width

    assert isinstance(slitf_vert, np.ndarray)
    assert slitf_vert.ndim == 2
    assert slitf_vert.shape[0] == orders.shape[0]
    assert slitf_vert.shape[1] <= height * oversample

    assert not np.any(spec_vert == 0)
    assert np.abs(np.diff(spec / spec_vert[0])).max() <= noise + 1e-1

    assert not np.any(sunc_vert == 0)
    assert np.abs(sunc_vert / spec_vert).max() <= noise * 1.1 * oversample + 1e-2

    # cut = (height - slitf_vert.shape[1]) / 2
    # cut = (int(np.floor(cut)), int(np.ceil(cut)))
    # xnew = np.linspace(cut[0] + 0.5, cut[1] + 0.5, slitf_vert.shape[1])
    # cutout = np.interp(xnew, np.arange(0, height), slitf)
    # assert cutout.shape[0] == slitf_vert.shape[1]
    # assert cutout / slitf == 1


def test_curved_equal_vertical_extraction(sample_data, orders, noise):
    # Currently extract always uses the vertical extraction, making this kind of useless
    img, spec, slitf = sample_data
    tilt = 0
    shear = 0

    spec_curved, sunc_curved, slitf_curved, _ = extract.extract(
        img, orders, tilt=tilt, shear=shear
    )
    spec_vert, sunc_vert, slitf_vert, _ = extract.extract(img, orders)

    assert np.allclose(spec_curved, spec_vert, rtol=1e-2)
    # assert np.allclose(sunc_curved, sunc_vert, rtol=0.1)
    assert np.allclose(slitf_curved, slitf_vert, rtol=1e-1)
