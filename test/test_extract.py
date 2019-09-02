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
    return img, spec, slitf


def test_class_swath():
    swath = extract.Swath(5)

    assert len(swath) == 5
    assert len(swath.spec) == 5
    assert len(swath.slitf) == 5
    assert len(swath.model) == 5
    assert len(swath.unc) == 5
    assert len(swath.mask) == 5

    for i in range(5):
        assert swath.spec[i] is None
        assert swath.slitf[i] is None
        assert swath.model[i] is None
        assert swath.unc[i] is None
        assert swath.mask[i] is None

def test_make_bins(width):
    #swath_width, xlow, xhigh, ycen, ncol
    xlow, xhigh = 0, width
    ycen = np.linspace(0, 10, width)
    swath_width = width // 10
    nbin, bins_start, bins_end = extract.make_bins(swath_width, xlow, xhigh, ycen)

    assert isinstance(nbin, (int, np.integer))
    assert nbin == 10
    assert isinstance(bins_start, np.ndarray)
    assert isinstance(bins_end, np.ndarray)
    assert len(bins_start) == 2 * nbin - 1
    assert len(bins_end) == 2 * nbin - 1

    nbin, bins_start, bins_end = extract.make_bins(None, xlow, xhigh, ycen)

    assert isinstance(nbin, (int, np.integer))
    assert isinstance(bins_start, np.ndarray)
    assert isinstance(bins_end, np.ndarray)
    assert len(bins_start) == 2 * nbin - 1
    assert len(bins_end) == 2 * nbin - 1

    nbin, bins_start, bins_end = extract.make_bins(width * 2, xlow, xhigh, ycen)

    assert nbin == 1
    assert len(bins_start) == 1
    assert len(bins_end) == 1
    assert bins_start[0] == 0
    assert bins_end[0] == width

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

def test_optimal_extraction(sample_data, orders, height, width):
    img, spec, slitf = sample_data
    xwd = np.array([[-height//2, height//2]])
    cr = np.array([[0, width]])
    tilt = shear = np.zeros((1, width))

    res_spec, res_slitf, res_unc = extract.optimal_extraction(img, orders, xwd, cr, tilt, shear)

    assert isinstance(res_spec, np.ndarray)
    assert isinstance(res_slitf, list)
    assert isinstance(res_unc, np.ndarray)

    assert res_spec.ndim == 2
    assert res_spec.shape[0] == 1
    assert res_spec.shape[1] == width

    assert res_unc.ndim == 2
    assert res_unc.shape[0] == 1
    assert res_unc.shape[1] == width

    assert len(res_slitf) == 1
    assert len(res_slitf[0]) != 0

