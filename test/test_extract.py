# -*- coding: utf-8 -*-
from os.path import dirname, join

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import gaussian

from pyreduce import extract, util
from pyreduce.cwrappers import slitfunc

# from skimage import transform as tf

# from pyreduce.clib.build_extract import build

# build()


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


@pytest.fixture
def sample_data(width, height, spec, slitf, oversample, ycen, tilt=0):
    img = spec[None, :] * slitf[:, None]
    # TODO more sophisticated sample data creation
    out = np.zeros((height, width))
    for i in range(width):
        out[:, i] = np.interp(
            np.arange(height), np.linspace(0, height, height * oversample), img[:, i]
        )

    return out, spec, slitf


def test_class_swath():
    swath = extract.Swath(5)

    assert len(swath) == 5
    assert len(swath.spec) == 5
    assert len(swath.slitf) == 5
    assert len(swath.model) == 5
    assert len(swath.unc) == 5
    assert len(swath.mask) == 5
    assert len(swath.info) == 5

    for i in range(5):
        assert swath.spec[i] is None
        assert swath.slitf[i] is None
        assert swath.model[i] is None
        assert swath.unc[i] is None
        assert swath.mask[i] is None
        assert swath.info[i] is None

        tmp = swath[i]
        for j in range(5):
            assert tmp[j] is None


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
    nrow, ncol = 50, 1000
    orders = np.array([[0.2, 3], [0.2, 5], [0.2, 7], [0.2, 9]])
    ew = np.array([[10, 10], [10, 10], [10, 10], [10, 10]])
    cr = np.array([[0, 1000], [0, 1000], [0, 1000], [0, 1000]])

    fixed = extract.fix_column_range(cr, orders, ew, nrow, ncol)

    assert np.array_equal(fixed[1], [25, 175])
    assert np.array_equal(fixed[2], [15, 165])
    assert np.array_equal(fixed[0], fixed[1])
    assert np.array_equal(fixed[-1], fixed[-1])

    # Nothing should change here
    orders = np.array([[20], [20], [20]])
    ew = np.array([[10, 10], [10, 10], [10, 10]])
    cr = np.array([[0, 1000], [0, 1000], [0, 1000]])

    fixed = extract.fix_column_range(np.copy(cr), orders, ew, nrow, ncol)
    assert np.array_equal(fixed, cr)


def test_make_bins(width):
    # swath_width, xlow, xhigh, ycen, ncol
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


def test_fix_parameters():
    orders = [[0, 0, 50]]
    ncol, nrow, nord = 100, 100, 1

    # Everything None, i.e. most default settings
    for xwd in [None, 0.2, (4, 4), [[10, 10]]]:
        for cr in [None, (1, 90), [[4, 100]]]:
            xwd, cr, orders = extract.fix_parameters(xwd, cr, orders, ncol, nrow, nord)
            assert isinstance(xwd, np.ndarray)
            assert isinstance(cr, np.ndarray)
            assert isinstance(orders, np.ndarray)

            assert xwd.ndim == 2
            assert xwd.shape[0] == nord
            assert xwd.shape[1] == 2
            assert cr.ndim == 2
            assert cr.shape[0] == nord
            assert cr.shape[1] == 2
            assert orders.ndim == 2
            assert orders.shape[0] == nord
            assert orders.shape[1] == 3

    with pytest.raises(ValueError):
        extract.fix_parameters(100, None, orders, ncol, nrow, nord)


def test_arc_extraction(sample_data, orders, width, oversample):
    img, spec, slitf = sample_data

    extraction_width = np.array([[10, 10]])
    column_range = np.array([[0, width]])

    nord = len(orders)
    tilt = np.zeros((nord, width))
    shear = np.zeros((nord, width))

    spec_out, unc_out = extract.arc_extraction(
        img, orders, extraction_width, column_range, tilt=tilt, shear=shear
    )

    assert isinstance(spec_out, np.ndarray)
    assert spec_out.ndim == 2
    assert spec_out.shape[0] == 1
    assert spec_out.shape[1] == width

    assert isinstance(unc_out, np.ndarray)
    assert unc_out.ndim == 2
    assert unc_out.shape[0] == 1
    assert unc_out.shape[1] == width

    assert np.abs(np.diff(spec_out / spec)).max() < 1e-8
    assert np.abs(np.diff(unc_out / spec_out)).max() < oversample / 5 + 1e-1


def test_vertical_extraction(sample_data, orders, width, height, oversample):
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

    assert isinstance(slitf_vert, list)
    assert len(slitf_vert) == orders.shape[0]
    assert len(slitf_vert[0]) <= height * oversample

    assert not np.any(spec_vert == 0)
    assert np.abs(np.diff(spec / spec_vert[0])).max() <= 1e-1

    assert not np.any(sunc_vert == 0)
    # assert np.abs(sunc_vert / spec_vert).max() <= 1e-2


def test_curved_equal_vertical_extraction(sample_data, orders):
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
    xwd = np.array([[height // 2, height // 2]])
    cr = np.array([[0, width]])
    tilt = shear = np.zeros((1, width))

    res_spec, res_slitf, res_unc = extract.optimal_extraction(
        img, orders, xwd, cr, tilt, shear
    )

    assert isinstance(res_spec, np.ndarray)
    assert isinstance(res_slitf, list)
    assert isinstance(res_unc, np.ndarray)

    assert res_spec.ndim == 2
    assert res_spec.shape[0] == 1
    assert res_spec.shape[1] == width
    assert not np.any(np.isnan(res_spec))

    assert res_unc.ndim == 2
    assert res_unc.shape[0] == 1
    assert res_unc.shape[1] == width

    assert len(res_slitf) == 1
    assert len(res_slitf[0]) != 0


def test_extract_spectrum(sample_data, orders, ycen, width, height):
    img, spec, slitf = sample_data

    column_range = np.array([[20, width]])
    extraction_width = np.array([[10, 10]])

    yrange = extract.get_y_scale(ycen, column_range[0], extraction_width[0], height)
    xrange = column_range[0]

    out_spec = np.zeros(width)
    out_sunc = np.zeros(width)
    out_slitf = np.zeros(10 + 10 + 2 + 1)
    out_mask = np.zeros(width)

    extract.extract_spectrum(
        np.copy(img),
        np.copy(ycen),
        np.copy(yrange),
        np.copy(xrange),
        out_spec=out_spec,
        out_sunc=out_sunc,
        out_slitf=out_slitf,
        out_mask=out_mask,
    )

    assert np.any(out_spec != 0)
    assert np.any(out_sunc != 0)
    assert np.any(out_slitf != 0)
    assert np.any(out_mask)

    spec, slitf, mask, sunc = extract.extract_spectrum(
        np.copy(img), np.copy(ycen), np.copy(yrange), np.copy(xrange)
    )

    assert np.array_equal(out_spec, spec)
    assert np.array_equal(out_sunc, sunc)
    assert np.array_equal(out_slitf, slitf)
    assert np.array_equal(out_mask, mask)


def test_get_y_scale(ycen, height, width):
    xrange = (0, width)
    xwd = (10, 10)
    y_lower_lim, y_upper_lim = extract.get_y_scale(ycen, xrange, xwd, height)

    assert isinstance(y_lower_lim, int)
    assert isinstance(y_upper_lim, int)
    assert y_lower_lim >= 0
    assert y_upper_lim < height

    xwd = (2 * height, 2 * height)
    y_lower_lim, y_upper_lim = extract.get_y_scale(ycen, xrange, xwd, height)
    assert isinstance(y_lower_lim, int)
    assert isinstance(y_upper_lim, int)
    assert y_lower_lim >= 0
    assert y_upper_lim < height

    ycen_tmp = ycen + height
    xwd = (10, 10)
    y_lower_lim, y_upper_lim = extract.get_y_scale(ycen_tmp, xrange, xwd, height)
    assert isinstance(y_lower_lim, int)
    assert isinstance(y_upper_lim, int)
    assert y_lower_lim >= 0
    assert y_upper_lim < height


def test_extract(sample_data, orders):
    img, spec, slitf = sample_data

    with pytest.raises(ValueError):
        extract.extract(img, orders, extraction_type="foobar")
