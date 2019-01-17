import pytest

from os.path import dirname, join
import astropy.io.fits as fits
import numpy as np
from scipy.io import readsav
from scipy.signal import gaussian
from skimage import transform as tf

from pyreduce.clib.build_extract import build

build(verbose=True)

from pyreduce import extract
from pyreduce import util
from pyreduce.cwrappers import slitfunc

@pytest.fixture
def testdata1():
    folder = dirname(__file__)
    fname = "test.dat"
    return join(folder, fname)

@pytest.fixture
def testdata2():
    folder = dirname(__file__)
    fname = "test2.dat"
    return join(folder, fname)


@pytest.fixture
def testdata3():
    folder = dirname(__file__)
    fname = "test3.dat"
    return join(folder, fname)


@pytest.fixture
def testdata2_after():
    folder = dirname(__file__)
    fname = "test2_after.dat"
    return join(folder, fname)



def create_data(
    self,
    width,
    height,
    spec="linear",
    slitf="gaussian",
    noise=0,
    shear=0,
    ycen=None,
    oversample=1,
):
    if spec == "linear":
        spec = 5 + np.linspace(0, 5, width)
    elif spec == "sinus":
        spec = 5 + np.sin(np.linspace(0, 20 * np.pi, width))

    if slitf == "gaussian":
        slitf = gaussian(height * oversample, height / 8 * oversample)

    if ycen is None:
        ycen = np.zeros(width)

    img = spec[None, :] * slitf[:, None] 
    img += noise * np.random.randn(*img.shape)

    afine_tf = tf.AffineTransform(shear=-shear)
    img = tf.warp(img, inverse_map=afine_tf)

    # apply order curvature
    big_height = (int(np.ceil(np.max(ycen))) + height) * oversample
    big_height += oversample - (big_height % oversample)
    big_img = np.zeros((big_height, width))
    index = util.make_index(
        (oversample * ycen).astype(int) - height // 2 * oversample,
        (ycen * oversample).astype(int) + height // 2 * oversample - 1,
        0,
        width,
    )
    big_img[index] = img

    # downsample oversampling
    img = big_img[::oversample]
    for i in range(1, oversample):
        img += big_img[i::oversample]
    img /= oversample

    return img, spec, slitf

# def test_extend_orders(self):
#     # Test normal case
#     orders = np.array([[0.1, 5], [0.1, 7]])
#     extended = extract.extend_orders(orders, 10)

#     self.assertTrue(np.array_equal(orders, extended[1:-1]))
#     self.assertTrue(np.array_equal(extended[0], [0.1, 3]))
#     self.assertTrue(np.array_equal(extended[-1], [0.1, 9]))

#     # Test just one order
#     orders = np.array([0.1, 5], ndmin=2)
#     extended = extract.extend_orders(orders, 10)

#     self.assertTrue(np.array_equal(orders, extended[1:-1]))
#     self.assertTrue(np.array_equal(extended[0], [0, 0]))
#     self.assertTrue(np.array_equal(extended[-1], [0, 10]))

# def test_fix_column_range(self):
#     img = np.zeros((50, 1000))
#     orders = np.array([[0.2, 3], [0.2, 5], [0.2, 7], [0.2, 9]])
#     ew = np.array([[10, 10], [10, 10], [10, 10], [10, 10]])
#     cr = np.array([[0, 1000], [0, 1000], [0, 1000], [0, 1000]])

#     fixed = extract.fix_column_range(img, orders, ew, cr)

#     self.assertTrue(np.array_equal(fixed[1], [25, 175]))
#     self.assertTrue(np.array_equal(fixed[2], [15, 165]))
#     self.assertTrue(np.array_equal(fixed[0], fixed[1]))
#     self.assertTrue(np.array_equal(fixed[-1], fixed[-1]))

#     orders = np.array([[20],[20], [20]])
#     ew = np.array([[10, 10], [10, 10], [10, 10]])
#     cr = np.array([[0, 1000], [0, 1000], [0, 1000]])

#     fixed = extract.fix_column_range(img, orders, ew, np.copy(cr))
#     self.assertTrue(np.array_equal(fixed, cr))

# def test_arc_extraction(self):
#     img, spec, slitf = self.create_data(1000, 50, spec="linear")
#     orders = np.array([[49/2], [49/2], [49/2]])
#     extraction_width = np.array([[10, 10], [10, 10], [10, 10]])
#     column_range = np.array([[0, 1000], [0, 1000], [0, 1000]])
#     column_range = extract.fix_column_range(img, orders, extraction_width, column_range)

#     spec_out, _, unc_out = extract.arc_extraction(
#         img, orders, extraction_width, column_range
#     )

#     self.assertTrue(np.allclose(np.diff(spec_out / spec), 0))
#     self.assertTrue(np.allclose(np.diff(unc_out / spec_out), 0, atol = 1e-4))

# def test_optimal_extraction(self):
#     img, spec_in, slitf_in = self.create_data(1000, 50, spec="linear")
#     orders = np.array([[49/2], [49/2], [49/2], [49/2]])
#     extraction_width = np.array([[10, 10], [10, 10], [10, 10], [10, 10]])
#     column_range = np.array([[0, 1000], [0, 1000], [0, 1000], [0, 1000]])
#     column_range = extract.fix_column_range(img, orders, extraction_width, column_range)
#     scatter = [None for _ in range(3)]

#     spec_out, slitf_out, unc_out = extract.optimal_extraction(img, orders, extraction_width, column_range, scatter)

#     # Test for linear increase
#     self.assertTrue(np.allclose(np.diff(spec_out / spec_in), 0))
#     self.assertTrue(np.allclose(np.diff(unc_out / spec_out), 0, atol = 1e-4))

#     # Test slitfunction
#     slitf_cmp = gaussian(slitf_out.shape[1], 50/8)
#     slitf_cmp = np.interp(np.arange(0.5, 23.5, 1), np.arange(23), slitf_cmp)
#     slitf_cmp = slitf_cmp[None, :] * np.max(slitf_out, axis=1)[:, None]
#     self.assertTrue(np.allclose(slitf_out, slitf_cmp))

# def test_extract(self):
#     img, spec, slitf = self.create_data(1000, 50, spec="sinus")
#     orders = np.array([[49/2]])
#     shear = 0

#     spec_curved, sunc_curved = extract.extract(img, orders, shear=shear)
#     spec_vert, sunc_vert = extract.extract(img, orders)

#     self.assertTrue(np.allclose(spec_curved, spec_vert))
#     self.assertTrue(np.allclose(sunc_curved, sunc_vert))

# def test_extract_offset_order(self):
#     # The shear counters the rotation
#     width, height = 1000, 50
#     orders = np.array([[177 / width, 40]])
#     ycen = np.polyval(orders[0], np.arange(width))
#     img, spec, slitf = self.create_data(width, height, spec="linear", ycen=ycen, oversample=10)

#     # spec_out, sunc = extract.extract(img, orders, plot=True, swath_width=100)
#     xrange = (300, 700)
#     spec_out, _, _, _ = extract.extract_spectrum(
#         img,
#         ycen,
#         (height // 2, height // 2),
#         xrange,
#         swath_width=100,
#         plot=False,
#         osample=1,
#     )

#     m = np.diff(spec_out[xrange[0] : xrange[1]])
#     tmp = spec_out[xrange[0] : xrange[1]] / spec[xrange[0] : xrange[1]]

#     tmp2 = np.linspace(spec_out[xrange[0]], spec_out[xrange[1]-1], num=xrange[1] - xrange[0])
#     tmp2 /= spec_out[xrange[0]:xrange[1]]

#     plt.plot(m)
#     plt.show()

#     #plt.plot(np.arange(xrange[0], xrange[1]), tmp2)
#     plt.plot(spec)
#     plt.plot(spec_out, '-+')
#     plt.show()

#     self.assertTrue(np.allclose(m, np.median(m), atol=0.02))  # constant linear increase
#     self.assertTrue(
#         np.allclose(tmp, tmp[0], atol=0.01)
#     )  # shape same as input shape

def test_idl_data(testdata2_after, testdata3):
    sav = readsav(testdata2_after)
    sav2 = readsav(testdata3)

    orders = sav2["orcend"]
    onum = sav2["onum"]
    order = orders[onum, ::-1]
    ycen2 = np.polyval(order, np.arange(4096, dtype="float64"))

    ibeg, iend = sav["ib"], sav["ie"]
    img = sav["im"]
    swath_img = sav["sf"]
    ycen = sav["ycen"]
    yc = sav["yc"]
    yoffset = ycen[ibeg : iend + 1] - yc[ibeg : iend + 1]
    ylow, yhigh = sav["y_lower_lim"], sav["y_upper_lim"]
    yoffset2 = (ycen2 - np.floor(ycen2))[ibeg : iend + 1]

    index = util.make_index(
        ycen2.astype(int) - ylow, ycen2.astype(int) + yhigh, ibeg, iend + 1
    )
    swath_img2 = img[index]

    lambda_sf = sav["lambda_sf"]
    lambda_sp = sav["lambda_sp"]
    osample = sav["osample"]

    # self.assertTrue(np.allclose(swath_img, swath_img2))
    # self.assertTrue(np.allclose(yoffset, yoffset2))

    spec, slitf, model, unc, mask = slitfunc(
        swath_img2,
        yoffset2,
        lambda_sp=lambda_sp,
        lambda_sf=lambda_sf,
        osample=osample,
    )

    spec2 = np.sum(swath_img2, axis=0)
    
    extract.extract_spectrum(img, yoffset, (ylow, yhigh), (ibeg, iend))
