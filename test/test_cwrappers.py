import numpy as np
import pytest

from pyreduce.cwrappers import slitfunc, slitfunc_curved

pytestmark = pytest.mark.unit


def test_slitfunc_vert():
    img = np.full((100, 100), 1, dtype=float)
    ycen = np.full(100, 0, dtype=float)
    lambda_sp = 0
    lambda_sf = 0.1
    osample = 1

    # Run it once the way it is supposed to
    slitfunc(img, ycen, lambda_sp, lambda_sf, osample)

    # Then try different incompatible inputs, which have to be caught before going to the C code
    with pytest.raises(AssertionError):
        slitfunc(None, ycen, lambda_sp, lambda_sf, osample)
    with pytest.raises(ValueError):
        slitfunc("bla", ycen, lambda_sp, lambda_sf, osample)

    with pytest.raises(AssertionError):
        slitfunc(img, None, lambda_sp, lambda_sf, osample)
    with pytest.raises(ValueError):
        slitfunc(img, "blub", lambda_sp, lambda_sf, osample)

    with pytest.raises(TypeError):
        slitfunc(img, ycen, None, lambda_sf, osample)
    with pytest.raises(ValueError):
        slitfunc(img, ycen, "bla", lambda_sf, osample)
    with pytest.raises(TypeError):
        slitfunc(img, ycen, lambda_sp, None, osample)
    with pytest.raises(ValueError):
        slitfunc(img, ycen, lambda_sp, "bla", osample)
    with pytest.raises(TypeError):
        slitfunc(img, ycen, lambda_sp, lambda_sf, None)
    with pytest.raises(ValueError):
        slitfunc(img, ycen, lambda_sp, lambda_sf, "bla")

    # Then try different sizes for img and ycen
    with pytest.raises(AssertionError):
        ycen_bad = np.full(50, 1, dtype=float)
        slitfunc(img, ycen_bad, lambda_sp, lambda_sf, osample)

    with pytest.raises(AssertionError):
        slitfunc(img, ycen, lambda_sp, lambda_sf, osample=0)
    with pytest.raises(AssertionError):
        slitfunc(img, ycen, lambda_sp, -1, osample)
    with pytest.raises(AssertionError):
        slitfunc(img, ycen, -1, lambda_sf, osample)


def test_slitfunc_curved():
    img = np.full((100, 100), 1)
    ycen = np.full(100, 50)
    p1 = np.full(100, 0)
    p2 = np.full(100, 0)
    lambda_sp = 0
    lambda_sf = 0.1
    osample = 1
    yrange = (49, 50)

    # Run it once the way it is supposed to
    slitfunc_curved(img, ycen, p1, p2, lambda_sp, lambda_sf, osample, yrange)
    slitfunc_curved(img, ycen, 1, 0.01, lambda_sp, lambda_sf, osample, yrange)

    # Then try different incompatible inputs, which have to be caught before going to the C code
    with pytest.raises(AssertionError):
        slitfunc_curved(None, ycen, p1, p2, lambda_sp, lambda_sf, osample, yrange)
    with pytest.raises(ValueError):
        slitfunc_curved("bla", ycen, p1, p2, lambda_sp, lambda_sf, osample, yrange)

    with pytest.raises(AssertionError):
        slitfunc_curved(img, None, p1, p2, lambda_sp, lambda_sf, osample, yrange)
    with pytest.raises(ValueError):
        slitfunc_curved(img, "blub", p1, p2, lambda_sp, lambda_sf, osample, yrange)

    with pytest.raises(AssertionError):
        slitfunc_curved(img, ycen, p1, None, lambda_sp, lambda_sf, osample, yrange)
    with pytest.raises(ValueError):
        slitfunc_curved(img, ycen, p1, "boo", lambda_sp, lambda_sf, osample, yrange)

    with pytest.raises(TypeError):
        slitfunc_curved(img, ycen, p1, p2, None, lambda_sf, osample, yrange)
    with pytest.raises(ValueError):
        slitfunc_curved(img, ycen, p1, p2, "bla", lambda_sf, osample, yrange)
    with pytest.raises(TypeError):
        slitfunc_curved(img, ycen, p1, p2, lambda_sp, None, osample, yrange)
    with pytest.raises(ValueError):
        slitfunc_curved(img, ycen, p1, p2, lambda_sp, "bla", osample, yrange)
    with pytest.raises(TypeError):
        slitfunc_curved(img, ycen, p1, p2, lambda_sp, lambda_sf, None, yrange)
    with pytest.raises(ValueError):
        slitfunc_curved(img, ycen, p1, p2, lambda_sp, lambda_sf, "bla", yrange)

    # Then try different sizes for img and ycen
    with pytest.raises(AssertionError):
        ycen_bad = np.full(50, 0, dtype=float)
        slitfunc_curved(img, ycen_bad, p1, p2, lambda_sp, lambda_sf, osample, yrange)

    with pytest.raises(AssertionError):
        p1_bad = np.full(50, 0, dtype=float)
        slitfunc_curved(img, ycen, p1_bad, p2, lambda_sp, lambda_sf, osample, yrange)

    with pytest.raises(AssertionError):
        p2_bad = np.full(50, 0, dtype=float)
        slitfunc_curved(img, ycen, p1, p2_bad, lambda_sp, lambda_sf, osample, yrange)

    with pytest.raises(AssertionError):
        slitfunc_curved(img, ycen, p1, p2, lambda_sp, lambda_sf, 0, yrange)
    with pytest.raises(AssertionError):
        slitfunc_curved(img, ycen, p1, p2, lambda_sp, -1, osample, yrange)
    with pytest.raises(AssertionError):
        slitfunc_curved(img, ycen, p1, p2, -1, lambda_sf, osample, yrange)
