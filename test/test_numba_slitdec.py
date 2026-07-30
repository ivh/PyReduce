"""Equivalence of the numba slitdec port against the CFFI reference.

The C implementation in clib/slitdec.c is the oracle: numba_slitdec is a
transliteration of it, so every output must agree to rounding level. Rounding
differs only through FMA contraction and vectorized accumulation order.
"""

import numpy as np
import pytest

from pyreduce.cwrappers import slitdec as slitdec_c

numba_slitdec = pytest.importorskip("pyreduce.numba_slitdec")
slitdec_numba = numba_slitdec.slitdec

pytestmark = pytest.mark.unit

RTOL = 1e-10


def _make_swath(ncols, nrows, osample, curved=True, hotpix=0, seed=1):
    rng = np.random.default_rng(seed)
    x = np.arange(ncols)
    ycen = nrows / 2 + 0.3 * np.sin(2 * np.pi * x / ncols) + 0.17
    spec = 1000 * (1 + 0.5 * np.sin(2 * np.pi * x / 37)) + 200
    yy = np.arange(nrows) - nrows / 2
    img = spec[None, :] * np.exp(-0.5 * (yy / (nrows / 5.0)) ** 2)[:, None]
    img += rng.normal(0, 3.0, img.shape)
    if hotpix:
        img[rng.integers(0, nrows, hotpix), rng.integers(0, ncols, hotpix)] += 5000

    slitcurve = np.zeros((ncols, 6))
    if curved:
        slitcurve[:, 1] = 0.15 + 0.02 * np.cos(2 * np.pi * x / ncols)
        slitcurve[:, 2] = 0.004

    return {
        "im": img,
        "pix_unc": np.sqrt(np.abs(img)),
        "mask": np.ones((nrows, ncols), dtype=np.uint8),
        "ycen": ycen,
        "slitcurve": slitcurve,
        "slitdeltas": np.zeros(osample * (nrows + 1) + 1),
        "osample": osample,
    }


def _assert_equivalent(swath, **kwargs):
    a = slitdec_c(**swath, **kwargs)
    b = slitdec_numba(**swath, **kwargs)

    assert b["return_code"] == a["return_code"]
    # Same convergence path: iteration count, status and delta_x must match
    np.testing.assert_array_equal(a["info"][2:], b["info"][2:])
    np.testing.assert_array_equal(a["mask"], b["mask"])
    for key in ("spectrum", "slitfunction", "model", "uncertainty"):
        # Compare against the peak, not element-wise: near-zero elements carry
        # no information and would fail any relative tolerance.
        scale = np.max(np.abs(a[key]))
        assert np.max(np.abs(a[key] - b[key])) <= RTOL * scale, key


@pytest.mark.parametrize(
    "case",
    [
        {"ncols": 200, "nrows": 15, "osample": 4, "curved": False},
        {"ncols": 200, "nrows": 15, "osample": 4},
        {"ncols": 300, "nrows": 21, "osample": 6},
        {"ncols": 300, "nrows": 21, "osample": 6, "hotpix": 40},
    ],
)
def test_matches_c_backend(case):
    _assert_equivalent(_make_swath(**case))


def test_matches_c_backend_smoothed_spectrum():
    """lambda_sP > 0 forces delta_x >= 1 and the sP regularization branch."""
    _assert_equivalent(_make_swath(200, 15, 4), lambda_sP=0.5)


def test_matches_c_backend_no_rejection():
    _assert_equivalent(_make_swath(300, 21, 6, hotpix=40), kappa=0.0)


def test_matches_c_backend_preset_slitfunc():
    nrows = 15
    yy = np.arange(nrows) - nrows / 2
    preset = np.exp(-0.5 * (yy / (nrows / 5.0)) ** 2)
    _assert_equivalent(_make_swath(200, nrows, 4), preset_slitfunc=preset)


def test_curvature_too_large_returns_error():
    """nx > ncols bails out with status -2, same as the C code."""
    swath = _make_swath(20, 15, 4)
    swath["slitcurve"][:, 1] = 20.0

    a = slitdec_c(**swath)
    b = slitdec_numba(**swath)
    assert a["return_code"] == b["return_code"] == -1
    assert a["info"][2] == b["info"][2] == -2
    assert a["info"][4] == b["info"][4]


def test_input_validation_matches_wrapper():
    swath = _make_swath(40, 15, 6)
    with pytest.raises(ValueError):
        slitdec_numba(**{**swath, "pix_unc": swath["pix_unc"][:, :-1]})
    with pytest.raises(ValueError):
        slitdec_numba(**{**swath, "mask": swath["mask"][:-1]})
    with pytest.raises(ValueError):
        slitdec_numba(**{**swath, "ycen": swath["ycen"][:-1]})
    with pytest.raises(ValueError):
        slitdec_numba(**{**swath, "slitcurve": np.zeros((40, 7))})
    with pytest.raises(ValueError):
        slitdec_numba(**{**swath, "slitdeltas": np.zeros(3)})
    with pytest.raises(ValueError):
        slitdec_numba(**swath, preset_slitfunc=np.ones(3))
