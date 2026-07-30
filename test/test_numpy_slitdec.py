"""Equivalence of the numpy slitdec port against the CFFI reference.

The C implementation in clib/slitdec.c is the oracle: numpy_slitdec computes the
same sums by scatter/gather instead of pixel loops, so every output must agree to
rounding level (measured: 1.4e-14 worst case). Rounding differs through FMA
contraction, bincount accumulation order, pairwise np.sum, and Cholesky in place
of the C's unpivoted bandsol.

The mask and iteration-count assertions are exact on purpose. Both could in
principle differ -- the rejection threshold kappa*dev is a floating-point
comparison, and a residual can sit arbitrarily close to it -- so a failure there
means a real divergence has appeared, not just noise.
"""

import numpy as np
import pytest

from pyreduce.cwrappers import slitdec as slitdec_c

numpy_slitdec = pytest.importorskip("pyreduce.numpy_slitdec")
slitdec_numpy = numpy_slitdec.slitdec

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
    b = slitdec_numpy(**swath, **kwargs)

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
        # tall slit: delta_x ~ 15, so the sP system is genuinely wide-banded
        {"ncols": 400, "nrows": 60, "osample": 6, "hotpix": 40},
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
    b = slitdec_numpy(**swath)
    assert a["return_code"] == b["return_code"] == -1
    assert a["info"][2] == b["info"][2] == -2
    assert a["info"][4] == b["info"][4]


def _dense_from_upper(Aup, n):
    """Symmetric dense matrix from the upper-band storage Aup[d, i] = A[i, i+d]."""
    A = np.zeros((n, n))
    for d in range(Aup.shape[0]):
        for i in range(n - d):
            A[i, i + d] = A[i + d, i] = Aup[d, i]
    return A


@pytest.mark.parametrize("u", [0, 1, 4])
def test_solve_matches_dense(u):
    """The band solve reproduces a dense solve of the same system."""
    rng = np.random.default_rng(4)
    n = 40
    Aup = rng.normal(0, 0.1, (u + 1, n))
    Aup[0] += 5.0  # diagonally dominant, so positive definite
    b = rng.normal(0, 1, n)

    got = numpy_slitdec._solve(Aup.copy(), b.copy(), u)
    want = np.linalg.solve(_dense_from_upper(Aup, n), b)
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12)


def test_solve_falls_back_when_not_positive_definite():
    """Cholesky fails on an indefinite matrix; the pivoted LU fallback must not."""
    rng = np.random.default_rng(5)
    n = 30
    u = 2
    Aup = rng.normal(0, 1.0, (u + 1, n))
    Aup[0] -= 3.0  # negative diagonal: symmetric but indefinite
    b = rng.normal(0, 1, n)
    dense = _dense_from_upper(Aup, n)
    assert np.any(np.linalg.eigvalsh(dense) < 0)

    got = numpy_slitdec._solve(Aup.copy(), b.copy(), u)
    want = np.linalg.solve(dense, b)
    assert np.allclose(got, want, rtol=1e-8, atol=1e-10)


def test_input_validation_matches_wrapper():
    swath = _make_swath(40, 15, 6)
    with pytest.raises(ValueError):
        slitdec_numpy(**{**swath, "pix_unc": swath["pix_unc"][:, :-1]})
    with pytest.raises(ValueError):
        slitdec_numpy(**{**swath, "mask": swath["mask"][:-1]})
    with pytest.raises(ValueError):
        slitdec_numpy(**{**swath, "ycen": swath["ycen"][:-1]})
    with pytest.raises(ValueError):
        slitdec_numpy(**{**swath, "slitcurve": np.zeros((40, 7))})
    with pytest.raises(ValueError):
        slitdec_numpy(**{**swath, "slitdeltas": np.zeros(3)})
    with pytest.raises(ValueError):
        slitdec_numpy(**swath, preset_slitfunc=np.ones(3))
