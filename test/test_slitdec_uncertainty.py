"""The spectrum uncertainty against the scatter over noise realisations.

A tilted, photon- plus read-noise-limited swath is extracted many times with
fresh noise. The "propagated" and "scaled" uncertainties must match the
scatter of the extracted spectrum. The historic "residual" heuristic is not
calibrated; on this frame, which the model fits exactly, it comes out at
about 0.7 of the scatter.
"""

import numpy as np
import pytest

from pyreduce.extract import _get_backend

pytestmark = pytest.mark.unit

GAIN = 1.0
RON = 5.0


def _truth(ncols=150, nrows=21, osample=6):
    """Noiseless image of a Gaussian slit profile under a tilted slit.

    Built as slitdec's own model of a high-S/N frame, so it is exactly
    representable by the decomposition, tilt included.
    """
    x = np.arange(ncols)
    yy = np.arange(nrows) - nrows // 2
    spec = 3000 * (1 + 0.3 * np.sin(2 * np.pi * x / 41))
    img = (
        spec[None, :]
        * np.exp(-0.5 * (yy / 2.5) ** 2)[:, None]
        / (2.5 * np.sqrt(2 * np.pi))
    )
    slitcurve = np.zeros((ncols, 6))
    slitcurve[:, 1] = 0.2
    args = {
        "mask": np.ones((nrows, ncols), dtype=np.uint8),
        "ycen": np.full(ncols, nrows // 2 + 0.3),
        "slitcurve": slitcurve,
        "slitdeltas": np.zeros(nrows),
        "osample": osample,
        "lambda_sL": 1.0,
    }
    res = _get_backend().slitdec(img, np.ones_like(img), **args, kappa=0.0)
    return res["model"], args


@pytest.mark.parametrize("weights", ["none", "model"])
def test_propagated_uncertainty_matches_scatter(weights):
    truth, args = _truth()
    rng = np.random.default_rng(7)
    specs, sig_prop, sig_scaled = [], [], []
    for _ in range(120):
        data = rng.poisson(np.clip(truth, 0, None) * GAIN) / GAIN
        data = data + rng.normal(0, RON / GAIN, data.shape)
        # as extract_spectrum builds them, from the noisy data
        pix_unc = np.sqrt(np.maximum(data, 0) / GAIN + (RON / GAIN) ** 2)
        a = _get_backend().slitdec(data, pix_unc, **args, weights=weights)
        b = _get_backend().slitdec(
            data, pix_unc, **args, weights=weights, uncertainty="scaled"
        )
        specs.append(a["spectrum"])
        sig_prop.append(a["uncertainty"])
        sig_scaled.append(b["uncertainty"])
        m = a["margin"]
    core = slice(m + 5, -m - 5)
    scatter = np.std(np.array(specs)[:, core], axis=0, ddof=1)
    ratio_prop = np.median(np.median(np.array(sig_prop)[:, core], axis=0) / scatter)
    ratio_scaled = np.median(np.median(np.array(sig_scaled)[:, core], axis=0) / scatter)
    # 120 realisations: the scatter itself is good to ~7% per column, much
    # better in the median over ~100 columns
    assert 0.9 < ratio_prop < 1.1
    # the model fits exactly, so the reduced chi^2 is ~1
    assert 0.9 < ratio_scaled < 1.1


def test_extract_passes_uncertainty_mode():
    """The setting must reach the backend for every mode, "residual" included."""
    from pyreduce.extract import extract
    from pyreduce.trace_model import Trace

    truth, args = _truth()
    nrows, ncols = truth.shape
    img = np.zeros((60, ncols))
    img[20 : 20 + nrows] = truth
    rng = np.random.default_rng(3)
    img = rng.poisson(np.clip(img, 0, None)).astype(float) + rng.normal(
        0, RON, img.shape
    )
    sig = {}
    for mode in ("residual", "propagated", "scaled"):
        trace = Trace(
            m=1, pos=np.array([20 + nrows // 2 + 0.3]), column_range=(0, ncols)
        )
        trace.slit = np.array([[0.0], [0.2]])
        (spec,) = extract(
            img,
            [trace],
            extraction_height=nrows,
            gain=GAIN,
            readnoise=RON,
            osample=6,
            uncertainty=mode,
        )
        sig[mode] = np.nanmedian(spec.sig)
    assert len({round(v, 6) for v in sig.values()}) == 3, sig


def _extract_difference(img, var, nrows, ncols, rows0=20):
    from pyreduce.extract import extract
    from pyreduce.trace_model import Trace

    trace = Trace(
        m=1, pos=np.array([rows0 + nrows // 2 + 0.3]), column_range=(0, ncols)
    )
    trace.slit = np.array([[0.0], [0.2]])
    (spec,) = extract(
        img,
        [trace],
        extraction_height=nrows,
        gain=GAIN,
        readnoise=RON,
        osample=6,
        var=var,
    )
    return spec


def test_variance_input_for_difference_frame():
    """A-B on a bright sky: the image alone misses the background noise."""
    truth, _ = _truth()
    nrows, ncols = truth.shape
    obj = np.zeros((60, ncols))
    obj[20 : 20 + nrows] = truth
    sky = 2000.0
    rng = np.random.default_rng(11)
    specs, sig_img, sig_var = [], [], []
    for _ in range(40):
        a = rng.poisson((obj + sky) * GAIN) / GAIN + rng.normal(
            0, RON / GAIN, obj.shape
        )
        b = rng.poisson(np.full(obj.shape, sky) * GAIN) / GAIN + rng.normal(
            0, RON / GAIN, obj.shape
        )
        var = (np.maximum(a, 0) + np.maximum(b, 0)) / GAIN + 2 * (RON / GAIN) ** 2
        s_img = _extract_difference(a - b, None, nrows, ncols)
        s_var = _extract_difference(a - b, var, nrows, ncols)
        specs.append(s_var.spec)
        sig_img.append(s_img.sig)
        sig_var.append(s_var.sig)
    core = slice(20, -20)
    scatter = np.nanstd(np.array(specs)[:, core], axis=0, ddof=1)
    ratio_img = np.nanmedian(np.nanmedian(np.array(sig_img)[:, core], axis=0) / scatter)
    ratio_var = np.nanmedian(np.nanmedian(np.array(sig_var)[:, core], axis=0) / scatter)
    assert 0.9 < ratio_var < 1.1
    # sky noise of both frames is missing without the variances
    assert ratio_img < 0.6


def test_variance_input_default_unchanged():
    """var equal to the built-in formula reproduces the default exactly."""
    truth, _ = _truth()
    nrows, ncols = truth.shape
    img = np.zeros((60, ncols))
    img[20 : 20 + nrows] = truth
    rng = np.random.default_rng(5)
    img = rng.poisson(np.clip(img, 0, None)).astype(float) + rng.normal(
        0, RON, img.shape
    )
    var = np.maximum(img, 0) / GAIN + (RON / GAIN) ** 2
    a = _extract_difference(img, None, nrows, ncols)
    b = _extract_difference(img, var, nrows, ncols)
    np.testing.assert_array_equal(a.spec, b.spec)
    np.testing.assert_array_equal(a.sig, b.sig)
    # scaling the variances scales the propagated uncertainty, not the spectrum
    c = _extract_difference(img, 4 * var, nrows, ncols)
    np.testing.assert_array_equal(a.spec, c.spec)
    ok = np.isfinite(a.sig) & (a.sig > 0)
    np.testing.assert_allclose(c.sig[ok], 2 * a.sig[ok], rtol=1e-6)
