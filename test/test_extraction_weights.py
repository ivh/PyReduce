"""Pixel weights in the slit decomposition (extraction_weights setting).

Plumbing: settings -> step kwargs -> extract -> backend. Results: the two
effects the modes exist for, on synthetic frames where they are predictable.
Backend equivalence for the weighted modes lives in test_numpy_slitdec.py and
test_numba_slitdec.py.
"""

import numpy as np
import pytest
from scipy.special import erf

from pyreduce import extract, reduce
from pyreduce.configuration import load_config, validate_config
from pyreduce.instruments.instrument_info import load_instrument
from pyreduce.trace_model import Trace

pytestmark = pytest.mark.unit

MODES = ["none", "unc", "model", "flux"]
NROWS, NCOLS, YC = 30, 600, 14.3


def _profile(sigma=2.0):
    edges = np.arange(NROWS + 1) - 0.5
    return 0.5 * np.diff(erf((edges - YC) / (np.sqrt(2) * sigma)))


def _traces():
    pos = np.polyfit(np.arange(NCOLS), np.full(NCOLS, YC), 2)
    return [Trace(m=0, group=0, pos=pos, column_range=(0, NCOLS))]


# --- plumbing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "step_name,step_cls",
    [("science", reduce.ScienceExtraction), ("norm_flat", reduce.NormalizeFlatField)],
)
def test_settings_reach_step_kwargs(tmp_path, step_name, step_cls):
    instrument = load_instrument("UVES")
    config = load_config(None, "UVES", 0)[step_name]
    step = step_cls(instrument, "", "", "", str(tmp_path), None, **config)
    assert step.extraction_kwargs["weights"] == "none"
    assert step.extraction_kwargs["noise_floor"] is None

    config = {**config, "extraction_weights": "model", "extraction_noise_floor": 12.5}
    step = step_cls(instrument, "", "", "", str(tmp_path), None, **config)
    assert step.extraction_kwargs["weights"] == "model"
    assert step.extraction_kwargs["noise_floor"] == 12.5


def test_schema_validates_weights():
    config = load_config(None, "UVES", 0)
    config["science"]["extraction_weights"] = "flux"
    config["science"]["extraction_noise_floor"] = 3.0
    validate_config(config)

    config["science"]["extraction_weights"] = "horne"
    with pytest.raises(ValueError):
        validate_config(config)

    config["science"]["extraction_weights"] = "model"
    config["science"]["extraction_noise_floor"] = -1.0
    with pytest.raises(ValueError):
        validate_config(config)


class _Recorder:
    def __init__(self, backend):
        self.backend = backend
        self.calls = []

    def slitdec(self, im, pix_unc, mask, *args, **kwargs):
        self.calls.append((im.copy(), pix_unc.copy(), kwargs))
        return self.backend.slitdec(im, pix_unc, mask, *args, **kwargs)


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder(extract._get_backend())
    monkeypatch.setattr(extract, "_get_backend", lambda: rec)
    return rec


def test_extract_passes_weights_and_pix_unc(recorder):
    rng = np.random.default_rng(0)
    gain, readnoise = 2.0, 5.0
    img = rng.poisson(_profile()[:, None] * np.full(NCOLS, 2e4)).astype(float)
    extract.extract(
        img,
        _traces(),
        extraction_height=16,
        gain=gain,
        readnoise=readnoise,
        weights="model",
        noise_floor=7.0,
    )

    assert recorder.calls
    for im, pix_unc, kwargs in recorder.calls:
        assert kwargs["weights"] == "model"
        assert kwargs["noise_floor"] == 7.0
        # ADU, with read noise: what weights="unc" and the model floor rely on
        var = np.maximum(im, 0) / gain + (readnoise / gain) ** 2
        np.testing.assert_allclose(pix_unc, np.sqrt(np.maximum(var, 1 / gain**2)))


def test_extract_unweighted_omits_weight_kwargs(recorder):
    """An external charslit predating weights must still work with the default."""
    img = np.random.default_rng(0).poisson(_profile()[:, None] * 1e4).astype(float)
    img = np.repeat(img[:, :1], NCOLS, axis=1)
    extract.extract(img, _traces(), extraction_height=16)
    assert recorder.calls
    for _, _, kwargs in recorder.calls:
        assert "weights" not in kwargs
        assert "noise_floor" not in kwargs


@pytest.mark.parametrize("backend", ["cwrappers", "numpy_slitdec", "numba_slitdec"])
def test_unknown_weights_raise(backend):
    mod = pytest.importorskip(f"pyreduce.{backend}")
    im = np.ones((5, 10))
    with pytest.raises(ValueError, match="weights"):
        mod.slitdec(
            im,
            im,
            np.ones(im.shape, dtype=np.uint8),
            np.full(10, 2.5),
            np.zeros((10, 6)),
            np.zeros(5),
            weights="horne",
        )


# --- results ----------------------------------------------------------------


def test_weights_reduce_photon_limited_variance():
    """Separable profile, pure photon noise. Unweighted LSQ has variance
    F * sum(p^3) / sum(p^2)^2 (2/sqrt(3) = 1.155 F for a Gaussian); every
    weighted mode reaches the box-sum optimum F here."""
    prof = _profile()
    flux = np.full(NCOLS, 2e5)
    expected_none = prof.sum() * (prof**3).sum() / (prof**2).sum() ** 2

    var = {}
    for mode in MODES:
        z = []
        for seed in range(3):
            img = np.random.default_rng(seed).poisson(prof[:, None] * flux)
            s = extract.extract(
                img.astype(float),
                _traces(),
                extraction_height=16,
                osample=6,
                weights=mode,
            )[0]
            ok = ~np.isnan(s.spec)
            z.append((s.spec[ok] - flux[ok]) / np.sqrt(flux[ok]))
        z = np.concatenate(z)
        assert abs(z.mean()) < 0.15, mode  # unbiased
        var[mode] = z.var()

    assert var["none"] == pytest.approx(expected_none, rel=0.06)
    for mode in ("unc", "model", "flux"):
        assert var[mode] == pytest.approx(1.0, abs=0.06), mode
        assert var["none"] / var[mode] > 1.08, mode


def test_weights_sharpen_lsf_when_line_width_varies_along_slit():
    """Wider lines in the bright central rows: unweighted rows count with
    sL^2 and broaden the LSF; flux weights count them with sL. Both match
    the analytic row averages; bright model weights land on the flux ones."""
    prof = _profile()
    rows = np.arange(NROWS)
    width = 1.4 + 0.5 * np.exp(-0.5 * ((rows - YC) / 2.0) ** 2)  # 1.9 px centre
    x = np.arange(NCOLS)
    centres = np.arange(25, NCOLS - 25, 50) + 0.3
    lines = sum(
        np.exp(-0.5 * ((x[None, :] - c) / width[:, None]) ** 2)
        / (np.sqrt(2 * np.pi) * width[:, None])
        for c in centres
    )
    img = 1e5 * prof[:, None] * (0.02 + lines)

    def line_variance(sp):
        out = []
        for c in centres:
            sel = np.abs(x - c) < 8
            y = sp[sel] - np.nanmedian(sp[np.abs(x - c) > 15])
            mu = (y * x[sel]).sum() / y.sum()
            out.append((y * (x[sel] - mu) ** 2).sum() / y.sum())
        return np.mean(out)

    lsf = {}
    for mode in ("none", "model", "flux"):
        # noiseless frame: no rejection, it would only see round-off
        s = extract.extract(
            img,
            _traces(),
            extraction_height=16,
            osample=6,
            weights=mode,
            reject_threshold=0,
        )[0]
        lsf[mode] = line_variance(s.spec)

    p = prof / prof.sum()
    assert lsf["none"] == pytest.approx(
        (p**2 * width**2).sum() / (p**2).sum(), rel=0.01
    )
    assert lsf["flux"] == pytest.approx((p * width**2).sum(), rel=0.01)
    assert lsf["model"] == pytest.approx(lsf["flux"], rel=0.01)
    assert lsf["none"] - lsf["flux"] > 0.1
