import numpy as np
import pytest

from pyreduce.cwrappers import slitdec

pytestmark = pytest.mark.unit


def _make_inputs(nrows=15, ncols=40, osample=6):
    sl_true = np.exp(-0.5 * ((np.arange(nrows) - nrows / 2) / 2.0) ** 2)
    sp_true = 100 + 50 * np.sin(np.arange(ncols) / 5.0)
    img = sl_true[:, None] * sp_true[None, :]
    pix_unc = np.sqrt(np.abs(img))
    pix_unc[pix_unc < 1] = 1
    mask = np.ones((nrows, ncols), dtype=np.uint8)
    ycen = np.full(ncols, nrows / 2.0)
    slitcurve = np.zeros((ncols, 6))
    slitdeltas = np.zeros(nrows)
    return img, pix_unc, mask, ycen, slitcurve, slitdeltas, sp_true


def test_slitdec_roundtrip():
    nrows, ncols, osample = 15, 40, 6
    img, pix_unc, mask, ycen, slitcurve, slitdeltas, sp_true = _make_inputs(
        nrows, ncols, osample
    )

    result = slitdec(img, pix_unc, mask, ycen, slitcurve, slitdeltas, osample=osample)

    ny = osample * (nrows + 1) + 1
    assert result["return_code"] == 0
    assert result["spectrum"].shape == (ncols,)
    assert result["slitfunction"].shape == (ny,)
    assert result["model"].shape == (nrows, ncols)
    assert result["uncertainty"].shape == (ncols,)
    assert result["mask"].shape == (nrows, ncols)
    # Recovered spectrum should track the true one closely
    assert np.corrcoef(result["spectrum"], sp_true)[0, 1] > 0.99


def test_slitdec_slitdeltas_ny_length():
    """slitdeltas may be passed pre-interpolated at ny length."""
    nrows, ncols, osample = 15, 40, 6
    img, pix_unc, mask, ycen, slitcurve, _, _ = _make_inputs(nrows, ncols, osample)
    ny = osample * (nrows + 1) + 1

    result = slitdec(img, pix_unc, mask, ycen, slitcurve, np.zeros(ny), osample=osample)
    assert result["slitfunction"].shape == (ny,)


def test_slitdec_preset_slitfunc():
    """A preset slit function is used as-is (single-pass) instead of being solved."""
    nrows, ncols, osample = 15, 40, 6
    img, pix_unc, mask, ycen, slitcurve, slitdeltas, base_sp = _make_inputs(
        nrows, ncols, osample
    )

    full = slitdec(img, pix_unc, mask, ycen, slitcurve, slitdeltas, osample=osample)
    sL = full["slitfunction"]

    preset = slitdec(
        img,
        pix_unc,
        mask,
        ycen,
        slitcurve,
        slitdeltas,
        osample=osample,
        maxiter=1,
        preset_slitfunc=sL.copy(),
    )

    assert preset["return_code"] == 0
    # The returned slit function is the preset, normalized to sum osample
    np.testing.assert_allclose(
        preset["slitfunction"], sL * (osample / sL.sum()), rtol=1e-6
    )
    # Single-pass spectrum against the same slit function matches the full solve
    assert np.corrcoef(preset["spectrum"], full["spectrum"])[0, 1] > 0.999

    # A per-row (nrows) preset is interpolated up to ny
    nrows_preset = slitdec(
        img,
        pix_unc,
        mask,
        ycen,
        slitcurve,
        slitdeltas,
        osample=osample,
        maxiter=1,
        preset_slitfunc=np.ones(nrows),
    )
    assert nrows_preset["return_code"] == 0

    # Wrong length is rejected
    with pytest.raises(ValueError):
        slitdec(
            img,
            pix_unc,
            mask,
            ycen,
            slitcurve,
            slitdeltas,
            osample=osample,
            preset_slitfunc=np.ones(3),
        )


def test_slitdec_input_validation():
    nrows, ncols, osample = 15, 40, 6
    img, pix_unc, mask, ycen, slitcurve, slitdeltas, _ = _make_inputs(
        nrows, ncols, osample
    )

    # pix_unc shape mismatch
    with pytest.raises(ValueError):
        slitdec(
            img, pix_unc[:, :-1], mask, ycen, slitcurve, slitdeltas, osample=osample
        )

    # mask shape mismatch
    with pytest.raises(ValueError):
        slitdec(img, pix_unc, mask[:-1], ycen, slitcurve, slitdeltas, osample=osample)

    # ycen length mismatch
    with pytest.raises(ValueError):
        slitdec(img, pix_unc, mask, ycen[:-1], slitcurve, slitdeltas, osample=osample)

    # slitcurve too many coefficients
    with pytest.raises(ValueError):
        slitdec(
            img, pix_unc, mask, ycen, np.zeros((ncols, 7)), slitdeltas, osample=osample
        )

    # slitdeltas wrong length (neither nrows nor ny)
    with pytest.raises(ValueError):
        slitdec(img, pix_unc, mask, ycen, slitcurve, np.zeros(3), osample=osample)
