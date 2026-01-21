"""Tests for Numba-accelerated curved slit extraction."""

import numpy as np
import pytest

from pyreduce.numba_extract import (
    _slit_func_curved_internal as slit_func_curved,
)
from pyreduce.numba_extract import (
    build_sP_system,
    compute_model,
    solve_band_system,
    xi_zeta_tensors,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def synthetic_swath():
    """Create a synthetic spectral swath for testing."""
    np.random.seed(42)

    ncols = 200
    nrows = 10
    osample = 10  # Higher osample for better slit function recovery

    # True spectrum: smooth with some features
    x = np.arange(ncols)
    true_spec = (
        1000 + 500 * np.sin(2 * np.pi * x / 50) + 200 * np.exp(-(((x - 100) / 20) ** 2))
    )

    # True slit function: Gaussian, normalized so sum/osample = 1
    ny = osample * (nrows + 1) + 1
    iy = np.arange(ny)
    center = ny / 2
    true_slitf = np.exp(-(((iy - center) / (ny / 6)) ** 2))
    true_slitf /= np.sum(true_slitf) / osample

    # Simple geometry: no curvature, centered
    ycen = np.full(ncols, 0.5)
    ycen_offset = np.zeros(ncols, dtype=np.int32)
    y_lower_lim = nrows // 2
    psf_curve = np.zeros((ncols, 3))

    # Build geometry and create synthetic image
    xi, zeta, m_zeta = xi_zeta_tensors(
        ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, psf_curve
    )
    model = compute_model(zeta, m_zeta, true_spec, true_slitf, ncols, nrows)

    # Add noise (relative to signal)
    noise_level = np.median(model) * 0.01  # 1% noise
    im = model + np.random.randn(nrows, ncols) * noise_level
    mask = np.ones((nrows, ncols))

    return {
        "im": im,
        "mask": mask,
        "ycen": ycen,
        "ycen_offset": ycen_offset,
        "y_lower_lim": y_lower_lim,
        "osample": osample,
        "psf_curve": psf_curve,
        "true_spec": true_spec,
        "true_slitf": true_slitf,
        "noise_level": noise_level,
        "model": model,
    }


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestNumbaExtract:
    """Tests for the Numba extraction implementation."""

    def test_round_trip_extraction(self):
        """Test that model -> extract -> model round-trips correctly."""
        ncols = 50
        nrows = 10
        osample = 10
        ny = osample * (nrows + 1) + 1

        ycen = np.full(ncols, 0.5)
        ycen_offset = np.zeros(ncols, dtype=np.int32)
        y_lower_lim = nrows // 2
        psf_curve = np.zeros((ncols, 3))

        # Spectrum with structure
        spec = 100.0 + 50.0 * np.sin(np.arange(ncols) * 0.2)

        # Gaussian slit function
        iy_arr = np.arange(ny)
        slitf = np.exp(-(((iy_arr - ny / 2) / (ny / 4)) ** 2))
        slitf /= np.sum(slitf) / osample

        # Build geometry and model
        xi, zeta, m_zeta = xi_zeta_tensors(
            ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, psf_curve
        )
        model = compute_model(zeta, m_zeta, spec, slitf, ncols, nrows)

        # Direct spectrum extraction with known slitf should be exact
        mask = np.ones((nrows, ncols))
        p_Aij, p_bj = build_sP_system(
            xi, zeta, m_zeta, slitf, mask, model, ncols, nrows, ny, 0
        )
        spec_direct = solve_band_system(p_Aij, p_bj, 0)

        margin = 5
        rel_error = (
            np.abs(spec_direct[margin:-margin] - spec[margin:-margin])
            / spec[margin:-margin]
        )
        assert np.max(rel_error) < 1e-10, (
            f"Direct extraction should be exact, got max error {np.max(rel_error)}"
        )

    def test_weight_conservation(self):
        """Test that zeta weights sum correctly for each pixel."""
        ncols = 50
        nrows = 6
        osample = 4
        ny = osample * (nrows + 1) + 1

        ycen = np.full(ncols, 0.5)
        ycen_offset = np.zeros(ncols, dtype=np.int32)
        y_lower_lim = nrows // 2
        psf_curve = np.zeros((ncols, 3))

        xi, zeta, m_zeta = xi_zeta_tensors(
            ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, psf_curve
        )

        # For no curvature, each pixel (x, y) should receive weight 1.0 total
        for y in range(nrows):
            for x in range(ncols):
                total_weight = 0.0
                for m in range(m_zeta[x, y]):
                    total_weight += zeta[x, y, m, 2]
                assert abs(total_weight - 1.0) < 0.01, (
                    f"Pixel ({x},{y}): weight sum {total_weight:.4f} != 1.0"
                )

    def test_model_column_sums_proportional(self):
        """Test that model column sums are proportional to spectrum."""
        ncols = 50
        nrows = 10
        osample = 10
        ny = osample * (nrows + 1) + 1

        ycen = np.full(ncols, 0.5)
        ycen_offset = np.zeros(ncols, dtype=np.int32)
        y_lower_lim = nrows // 2
        psf_curve = np.zeros((ncols, 3))

        # Spectrum with variation
        spec = 100.0 + 50.0 * np.sin(np.arange(ncols) * 0.2)

        # Gaussian slit function (localized)
        iy_arr = np.arange(ny)
        slitf = np.exp(-(((iy_arr - ny / 2) / (ny / 4)) ** 2))
        slitf /= np.sum(slitf) / osample

        xi, zeta, m_zeta = xi_zeta_tensors(
            ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, psf_curve
        )
        model = compute_model(zeta, m_zeta, spec, slitf, ncols, nrows)

        # Column sums should be proportional to spectrum
        model_column_sums = np.sum(model, axis=0)

        margin = 5
        ratios = model_column_sums[margin:-margin] / spec[margin:-margin]
        ratio_std = np.std(ratios) / np.mean(ratios)

        assert ratio_std < 0.01, (
            f"Column sum ratios not constant: std/mean = {ratio_std:.4f}"
        )

    def test_xi_zeta_tensors_shape(self, synthetic_swath):
        """Test that geometry tensors have correct shapes."""
        s = synthetic_swath
        ncols = len(s["ycen"])
        nrows = s["im"].shape[0]
        ny = s["osample"] * (nrows + 1) + 1

        xi, zeta, m_zeta = xi_zeta_tensors(
            ncols,
            nrows,
            ny,
            s["ycen"],
            s["ycen_offset"],
            s["y_lower_lim"],
            s["osample"],
            s["psf_curve"],
        )

        assert xi.shape == (ncols, ny, 4, 3)
        assert zeta.shape[0] == ncols
        assert zeta.shape[1] == nrows
        assert m_zeta.shape == (ncols, nrows)

    def test_extraction_convergence(self, synthetic_swath):
        """Test that extraction converges."""
        s = synthetic_swath

        result = slit_func_curved(
            s["im"],
            s["mask"],
            s["ycen"],
            s["ycen_offset"],
            s["y_lower_lim"],
            s["osample"],
            s["psf_curve"],
            lambda_sL=1.0,
            lambda_sP=0.0,
            maxiter=20,
        )

        assert result["niter"] < 20, "Should converge before maxiter"

    def test_spectrum_recovery(self, synthetic_swath):
        """Test that we recover the input spectrum reasonably well."""
        s = synthetic_swath

        result = slit_func_curved(
            s["im"],
            s["mask"],
            s["ycen"],
            s["ycen_offset"],
            s["y_lower_lim"],
            s["osample"],
            s["psf_curve"],
            lambda_sL=10.0,
            lambda_sP=0.0,
            maxiter=20,
        )

        margin = 10
        extracted = result["spec"][margin:-margin]
        true = s["true_spec"][margin:-margin]

        correlation = np.corrcoef(extracted, true)[0, 1]
        assert correlation > 0.99, f"Spectrum correlation {correlation:.4f} too low"

        rel_error = np.abs(extracted - true) / true
        median_error = np.median(rel_error)
        assert median_error < 0.15, f"Median relative error {median_error:.3f} too high"

    def test_model_residuals(self, synthetic_swath):
        """Test that model residuals are reasonable."""
        s = synthetic_swath

        result = slit_func_curved(
            s["im"],
            s["mask"],
            s["ycen"],
            s["ycen_offset"],
            s["y_lower_lim"],
            s["osample"],
            s["psf_curve"],
            lambda_sL=10.0,
            lambda_sP=0.0,
            maxiter=20,
        )

        residuals = s["im"] - result["model"]
        rms = np.sqrt(np.mean(residuals**2))
        signal_rms = np.sqrt(np.mean(s["model"] ** 2))

        rel_rms = rms / signal_rms
        assert rel_rms < 0.2, f"Relative RMS {rel_rms:.3f} too high"

    def test_with_curvature(self):
        """Test extraction with slit curvature."""
        np.random.seed(123)

        ncols = 100
        nrows = 10
        osample = 10
        ny = osample * (nrows + 1) + 1

        true_spec = 1000.0 + 200.0 * np.sin(np.arange(ncols) * 0.1)

        iy = np.arange(ny)
        true_slitf = np.exp(-(((iy - ny / 2) / (ny / 6)) ** 2))
        true_slitf /= np.sum(true_slitf) / osample

        ycen = np.full(ncols, 0.5)
        ycen_offset = np.zeros(ncols, dtype=np.int32)
        y_lower_lim = nrows // 2
        psf_curve = np.zeros((ncols, 3))
        psf_curve[:, 1] = 0.05  # linear tilt coefficient

        xi, zeta, m_zeta = xi_zeta_tensors(
            ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, psf_curve
        )
        model = compute_model(zeta, m_zeta, true_spec, true_slitf, ncols, nrows)

        noise = np.median(model) * 0.01
        im = model + np.random.randn(nrows, ncols) * noise
        mask = np.ones((nrows, ncols))

        result = slit_func_curved(
            im,
            mask,
            ycen,
            ycen_offset,
            y_lower_lim,
            osample,
            psf_curve,
            lambda_sL=10.0,
            lambda_sP=0.0,
            maxiter=20,
        )

        assert result["niter"] < 20

        margin = 20
        extracted = result["spec"][margin:-margin]
        true = true_spec[margin:-margin]

        correlation = np.corrcoef(extracted, true)[0, 1]
        assert correlation > 0.99, f"Spectrum correlation {correlation:.4f} too low"


@pytest.mark.instrument
def test_numba_vs_c_extraction(flat, orders, instrument):
    """Compare Numba and C curved extraction on real UVES data."""
    import os
    from pathlib import Path

    from pyreduce.cwrappers import slitfunc_curved as c_slitfunc_curved
    from pyreduce.numba_extract import slitfunc_curved as numba_slitfunc_curved
    from pyreduce.util import make_index

    if instrument != "UVES":
        pytest.skip("Test designed for UVES data only")

    flat_img, flat_head = flat
    if flat_img is None:
        pytest.skip("No flat data available")

    # Setup debug output directory
    reduce_data = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
    debug_dir = Path(reduce_data) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    traces, column_range = orders

    # Pick a middle order and a swath in the middle of it
    nord = traces.shape[0]
    order_idx = nord // 2
    trace = traces[order_idx]
    cr = column_range[order_idx]

    # Swath parameters
    swath_width = 300
    extraction_height = 50
    xlow = (cr[0] + cr[1]) // 2
    xhigh = xlow + swath_width
    if xhigh > cr[1]:
        xhigh = cr[1]
        xlow = xhigh - swath_width

    # Get ycen for this swath
    x = np.arange(xlow, xhigh)
    ycen_full = np.polyval(trace, x)
    ycen_int = np.floor(ycen_full).astype(int)
    ycen = ycen_full - ycen_int

    # Cut out swath
    ylow = yhigh = extraction_height
    index = make_index(ycen_int - ylow, ycen_int + yhigh, xlow, xhigh, zero=xlow)
    swath_img = flat_img[index].astype(float)

    # Common parameters
    lambda_sp = 0
    lambda_sf = 0.1
    osample = 1
    yrange = (ylow, yhigh)

    # C extraction (curved with zero curvature)
    sp_c, sl_c, model_c, unc_c, mask_c, info_c = c_slitfunc_curved(
        swath_img,
        ycen,
        p1=0,
        p2=0,
        lambda_sp=lambda_sp,
        lambda_sf=lambda_sf,
        osample=osample,
        yrange=yrange,
    )

    # Numba extraction
    sp_numba, sl_numba, model_numba, unc_numba, mask_numba, info_numba = (
        numba_slitfunc_curved(
            swath_img,
            ycen,
            p1=0,
            p2=0,
            lambda_sp=lambda_sp,
            lambda_sf=lambda_sf,
            osample=osample,
            yrange=yrange,
        )
    )

    # Compare shapes
    assert sp_c.shape == sp_numba.shape, (
        f"Spectrum shapes differ: {sp_c.shape} vs {sp_numba.shape}"
    )
    assert sl_c.shape == sl_numba.shape, (
        f"Slitfunc shapes differ: {sl_c.shape} vs {sl_numba.shape}"
    )

    # Compare spectra
    margin = 10
    sp_c_mid = sp_c[margin:-margin]
    sp_numba_mid = sp_numba[margin:-margin]

    correlation = np.corrcoef(sp_c_mid, sp_numba_mid)[0, 1]
    assert correlation > 0.999, f"Spectrum correlation {correlation:.6f} too low"

    sp_rel_diff = np.abs(sp_c_mid - sp_numba_mid) / np.maximum(sp_c_mid, 1)
    median_diff = np.median(sp_rel_diff)
    max_diff = np.max(sp_rel_diff)

    print("\nSpectrum comparison:")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Median rel diff: {median_diff:.4f}")
    print(f"  Max rel diff: {max_diff:.4f}")

    sl_rel_diff = np.abs(sl_c - sl_numba) / np.maximum(np.abs(sl_c), 1e-10)
    print("Slitfunc comparison:")
    print(f"  Median rel diff: {np.median(sl_rel_diff):.4f}")
    print(f"  Max rel diff: {np.max(sl_rel_diff):.4f}")

    model_rel_diff = np.abs(model_c - model_numba) / np.maximum(model_c, 1)
    print("Model comparison:")
    print(f"  Median rel diff: {np.median(model_rel_diff):.4f}")
    print(f"  Max rel diff: {np.max(model_rel_diff):.4f}")

    # Save results
    outfile = debug_dir / "numba_vs_c_extraction.npz"
    np.savez(
        outfile,
        swath_img=swath_img,
        ycen=ycen,
        sp_c=sp_c,
        sl_c=sl_c,
        model_c=model_c,
        unc_c=unc_c,
        mask_c=mask_c,
        info_c=info_c,
        sp_numba=sp_numba,
        sl_numba=sl_numba,
        model_numba=model_numba,
        unc_numba=unc_numba,
        mask_numba=mask_numba,
        niter_numba=info_numba[3],
    )
    print(f"Saved results to {outfile}")

    assert median_diff < 0.05, f"Median spectrum difference {median_diff:.4f} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
