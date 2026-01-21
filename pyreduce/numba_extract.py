"""
Numba-accelerated curved slit extraction.

This is a reimplementation of the slit_func_2d_xi_zeta_bd algorithm
using Numba JIT compilation for performance.

The algorithm decomposes a 2D spectral image into:
- sP: 1D spectrum (flux vs wavelength)
- sL: 1D slit illumination function (flux vs cross-dispersion)

such that: model[x,y] = sum over contributing subpixels of sP[x'] * sL[iy] * weight
"""

import numpy as np
from numba import njit
from scipy.linalg import solve_banded

# -----------------------------------------------------------------------------
# Geometry tensor construction
# -----------------------------------------------------------------------------


@njit(cache=True)
def _add_to_zeta(zeta, m_zeta, xx, yy, x, iy, weight, max_contrib):
    """Helper to add a contribution to zeta tensor."""
    m = m_zeta[xx, yy]
    if m < max_contrib:
        zeta[xx, yy, m, 0] = x
        zeta[xx, yy, m, 1] = iy
        zeta[xx, yy, m, 2] = weight
        m_zeta[xx, yy] = m + 1


@njit(cache=True)
def xi_zeta_tensors(
    ncols: int,
    nrows: int,
    ny: int,
    ycen: np.ndarray,
    ycen_offset: np.ndarray,
    y_lower_lim: int,
    osample: int,
    psf_curve: np.ndarray,
) -> tuple:
    """
    Build the xi and zeta geometry tensors describing subpixel contributions.

    Parameters
    ----------
    ncols : int
        Swath width in pixels
    nrows : int
        Extraction slit height in pixels
    ny : int
        Size of oversampled slit function array: ny = osample * (nrows + 1) + 1
    ycen : array of shape (ncols,)
        Fractional y-offset of order center (0 <= ycen < 1)
    ycen_offset : array of shape (ncols,)
        Integer y-offset for order packing
    y_lower_lim : int
        Pixels below the central line
    osample : int
        Oversampling factor
    psf_curve : array of shape (ncols, 3)
        Quadratic curvature coefficients [c0, c1, c2] per column

    Returns
    -------
    xi : array of shape (ncols, ny, 4, 3)
        Last axis: [target_x, target_y, weight]
    zeta : array of shape (ncols, nrows, max_contrib, 3)
        Last axis: [source_x, source_iy, weight]
    m_zeta : array of shape (ncols, nrows)
        Number of contributors per pixel
    """
    max_contrib = 3 * (osample + 1)
    step = 1.0 / osample

    # Initialize output arrays
    xi = np.full((ncols, ny, 4, 3), -1.0)
    xi[:, :, :, 2] = 0.0

    zeta = np.full((ncols, nrows, max_contrib, 3), -1.0)
    zeta[:, :, :, 2] = 0.0

    m_zeta = np.zeros((ncols, nrows), dtype=np.int32)

    for x in range(ncols):
        # Initial subpixel indices (before row 0)
        iy2 = osample - int(np.floor(ycen[x] * osample))
        iy1 = iy2 - osample

        # Partial subpixel weights at boundaries
        d1 = ycen[x] % step
        if d1 == 0:
            d1 = step
        d2 = step - d1

        # Initial dy (distance from ycen for first subpixel in row y_lower_lim)
        dy = ycen[x] - np.floor((y_lower_lim + ycen[x]) / step) * step - step

        for y in range(nrows):
            iy1 += osample  # Bottom subpixel falling in row y
            iy2 += osample  # Top subpixel falling in row y
            dy -= step

            for iy in range(iy1, iy2 + 1):
                # Weight for this subpixel
                if iy == iy1:
                    w = d1
                elif iy == iy2:
                    w = d2
                else:
                    w = step

                dy += step

                # Compute horizontal shift from curvature
                dy_centered = dy - ycen[x]
                delta = (psf_curve[x, 1] + psf_curve[x, 2] * dy_centered) * dy_centered

                # Integer shift and direction
                ix1 = int(delta)
                if delta > 0:
                    ix2 = ix1 + 1
                elif delta < 0:
                    ix2 = ix1 - 1
                else:
                    ix2 = ix1

                # Three cases based on subpixel position in row y:
                # A: iy == iy1 (entering row y) -> upper corners (2, 3)
                # B: iy1 < iy < iy2 (fully inside) -> lower corners (0, 1)
                # C: iy == iy2 (leaving row y) -> lower corners (0, 1)

                if iy == iy1:  # Case A: entering row y
                    if ix1 < ix2:  # Shifting right
                        if x + ix1 >= 0 and x + ix2 < ncols:
                            # Upper right corner (3)
                            xx = x + ix1
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            weight = w - abs(delta - ix1) * w
                            if 0 <= yy < nrows and weight > 0:
                                xi[x, iy, 3, 0] = xx
                                xi[x, iy, 3, 1] = yy
                                xi[x, iy, 3, 2] = weight
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, weight, max_contrib
                                )
                            # Upper left corner (2)
                            xx = x + ix2
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            weight = abs(delta - ix1) * w
                            if 0 <= yy < nrows and weight > 0:
                                xi[x, iy, 2, 0] = xx
                                xi[x, iy, 2, 1] = yy
                                xi[x, iy, 2, 2] = weight
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, weight, max_contrib
                                )
                    elif ix1 > ix2:  # Shifting left
                        if x + ix2 >= 0 and x + ix1 < ncols:
                            # Upper left corner (2)
                            xx = x + ix2
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            weight = abs(delta - ix1) * w
                            if 0 <= yy < nrows and weight > 0:
                                xi[x, iy, 2, 0] = xx
                                xi[x, iy, 2, 1] = yy
                                xi[x, iy, 2, 2] = weight
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, weight, max_contrib
                                )
                            # Upper right corner (3)
                            xx = x + ix1
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            weight = w - abs(delta - ix1) * w
                            if 0 <= yy < nrows and weight > 0:
                                xi[x, iy, 3, 0] = xx
                                xi[x, iy, 3, 1] = yy
                                xi[x, iy, 3, 2] = weight
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, weight, max_contrib
                                )
                    else:  # No shift
                        xx = x + ix1
                        if 0 <= xx < ncols:
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            if 0 <= yy < nrows and w > 0:
                                xi[x, iy, 2, 0] = xx
                                xi[x, iy, 2, 1] = yy
                                xi[x, iy, 2, 2] = w
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, w, max_contrib
                                )

                elif iy == iy2:  # Case C: leaving row y
                    if ix1 < ix2:  # Shifting right
                        if x + ix1 >= 0 and x + ix2 < ncols:
                            # Lower right corner (1)
                            xx = x + ix1
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            weight = w - abs(delta - ix1) * w
                            if 0 <= yy < nrows and weight > 0:
                                xi[x, iy, 1, 0] = xx
                                xi[x, iy, 1, 1] = yy
                                xi[x, iy, 1, 2] = weight
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, weight, max_contrib
                                )
                            # Lower left corner (0)
                            xx = x + ix2
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            weight = abs(delta - ix1) * w
                            if 0 <= yy < nrows and weight > 0:
                                xi[x, iy, 0, 0] = xx
                                xi[x, iy, 0, 1] = yy
                                xi[x, iy, 0, 2] = weight
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, weight, max_contrib
                                )
                    elif ix1 > ix2:  # Shifting left
                        if x + ix2 >= 0 and x + ix1 < ncols:
                            # Lower left corner (0)
                            xx = x + ix2
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            weight = abs(delta - ix1) * w
                            if 0 <= yy < nrows and weight > 0:
                                xi[x, iy, 0, 0] = xx
                                xi[x, iy, 0, 1] = yy
                                xi[x, iy, 0, 2] = weight
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, weight, max_contrib
                                )
                            # Lower right corner (1)
                            xx = x + ix1
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            weight = w - abs(delta - ix1) * w
                            if 0 <= yy < nrows and weight > 0:
                                xi[x, iy, 1, 0] = xx
                                xi[x, iy, 1, 1] = yy
                                xi[x, iy, 1, 2] = weight
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, weight, max_contrib
                                )
                    else:  # No shift
                        xx = x + ix1
                        if 0 <= xx < ncols:
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            if 0 <= yy < nrows and w > 0:
                                xi[x, iy, 0, 0] = xx
                                xi[x, iy, 0, 1] = yy
                                xi[x, iy, 0, 2] = w
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, w, max_contrib
                                )

                else:  # Case B: fully inside row y
                    if ix1 < ix2:  # Shifting right
                        if x + ix1 >= 0 and x + ix2 < ncols:
                            # Lower right (1)
                            xx = x + ix1
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            weight = w - abs(delta - ix1) * w
                            if 0 <= yy < nrows and weight > 0:
                                xi[x, iy, 1, 0] = xx
                                xi[x, iy, 1, 1] = yy
                                xi[x, iy, 1, 2] = weight
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, weight, max_contrib
                                )
                            # Lower left (0)
                            xx = x + ix2
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            weight = abs(delta - ix1) * w
                            if 0 <= yy < nrows and weight > 0:
                                xi[x, iy, 0, 0] = xx
                                xi[x, iy, 0, 1] = yy
                                xi[x, iy, 0, 2] = weight
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, weight, max_contrib
                                )
                    elif ix1 > ix2:  # Shifting left
                        if x + ix2 >= 0 and x + ix1 < ncols:
                            # Lower right (1)
                            xx = x + ix2
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            weight = abs(delta - ix1) * w
                            if 0 <= yy < nrows and weight > 0:
                                xi[x, iy, 1, 0] = xx
                                xi[x, iy, 1, 1] = yy
                                xi[x, iy, 1, 2] = weight
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, weight, max_contrib
                                )
                            # Lower left (0)
                            xx = x + ix1
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            weight = w - abs(delta - ix1) * w
                            if 0 <= yy < nrows and weight > 0:
                                xi[x, iy, 0, 0] = xx
                                xi[x, iy, 0, 1] = yy
                                xi[x, iy, 0, 2] = weight
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, weight, max_contrib
                                )
                    else:  # No shift
                        xx = x + ix2
                        if 0 <= xx < ncols:
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            if 0 <= yy < nrows and w > 0:
                                xi[x, iy, 0, 0] = xx
                                xi[x, iy, 0, 1] = yy
                                xi[x, iy, 0, 2] = w
                                _add_to_zeta(
                                    zeta, m_zeta, xx, yy, x, iy, w, max_contrib
                                )

    return xi, zeta, m_zeta


# -----------------------------------------------------------------------------
# Linear system builders
# -----------------------------------------------------------------------------


@njit(cache=True)
def build_sL_system(
    xi: np.ndarray,
    zeta: np.ndarray,
    m_zeta: np.ndarray,
    sP: np.ndarray,
    mask: np.ndarray,
    im: np.ndarray,
    ncols: int,
    nrows: int,
    ny: int,
    osample: int,
) -> tuple:
    """Build the band-diagonal system for slit function."""
    bandwidth = 2 * osample
    l_Aij = np.zeros((ny, 4 * osample + 1))
    l_bj = np.zeros(ny)

    for iy in range(ny):
        for x in range(ncols):
            for n in range(4):
                ww = xi[x, iy, n, 2]
                if ww > 0:
                    xx = int(xi[x, iy, n, 0])
                    yy = int(xi[x, iy, n, 1])
                    if 0 <= xx < ncols and 0 <= yy < nrows:
                        m_count = m_zeta[xx, yy]
                        if m_count > 0:
                            for m in range(m_count):
                                xxx = int(zeta[xx, yy, m, 0])
                                jy = int(zeta[xx, yy, m, 1])
                                www = zeta[xx, yy, m, 2]
                                col_idx = jy - iy + bandwidth
                                if 0 <= col_idx < 4 * osample + 1:
                                    l_Aij[iy, col_idx] += (
                                        sP[xxx] * sP[x] * www * ww * mask[yy, xx]
                                    )
                            l_bj[iy] += im[yy, xx] * mask[yy, xx] * sP[x] * ww

    return l_Aij, l_bj


@njit(cache=True)
def build_sP_system(
    xi: np.ndarray,
    zeta: np.ndarray,
    m_zeta: np.ndarray,
    sL: np.ndarray,
    mask: np.ndarray,
    im: np.ndarray,
    ncols: int,
    nrows: int,
    ny: int,
    delta_x: int,
) -> tuple:
    """Build the band-diagonal system for spectrum."""
    bandwidth = 2 * delta_x
    nx = 4 * delta_x + 1
    p_Aij = np.zeros((ncols, nx))
    p_bj = np.zeros(ncols)

    for x in range(ncols):
        for iy in range(ny):
            for n in range(4):
                ww = xi[x, iy, n, 2]
                if ww > 0:
                    xx = int(xi[x, iy, n, 0])
                    yy = int(xi[x, iy, n, 1])
                    if 0 <= xx < ncols and 0 <= yy < nrows:
                        m_count = m_zeta[xx, yy]
                        if m_count > 0:
                            for m in range(m_count):
                                xxx = int(zeta[xx, yy, m, 0])
                                jy = int(zeta[xx, yy, m, 1])
                                www = zeta[xx, yy, m, 2]
                                col_idx = xxx - x + bandwidth
                                if 0 <= col_idx < nx:
                                    p_Aij[x, col_idx] += (
                                        sL[jy] * sL[iy] * www * ww * mask[yy, xx]
                                    )
                            p_bj[x] += im[yy, xx] * mask[yy, xx] * sL[iy] * ww

    return p_Aij, p_bj


@njit(cache=True)
def add_regularization(A: np.ndarray, n: int, bandwidth: int, lambda_: float):
    """Add first-order Tikhonov regularization to band matrix."""
    # First row
    A[0, bandwidth] += lambda_
    A[0, bandwidth + 1] -= lambda_
    # Middle rows
    for i in range(1, n - 1):
        A[i, bandwidth - 1] -= lambda_
        A[i, bandwidth] += 2 * lambda_
        A[i, bandwidth + 1] -= lambda_
    # Last row
    A[n - 1, bandwidth - 1] -= lambda_
    A[n - 1, bandwidth] += lambda_


@njit(cache=True)
def compute_model(
    zeta: np.ndarray,
    m_zeta: np.ndarray,
    sP: np.ndarray,
    sL: np.ndarray,
    ncols: int,
    nrows: int,
) -> np.ndarray:
    """Compute the model image from spectrum and slit function."""
    model = np.zeros((nrows, ncols))

    for y in range(nrows):
        for x in range(ncols):
            val = 0.0
            for m in range(m_zeta[x, y]):
                xx = int(zeta[x, y, m, 0])
                iy = int(zeta[x, y, m, 1])
                ww = zeta[x, y, m, 2]
                val += sP[xx] * sL[iy] * ww
            model[y, x] = val

    return model


@njit(cache=True)
def compute_uncertainties(
    zeta: np.ndarray,
    m_zeta: np.ndarray,
    im: np.ndarray,
    model: np.ndarray,
    mask: np.ndarray,
    ncols: int,
    nrows: int,
) -> np.ndarray:
    """Compute spectrum uncertainties from residuals."""
    unc = np.zeros(ncols)
    weights = np.zeros(ncols)

    for y in range(nrows):
        for x in range(ncols):
            if mask[y, x] > 0:
                resid = im[y, x] - model[y, x]
                for m in range(m_zeta[x, y]):
                    xx = int(zeta[x, y, m, 0])
                    ww = zeta[x, y, m, 2]
                    unc[xx] += resid * resid * ww
                    weights[xx] += ww

    for x in range(ncols):
        if weights[x] > 0:
            unc[x] = np.sqrt(unc[x] / weights[x] * nrows)
        else:
            unc[x] = 0.0

    return unc


# -----------------------------------------------------------------------------
# Band solver wrapper (uses scipy LAPACK)
# -----------------------------------------------------------------------------


def solve_band_system(A_band: np.ndarray, b: np.ndarray, bandwidth: int) -> np.ndarray:
    """Solve a band-diagonal system using scipy's LAPACK wrapper."""
    n = len(b)

    # Special case: diagonal system (bandwidth=0)
    if bandwidth == 0:
        diag = A_band[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(diag != 0, b / diag, 0.0)
        return result

    full_bandwidth = 2 * bandwidth + 1

    # Convert to scipy's banded format
    ab = np.zeros((full_bandwidth, n))
    for i in range(full_bandwidth):
        diag_offset = bandwidth - i
        if diag_offset >= 0:
            ab[i, diag_offset:] = A_band[: n - diag_offset, i]
        else:
            ab[i, : n + diag_offset] = A_band[-diag_offset:, i]

    try:
        return solve_banded((bandwidth, bandwidth), ab, b)
    except np.linalg.LinAlgError:
        # Matrix is singular - add small regularization to diagonal and retry
        diag_row = bandwidth  # main diagonal is at row 'bandwidth' in ab
        reg = 1e-10 * (np.abs(ab[diag_row]).max() + 1)
        ab[diag_row] += reg
        try:
            return solve_banded((bandwidth, bandwidth), ab, b)
        except np.linalg.LinAlgError:
            # Still singular - fall back to simple diagonal solve
            diag = A_band[:, bandwidth]
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.where(diag != 0, b / diag, 0.0)
            return result


# -----------------------------------------------------------------------------
# Internal extraction function
# -----------------------------------------------------------------------------


def _slit_func_curved_internal(
    im: np.ndarray,
    mask: np.ndarray,
    ycen: np.ndarray,
    ycen_offset: np.ndarray,
    y_lower_lim: int,
    osample: int,
    psf_curve: np.ndarray,
    lambda_sP: float = 0.0,
    lambda_sL: float = 0.1,
    maxiter: int = 20,
    threshold: float = 6.0,
    preset_slitfunc: np.ndarray = None,
) -> dict:
    """
    Internal extraction function - returns dict.

    See slitfunc_curved for the public API with cwrappers-compatible signature.
    """
    nrows, ncols = im.shape
    ny = osample * (nrows + 1) + 1

    # Ensure arrays are contiguous and correct dtype
    im = np.ascontiguousarray(im, dtype=np.float64)
    mask = np.ascontiguousarray(mask, dtype=np.float64)
    ycen = np.ascontiguousarray(ycen, dtype=np.float64)
    ycen_offset = np.ascontiguousarray(ycen_offset, dtype=np.int32)
    psf_curve = np.ascontiguousarray(psf_curve, dtype=np.float64)

    # Compute delta_x from curvature (vectorized)
    delta_x = 1 if lambda_sP > 0 else 0
    if np.any(psf_curve[:, 1:] != 0):
        y_vals = np.arange(-y_lower_lim, nrows - y_lower_lim + 1)
        shifts = np.abs(
            np.outer(psf_curve[:, 1], y_vals) + np.outer(psf_curve[:, 2], y_vals**2)
        )
        delta_x = max(delta_x, int(np.ceil(np.max(shifts))))

    # Build geometry tensors
    xi, zeta, m_zeta = xi_zeta_tensors(
        ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, psf_curve
    )

    # Initial spectrum estimate: sum along rows
    sP = np.sum(im * mask, axis=0)
    sP = np.maximum(sP, 1.0)

    # Initial slit function
    if preset_slitfunc is not None:
        sL = np.ascontiguousarray(preset_slitfunc, dtype=np.float64)
    else:
        sL = np.ones(ny) / osample

    # Iteration
    for iteration in range(maxiter):
        sP_old = sP.copy()

        # Solve for slit function (skip if preset)
        if preset_slitfunc is None:
            l_Aij, l_bj = build_sL_system(
                xi, zeta, m_zeta, sP, mask, im, ncols, nrows, ny, osample
            )
            diag_sum = np.sum(np.abs(l_Aij[:, 2 * osample]))
            lambda_L = lambda_sL * diag_sum / ny
            add_regularization(l_Aij, ny, 2 * osample, lambda_L)
            sL = solve_band_system(l_Aij, l_bj, 2 * osample)

            # Normalize slit function
            norm = np.sum(sL) / osample
            if norm > 0:
                sL = sL / norm

        # Solve for spectrum
        p_Aij, p_bj = build_sP_system(
            xi, zeta, m_zeta, sL, mask, im, ncols, nrows, ny, delta_x
        )
        if lambda_sP > 0:
            add_regularization(p_Aij, ncols, 2 * delta_x, lambda_sP)

        sP = solve_band_system(p_Aij, p_bj, 2 * delta_x)

        # Compute model and residuals
        model = compute_model(zeta, m_zeta, sP, sL, ncols, nrows)

        residual = (model - im) * mask
        sigma = np.sqrt(np.sum(residual**2) / np.sum(mask))

        # Update mask (outlier rejection)
        if threshold > 0:
            mask = np.where(np.abs(model - im) < threshold * sigma, 1.0, 0.0)

        # Check convergence
        change = np.percentile(np.abs(sP - sP_old), 99)
        median_sP = np.median(np.abs(sP))
        if change < 5e-5 * median_sP and iteration > 0:
            break

    # Compute uncertainties
    unc = compute_uncertainties(zeta, m_zeta, im, model, mask, ncols, nrows)

    # Zero out edge columns affected by curvature
    sP[:delta_x] = 0
    sP[-delta_x:] = 0 if delta_x > 0 else sP[-delta_x:]
    unc[:delta_x] = 0
    unc[-delta_x:] = 0 if delta_x > 0 else unc[-delta_x:]

    return {
        "spec": sP,
        "slitf": sL,
        "model": model,
        "mask": mask,
        "unc": unc,
        "niter": iteration + 1,
        "delta_x": delta_x,
    }


# -----------------------------------------------------------------------------
# Public API (compatible with cwrappers.slitfunc_curved)
# -----------------------------------------------------------------------------


def slitfunc_curved(
    img,
    ycen,
    p1,
    p2,
    lambda_sp,
    lambda_sf,
    osample,
    yrange,
    maxiter=20,
    gain=1,
    reject_threshold=6,
    preset_slitfunc=None,
):
    """Decompose an image into a spectrum and a slitfunction, image may be curved.

    This is a drop-in replacement for cwrappers.slitfunc_curved using Numba.

    Parameters
    ----------
    img : array[n, m]
        input image
    ycen : array[m]
        traces the center of the order (fractional part only)
    p1 : array[m] or scalar
        1st order curvature
    p2 : array[m] or scalar
        2nd order curvature
    lambda_sp : float
        smoothing factor spectrum
    lambda_sf : float
        smoothing factor slitfunction
    osample : int
        Subpixel oversampling factor
    yrange : tuple(int, int)
        number of pixels below and above the central line
    maxiter : int, optional
        maximum number of iterations
    gain : float, optional
        gain of the image (not used, for API compatibility)
    reject_threshold : float, optional
        outlier rejection threshold in sigma
    preset_slitfunc : array[ny], optional
        If provided, use this slit function instead of solving for it.

    Returns
    -------
    sp : array[m]
        spectrum
    sl : array[ny]
        slitfunction
    model : array[n, m]
        model image
    unc : array[m]
        spectrum uncertainties
    mask : array[n, m]
        bad pixel mask (True = rejected)
    info : array[5]
        convergence info [success, 0, 0, niter, delta_x]
    """
    img = np.ma.filled(img, 0).astype(np.float64)
    nrows, ncols = img.shape
    ylow, yhigh = yrange

    # Convert ycen to fractional part
    ycen = np.asarray(ycen, dtype=np.float64)
    ycen_int = np.floor(ycen).astype(int)
    ycen_frac = ycen - ycen_int

    # Build ycen_offset from integer parts
    ycen_offset = ycen_int - ycen_int[0]

    # Build psf_curve array [ncols, 3] with [0, p1, p2]
    psf_curve = np.zeros((ncols, 3), dtype=np.float64)
    if np.isscalar(p1):
        psf_curve[:, 1] = p1
    else:
        psf_curve[:, 1] = np.asarray(p1, dtype=np.float64)
    if np.isscalar(p2):
        psf_curve[:, 2] = p2
    else:
        psf_curve[:, 2] = np.asarray(p2, dtype=np.float64)

    # Initial mask (1 = good, 0 = bad)
    mask = np.ones((nrows, ncols), dtype=np.float64)

    # Call internal extraction
    result = _slit_func_curved_internal(
        im=img,
        mask=mask,
        ycen=ycen_frac,
        ycen_offset=ycen_offset.astype(np.int32),
        y_lower_lim=ylow,
        osample=osample,
        psf_curve=psf_curve,
        lambda_sP=lambda_sp,
        lambda_sL=lambda_sf,
        maxiter=maxiter,
        threshold=reject_threshold,
        preset_slitfunc=preset_slitfunc,
    )

    # Convert mask to boolean (True = rejected)
    out_mask = result["mask"] < 0.5

    # Build info array for compatibility
    info = np.array([1.0, 0.0, 0.0, float(result["niter"]), float(result["delta_x"])])

    return (
        result["spec"],
        result["slitf"],
        result["model"],
        result["unc"],
        out_mask,
        info,
    )
