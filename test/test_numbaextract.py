"""
Numba-accelerated curved slit extraction.

This is a green-field reimplementation of the slit_func_2d_xi_zeta_bd algorithm
using Numba JIT compilation for performance.

The algorithm decomposes a 2D spectral image into:
- sP: 1D spectrum (flux vs wavelength)
- sL: 1D slit illumination function (flux vs cross-dispersion)

such that: model[x,y] = sum over contributing subpixels of sP[x'] * sL[iy] * weight
"""

import numpy as np
import pytest
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

    This follows the C implementation closely:
    - xi[x, iy, corner]: where does subpixel (x, iy) contribute?
      corners: 0=LL, 1=LR, 2=UL, 3=UR
    - zeta[x, y, m]: which subpixels contribute to pixel (x, y)?

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
    """
    Solve a band-diagonal system using scipy's LAPACK wrapper.

    Parameters
    ----------
    A_band : array of shape (n, 2*bandwidth+1)
        Band matrix in row-major form (our convention)
    b : array of shape (n,)
        Right-hand side
    bandwidth : int
        Half-bandwidth (number of sub/super diagonals)

    Returns
    -------
    x : array of shape (n,)
        Solution
    """
    n = len(b)
    full_bandwidth = 2 * bandwidth + 1

    # Convert to scipy's banded format:
    # scipy wants shape (l+u+1, n) where l=u=bandwidth
    # Row i contains diagonal offset (bandwidth - i)
    ab = np.zeros((full_bandwidth, n))
    for i in range(full_bandwidth):
        diag_offset = bandwidth - i  # positive = above main diagonal
        if diag_offset >= 0:
            # Upper diagonal
            ab[i, diag_offset:] = A_band[: n - diag_offset, i]
        else:
            # Lower diagonal
            ab[i, : n + diag_offset] = A_band[-diag_offset:, i]

    return solve_banded((bandwidth, bandwidth), ab, b)


# -----------------------------------------------------------------------------
# Main extraction function
# -----------------------------------------------------------------------------


def slit_func_curved(
    im: np.ndarray,
    mask: np.ndarray,
    ycen: np.ndarray,
    ycen_offset: np.ndarray,
    y_lower_lim: int,
    osample: int,
    psf_curve: np.ndarray,
    lambda_sP: float = 0.0,
    lambda_sL: float = 1.0,
    maxiter: int = 20,
    threshold: float = 10.0,
) -> dict:
    """
    Extract spectrum and slit function from a curved slit image.

    Parameters
    ----------
    im : array of shape (nrows, ncols)
        Input image swath
    mask : array of shape (nrows, ncols)
        Bad pixel mask (1 = good, 0 = bad)
    ycen : array of shape (ncols,)
        Fractional y-position of order center (0 <= ycen < 1)
    ycen_offset : array of shape (ncols,)
        Integer y-offset for order packing
    y_lower_lim : int
        Pixels below the center line
    osample : int
        Oversampling factor for slit function
    psf_curve : array of shape (ncols, 3)
        Curvature coefficients [c0, c1, c2] per column
    lambda_sP : float
        Spectrum smoothing parameter
    lambda_sL : float
        Slit function smoothing parameter
    maxiter : int
        Maximum iterations
    threshold : float
        Outlier rejection threshold in sigma

    Returns
    -------
    dict with keys:
        'spec': extracted spectrum
        'slitf': slit function
        'model': reconstructed model
        'mask': final mask
        'unc': uncertainties
        'niter': number of iterations
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
    sP = np.maximum(sP, 1.0)  # avoid zeros

    # Initial slit function: flat
    sL = np.ones(ny) / osample

    # Iteration
    for iteration in range(maxiter):
        sP_old = sP.copy()

        # Solve for slit function
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

        # Update mask
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
    }


# -----------------------------------------------------------------------------
# Tests
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
        # from subpixels that contribute to it
        for y in range(nrows):
            for x in range(ncols):
                total_weight = 0.0
                for m in range(m_zeta[x, y]):
                    total_weight += zeta[x, y, m, 2]
                # Each pixel should get weight ~1.0 (one detector pixel worth)
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

        # Column sums should be proportional to spectrum (same ratio for all columns)
        model_column_sums = np.sum(model, axis=0)

        # The ratio should be constant across columns
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

        # Compare extracted vs true spectrum (ignoring edges)
        margin = 10
        extracted = result["spec"][margin:-margin]
        true = s["true_spec"][margin:-margin]

        # Should correlate well
        correlation = np.corrcoef(extracted, true)[0, 1]
        assert correlation > 0.99, f"Spectrum correlation {correlation:.4f} too low"

        # Relative error - expect some bias from regularization
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

        # RMS should be small relative to signal
        rel_rms = rms / signal_rms
        assert rel_rms < 0.2, f"Relative RMS {rel_rms:.3f} too high"

    def test_with_curvature(self):
        """Test extraction with slit curvature."""
        np.random.seed(123)

        ncols = 100
        nrows = 10
        osample = 10
        ny = osample * (nrows + 1) + 1

        # Spectrum with variation
        true_spec = 1000.0 + 200.0 * np.sin(np.arange(ncols) * 0.1)

        # Gaussian slit function
        iy = np.arange(ny)
        true_slitf = np.exp(-(((iy - ny / 2) / (ny / 6)) ** 2))
        true_slitf /= np.sum(true_slitf) / osample

        # Add curvature: linear tilt
        ycen = np.full(ncols, 0.5)
        ycen_offset = np.zeros(ncols, dtype=np.int32)
        y_lower_lim = nrows // 2
        psf_curve = np.zeros((ncols, 3))
        psf_curve[:, 1] = 0.05  # linear tilt coefficient

        # Build model with curvature
        xi, zeta, m_zeta = xi_zeta_tensors(
            ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, psf_curve
        )
        model = compute_model(zeta, m_zeta, true_spec, true_slitf, ncols, nrows)

        # Add noise
        noise = np.median(model) * 0.01
        im = model + np.random.randn(nrows, ncols) * noise
        mask = np.ones((nrows, ncols))

        # Extract
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

        # Check convergence
        assert result["niter"] < 20

        # Check spectrum correlation
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
    nrows, ncols = swath_img.shape

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
    mask = np.ones((nrows, ncols))
    ycen_offset = np.zeros(ncols, dtype=np.int32)
    psf_curve = np.zeros((ncols, 3))
    y_lower_lim = ylow

    result_numba = slit_func_curved(
        swath_img,
        mask,
        ycen,
        ycen_offset,
        y_lower_lim,
        osample,
        psf_curve,
        lambda_sP=lambda_sp,
        lambda_sL=lambda_sf,
        maxiter=20,
    )

    sp_numba = result_numba["spec"]
    sl_numba = result_numba["slitf"]
    model_numba = result_numba["model"]

    # Compare shapes
    assert sp_c.shape == sp_numba.shape, (
        f"Spectrum shapes differ: {sp_c.shape} vs {sp_numba.shape}"
    )
    assert sl_c.shape == sl_numba.shape, (
        f"Slitfunc shapes differ: {sl_c.shape} vs {sl_numba.shape}"
    )

    # Compare spectra (ignore edges where curvature effects may differ)
    margin = 10
    sp_c_mid = sp_c[margin:-margin]
    sp_numba_mid = sp_numba[margin:-margin]

    # Spectra should be highly correlated
    correlation = np.corrcoef(sp_c_mid, sp_numba_mid)[0, 1]
    assert correlation > 0.999, f"Spectrum correlation {correlation:.6f} too low"

    # Relative difference
    sp_rel_diff = np.abs(sp_c_mid - sp_numba_mid) / np.maximum(sp_c_mid, 1)
    median_diff = np.median(sp_rel_diff)
    max_diff = np.max(sp_rel_diff)

    print("\nSpectrum comparison:")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Median rel diff: {median_diff:.4f}")
    print(f"  Max rel diff: {max_diff:.4f}")

    # Slit function comparison
    sl_rel_diff = np.abs(sl_c - sl_numba) / np.maximum(np.abs(sl_c), 1e-10)
    print("Slitfunc comparison:")
    print(f"  Median rel diff: {np.median(sl_rel_diff):.4f}")
    print(f"  Max rel diff: {np.max(sl_rel_diff):.4f}")

    # Model comparison
    model_rel_diff = np.abs(model_c - model_numba) / np.maximum(model_c, 1)
    print("Model comparison:")
    print(f"  Median rel diff: {np.median(model_rel_diff):.4f}")
    print(f"  Max rel diff: {np.max(model_rel_diff):.4f}")

    # Save results to debug directory
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
        unc_numba=result_numba["unc"],
        mask_numba=result_numba["mask"],
        niter_numba=result_numba["niter"],
    )
    print(f"Saved results to {outfile}")

    # Spectra should match within a few percent
    assert median_diff < 0.05, f"Median spectrum difference {median_diff:.4f} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
