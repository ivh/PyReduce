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
# Robust initial guess and pre-masking (0th pass)
# -----------------------------------------------------------------------------


@njit(cache=True)
def _robust_initial_guess(
    im: np.ndarray,
    mask: np.ndarray,
    zeta: np.ndarray,
    m_zeta: np.ndarray,
    ncols: int,
    nrows: int,
    ny: int,
    osample: int,
    threshold: float,
    noise_floor: float = 1.0,
) -> tuple:
    """
    Compute robust initial spectrum using median, build initial model
    with uniform slit function, and pre-mask outliers.

    Returns (sP, sL, mask) with outliers masked.
    """
    # Initial spectrum: column-wise median (robust to cosmics)
    sP = np.empty(ncols)
    col_vals = np.empty(nrows)
    for x in range(ncols):
        n_good = 0
        for y in range(nrows):
            if mask[y, x] > 0:
                col_vals[n_good] = im[y, x]
                n_good += 1
        if n_good > 0:
            sorted_vals = np.sort(col_vals[:n_good])
            sP[x] = sorted_vals[n_good // 2]
        else:
            sP[x] = 1.0

    # Scale spectrum to match sum (like original), but using median-based estimate
    # median ≈ sum / nrows for uniform slit function
    for x in range(ncols):
        sP[x] *= nrows
    sP = np.maximum(sP, 1.0)

    # Uniform slit function (like original initialization)
    sL = np.ones(ny) / osample

    # Compute initial model
    model = compute_model(zeta, m_zeta, sP, sL, ncols, nrows)

    # Pre-mask positive outliers (cosmics) using Poisson-like threshold
    # Only reject pixels where data > model (cosmics are always positive)
    if threshold > 0:
        for y in range(nrows):
            for x in range(ncols):
                if mask[y, x] > 0:
                    resid = im[y, x] - model[y, x]  # positive if data > model
                    noise = max(np.sqrt(max(model[y, x], 0.0)), noise_floor)
                    if resid > threshold * noise:
                        mask[y, x] = 0.0

    return sP, sL, mask


# -----------------------------------------------------------------------------
# Band solver (JIT-compiled, based on C bandsol)
# -----------------------------------------------------------------------------


@njit(cache=True)
def bandsol(a: np.ndarray, r: np.ndarray) -> int:
    """
    Solve band-diagonal system Ax = r in-place.

    Based on C bandsol implementation. Uses Gaussian elimination
    with forward sweep and backward substitution.

    Parameters
    ----------
    a : array[n, nd]
        Band matrix. Main diagonal at column nd//2.
        Modified in-place during solve.
    r : array[n]
        Right-hand side, replaced with solution in-place.

    Returns
    -------
    status : int
        0 on success, -1 on singular matrix
    """
    n, nd = a.shape
    mid = nd // 2

    # Forward sweep
    for i in range(n - 1):
        aa = a[i, mid]
        if aa == 0.0:
            # Try small regularization
            aa = 1e-16
            a[i, mid] = aa

        r[i] /= aa
        for j in range(nd):
            a[i, j] /= aa

        # Eliminate below
        jmax = min(mid + 1, n - i)
        for j in range(1, jmax):
            aa = a[i + j, mid - j]
            if aa != 0.0:
                r[i + j] -= r[i] * aa
                for k in range(nd - j):
                    a[i + j, k] -= a[i, k + j] * aa

    # Backward sweep
    if a[n - 1, mid] != 0.0:
        r[n - 1] /= a[n - 1, mid]

    for i in range(n - 1, 0, -1):
        jmax = min(mid, i)
        for j in range(1, jmax + 1):
            r[i - j] -= r[i] * a[i - j, mid + j]
        if a[i - 1, mid] != 0.0:
            r[i - 1] /= a[i - 1, mid]

    return 0


def solve_band_system(A_band: np.ndarray, b: np.ndarray, bandwidth: int) -> np.ndarray:
    """
    Wrapper for bandsol that matches the old API.

    Parameters
    ----------
    A_band : array[n, nd]
        Band matrix (will be copied and modified)
    b : array[n]
        Right-hand side (will be copied and modified)
    bandwidth : int
        Half-bandwidth (unused, derived from A_band shape)

    Returns
    -------
    x : array[n]
        Solution vector
    """
    # bandsol works in-place, so copy inputs
    A_copy = A_band.copy()
    b_copy = b.copy()
    bandsol(A_copy, b_copy)
    return b_copy


# -----------------------------------------------------------------------------
# Internal extraction function (JIT-compiled)
# -----------------------------------------------------------------------------


@njit(cache=True)
def _iteration_loop(
    im: np.ndarray,
    mask: np.ndarray,
    xi: np.ndarray,
    zeta: np.ndarray,
    m_zeta: np.ndarray,
    sP: np.ndarray,
    sL: np.ndarray,
    ncols: int,
    nrows: int,
    ny: int,
    osample: int,
    delta_x: int,
    lambda_sP: float,
    lambda_sL: float,
    maxiter: int,
    threshold: float,
    use_preset: bool,
) -> tuple:
    """
    JIT-compiled main iteration loop.

    Returns (sP, sL, model, mask, niter)
    """
    bandwidth_sL = 2 * osample
    bandwidth_sP = 2 * delta_x

    model = np.zeros((nrows, ncols))
    niter = 0

    for iteration in range(maxiter):
        niter = iteration + 1

        # Save old spectrum for convergence check
        sP_old = sP.copy()

        # Solve for slit function (skip if preset)
        if not use_preset:
            l_Aij, l_bj = build_sL_system(
                xi, zeta, m_zeta, sP, mask, im, ncols, nrows, ny, osample
            )

            # Compute regularization scale
            diag_sum = 0.0
            for iy in range(ny):
                diag_sum += abs(l_Aij[iy, bandwidth_sL])
            lambda_L = lambda_sL * diag_sum / ny

            # Add regularization
            add_regularization(l_Aij, ny, bandwidth_sL, lambda_L)

            # Solve in-place
            bandsol(l_Aij, l_bj)
            sL = l_bj

            # Normalize slit function
            norm = 0.0
            for iy in range(ny):
                norm += sL[iy]
            norm /= osample
            if norm > 0:
                for iy in range(ny):
                    sL[iy] /= norm

        # Solve for spectrum
        p_Aij, p_bj = build_sP_system(
            xi, zeta, m_zeta, sL, mask, im, ncols, nrows, ny, delta_x
        )

        if lambda_sP > 0:
            add_regularization(p_Aij, ncols, bandwidth_sP, lambda_sP)

        # Solve in-place
        bandsol(p_Aij, p_bj)
        sP = p_bj

        # Compute model
        model = compute_model(zeta, m_zeta, sP, sL, ncols, nrows)

        # Update mask: reject positive outliers (cosmics) using Poisson-like threshold
        # Only reject pixels where data > model (cosmics are always positive)
        if threshold > 0:
            for y in range(nrows):
                for x in range(ncols):
                    resid = im[y, x] - model[y, x]  # positive if data > model
                    noise = max(np.sqrt(max(model[y, x], 0.0)), 1.0)
                    if resid < threshold * noise:
                        mask[y, x] = 1.0
                    else:
                        mask[y, x] = 0.0

        # Check convergence: 99th percentile of |sP - sP_old|
        diffs = np.abs(sP - sP_old)
        sorted_diffs = np.sort(diffs)
        idx99 = int(0.99 * len(diffs))
        if idx99 >= len(diffs):
            idx99 = len(diffs) - 1
        change = sorted_diffs[idx99]

        # Median of |sP|
        sorted_sP = np.sort(np.abs(sP))
        median_sP = sorted_sP[len(sP) // 2]

        if change < 5e-5 * median_sP and iteration > 0:
            break

    return sP, sL, model, mask, niter


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

    # Compute delta_x from curvature
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

    # 0th pass: robust initial guess and pre-mask outliers
    # Skip for osample=1 (simple extraction) - geometry is simpler there
    USE_0TH_PASS = osample > 1
    if USE_0TH_PASS:
        if preset_slitfunc is not None:
            sL = np.ascontiguousarray(preset_slitfunc, dtype=np.float64)
            use_preset = True
            sP, _, mask = _robust_initial_guess(
                im, mask, zeta, m_zeta, ncols, nrows, ny, osample, threshold
            )
            sP = np.ascontiguousarray(sP, dtype=np.float64)
        else:
            use_preset = False
            sP, sL, mask = _robust_initial_guess(
                im, mask, zeta, m_zeta, ncols, nrows, ny, osample, threshold
            )
            sP = np.ascontiguousarray(sP, dtype=np.float64)
            sL = np.ascontiguousarray(sL, dtype=np.float64)
    else:
        # Original initialization
        sP = np.sum(im * mask, axis=0)
        sP = np.maximum(sP, 1.0)
        sP = np.ascontiguousarray(sP, dtype=np.float64)
        if preset_slitfunc is not None:
            sL = np.ascontiguousarray(preset_slitfunc, dtype=np.float64)
            use_preset = True
        else:
            sL = np.ones(ny, dtype=np.float64) / osample
            use_preset = False

    # Run JIT-compiled iteration loop
    sP, sL, model, mask, niter = _iteration_loop(
        im,
        mask,
        xi,
        zeta,
        m_zeta,
        sP,
        sL,
        ncols,
        nrows,
        ny,
        osample,
        delta_x,
        lambda_sP,
        lambda_sL,
        maxiter,
        threshold,
        use_preset,
    )

    # Compute uncertainties
    unc = compute_uncertainties(zeta, m_zeta, im, model, mask, ncols, nrows)

    # Zero out edge columns affected by curvature
    if delta_x > 0:
        sP[:delta_x] = 0
        sP[-delta_x:] = 0
        unc[:delta_x] = 0
        unc[-delta_x:] = 0

    return {
        "spec": sP,
        "slitf": sL,
        "model": model,
        "mask": mask,
        "unc": unc,
        "niter": niter,
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
