"""
Pure-Python (Numba JIT) port of the slitdec extraction algorithm.

This is a transliteration of the *current* ``clib/slitdec.c``: pixel-centric
SLE fills, dense merge windows keyed on the per-pixel zeta ranges, zeta only
(no xi tensor). It exists so PyReduce can extract without a compiled C
extension; ``cwrappers.slitdec`` remains the reference implementation and this
module is expected to stay somewhat slower.

Set ``PYREDUCE_EXTRACTION=numba`` to select it as the extraction backend.

The zeta tensor is stored as three parallel arrays (x, iy, w) rather than the
C struct-of-three, which lets Numba/LLVM vectorize the contiguous walks.

Mask convention matches ``cwrappers``: 0 = bad pixel, 1 = good pixel.
"""

import math

import numpy as np
from numba import njit

_INT32_MAX = 2147483647
_INT32_MIN = -2147483648

# The fill loops are element-wise (no reductions), so fastmath buys
# vectorization without reassociating any accumulation. bandsol is left
# strict: it is unpivoted Gaussian elimination and numerically delicate.
_KW = {"cache": True, "nogil": True, "fastmath": True}
_KW_STRICT = {"cache": True, "nogil": True}


@njit(inline="always")
def _zeta_add(zx, ziy, zw, m_zeta, z_rng, ncols, nrows, x, iy, xx, yy, w):
    if 0 <= xx < ncols and 0 <= yy < nrows and w > 0:
        m = m_zeta[xx, yy]
        zx[xx, yy, m] = x
        ziy[xx, yy, m] = iy
        zw[xx, yy, m] = w
        m_zeta[xx, yy] = m + 1
        if iy < z_rng[xx, yy, 0]:
            z_rng[xx, yy, 0] = iy
        if iy > z_rng[xx, yy, 1]:
            z_rng[xx, yy, 1] = iy
        if x < z_rng[xx, yy, 2]:
            z_rng[xx, yy, 2] = x
        if x > z_rng[xx, yy, 3]:
            z_rng[xx, yy, 3] = x


@njit(**_KW_STRICT)
def _zeta_tensors(
    ncols,
    nrows,
    ycen,
    ycen_offset,
    y_lower_lim,
    osample,
    slitcurve,
    slitdeltas,
    zx,
    ziy,
    zw,
    m_zeta,
    z_rng,
):
    """Contribution of each oversampled slit subpixel to each detector pixel.

    z_rng packs (min_iy, max_iy, min_x, max_x) per pixel; the SLE fills use
    those ranges to merge into a dense window instead of key searching.
    """
    step = 1.0 / osample

    for x in range(ncols):
        for y in range(nrows):
            m_zeta[x, y] = 0
            z_rng[x, y, 0] = _INT32_MAX
            z_rng[x, y, 1] = _INT32_MIN
            z_rng[x, y, 2] = _INT32_MAX
            z_rng[x, y, 3] = _INT32_MIN

    for x in range(ncols):
        yc = ycen[x]
        iy2 = osample - int(math.floor(yc * osample))
        iy1 = iy2 - osample

        d1 = np.fmod(yc, step)  # numba has no math.fmod
        if d1 == 0:
            d1 = step
        d2 = step - d1

        dy = yc - math.floor((y_lower_lim + yc) / step) * step - step

        c1 = slitcurve[x, 1]
        c2 = slitcurve[x, 2]
        c3 = slitcurve[x, 3]
        c4 = slitcurve[x, 4]
        c5 = slitcurve[x, 5]
        off = ycen_offset[x]

        for y in range(nrows):
            iy1 += osample
            iy2 += osample
            dy -= step
            for iy in range(iy1, iy2 + 1):
                if iy == iy1:
                    w = d1
                elif iy == iy2:
                    w = d2
                else:
                    w = step
                dy += step
                t = dy - yc
                delta = (
                    t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) + slitdeltas[iy]
                )
                ix1 = int(delta)
                if delta > 0:
                    ix2 = ix1 + 1
                elif delta < 0:
                    ix2 = ix1 - 1
                else:
                    ix2 = ix1
                frac = abs(delta - ix1)

                if ix1 < ix2:
                    if x + ix1 >= 0 and x + ix2 < ncols:
                        xx = x + ix1
                        yy = y + off - ycen_offset[xx]
                        _zeta_add(
                            zx,
                            ziy,
                            zw,
                            m_zeta,
                            z_rng,
                            ncols,
                            nrows,
                            x,
                            iy,
                            xx,
                            yy,
                            w - frac * w,
                        )
                        xx = x + ix2
                        yy = y + off - ycen_offset[xx]
                        _zeta_add(
                            zx,
                            ziy,
                            zw,
                            m_zeta,
                            z_rng,
                            ncols,
                            nrows,
                            x,
                            iy,
                            xx,
                            yy,
                            frac * w,
                        )
                elif ix1 > ix2:
                    if x + ix2 >= 0 and x + ix1 < ncols:
                        xx = x + ix2
                        yy = y + off - ycen_offset[xx]
                        _zeta_add(
                            zx,
                            ziy,
                            zw,
                            m_zeta,
                            z_rng,
                            ncols,
                            nrows,
                            x,
                            iy,
                            xx,
                            yy,
                            frac * w,
                        )
                        xx = x + ix1
                        yy = y + off - ycen_offset[xx]
                        _zeta_add(
                            zx,
                            ziy,
                            zw,
                            m_zeta,
                            z_rng,
                            ncols,
                            nrows,
                            x,
                            iy,
                            xx,
                            yy,
                            w - frac * w,
                        )
                else:
                    xx = x + ix1
                    yy = y + off - ycen_offset[xx]
                    _zeta_add(
                        zx, ziy, zw, m_zeta, z_rng, ncols, nrows, x, iy, xx, yy, w
                    )
    return 0


@njit(**_KW_STRICT)
def _bandsol(a, r, n, nd):
    """Solve a band-diagonal system A x = r in place; band stored row-major.

    a[i, nd//2] is the main diagonal. Mirrors clib/slitdec.c bandsol,
    including the redundant final division of r[0] (the pivot is 1 by then).
    """
    nd2 = nd // 2

    for i in range(n - 1):
        aa = a[i, nd2]
        r[i] /= aa
        for j in range(nd):
            a[i, j] /= aa
        jmax = min(nd2 + 1, n - i)
        for j in range(1, jmax):
            aa = a[i + j, nd2 - j]
            r[i + j] -= r[i] * aa
            for k in range(nd - j):
                a[i + j, k] -= a[i, k + j] * aa

    aa = a[n - 1, nd2]
    r[n - 1] /= aa
    for i in range(n - 1, 0, -1):
        for j in range(1, min(nd2, i) + 1):
            r[i - j] -= r[i] * a[i - j, nd2 + j]
        r[i - 1] /= a[i - 1, nd2]

    r[0] /= a[0, nd2]
    return 0


@njit(**_KW)
def _fill_sl(
    ncols,
    nrows,
    osample,
    im,
    mask,
    sP,
    zx,
    ziy,
    zw,
    m_zeta,
    z_rng,
    l_Aij,
    l_bj,
    buf_w,
    buf_k,
):
    two_os = 2 * osample
    l_Aij[:, :] = 0.0
    l_bj[:] = 0.0

    for xx in range(ncols):
        for yy in range(nrows):
            mz = m_zeta[xx, yy]
            if mz <= 0 or mask[yy, xx] == 0:
                continue
            imv = im[yy, xx]
            k0 = z_rng[xx, yy, 0]
            rng = z_rng[xx, yy, 1] - k0

            if rng <= two_os:
                # Merge entries sharing a subpixel index into a dense window;
                # gaps between actual keys are zero and contribute nothing.
                for n in range(rng + 1):
                    buf_w[n] = 0.0
                for m in range(mz):
                    buf_w[ziy[xx, yy, m] - k0] += sP[zx[xx, yy, m]] * zw[xx, yy, m]
                # Symmetric matrix: upper bands only, mirrored after the fill.
                for m in range(rng + 1):
                    um = buf_w[m]
                    row = k0 + m
                    for n in range(rng - m + 1):
                        l_Aij[row, two_os + n] += um * buf_w[m + n]
                    l_bj[row] += imv * um
                continue

            # Over-wide list (extreme geometry): merge by key search
            nk = 0
            for m in range(mz):
                key = ziy[xx, yy, m]
                v = sP[zx[xx, yy, m]] * zw[xx, yy, m]
                hit = False
                for n in range(nk):
                    if buf_k[n] == key:
                        buf_w[n] += v
                        hit = True
                        break
                if not hit:
                    buf_k[nk] = key
                    buf_w[nk] = v
                    nk += 1
            for m in range(nk):
                iy = buf_k[m]
                um = buf_w[m]
                l_Aij[iy, two_os] += um * um
                for n in range(m + 1, nk):
                    iyn = buf_k[n]
                    lo = min(iy, iyn)
                    d = abs(iyn - iy)
                    l_Aij[lo, d + two_os] += buf_w[n] * um
                l_bj[iy] += imv * um


@njit(**_KW)
def _fill_sp(
    ncols,
    nrows,
    delta_x,
    im,
    mask,
    sL,
    zx,
    ziy,
    zw,
    m_zeta,
    z_rng,
    p_Aij,
    p_bj,
    buf_w,
    buf_k,
):
    two_dx = 2 * delta_x
    p_Aij[:, :] = 0.0
    p_bj[:] = 0.0

    for xx in range(ncols):
        for yy in range(nrows):
            mz = m_zeta[xx, yy]
            if mz <= 0 or mask[yy, xx] == 0:
                continue
            imv = im[yy, xx]
            k0 = z_rng[xx, yy, 2]
            rng = z_rng[xx, yy, 3] - k0

            if rng <= two_dx:
                for n in range(rng + 1):
                    buf_w[n] = 0.0
                for m in range(mz):
                    buf_w[zx[xx, yy, m] - k0] += sL[ziy[xx, yy, m]] * zw[xx, yy, m]
                for m in range(rng + 1):
                    um = buf_w[m]
                    row = k0 + m
                    for n in range(rng - m + 1):
                        p_Aij[row, two_dx + n] += um * buf_w[m + n]
                    p_bj[row] += imv * um
                continue

            nk = 0
            for m in range(mz):
                key = zx[xx, yy, m]
                v = sL[ziy[xx, yy, m]] * zw[xx, yy, m]
                hit = False
                for n in range(nk):
                    if buf_k[n] == key:
                        buf_w[n] += v
                        hit = True
                        break
                if not hit:
                    buf_k[nk] = key
                    buf_w[nk] = v
                    nk += 1
            for m in range(nk):
                x = buf_k[m]
                um = buf_w[m]
                p_Aij[x, two_dx] += um * um
                for n in range(m + 1, nk):
                    xn = buf_k[n]
                    lo = min(x, xn)
                    d = abs(xn - x)
                    p_Aij[lo, d + two_dx] += buf_w[n] * um
                p_bj[x] += imv * um


@njit(**_KW)
def _mirror_bands(a, n, half):
    """A[r+d, half-d] = A[r, half+d] for d = 1..half."""
    for m in range(1, half + 1):
        for i in range(n - m):
            a[i + m, half - m] = a[i, half + m]


@njit(**_KW)
def _regularize_smooth(a, n, half, lam):
    """First-derivative smoothing penalty on the tri-diagonal bands."""
    a[0, half] += lam
    a[0, half + 1] -= lam
    for i in range(1, n - 1):
        a[i, half - 1] -= lam
        a[i, half] += lam * 2.0
        a[i, half + 1] -= lam
    a[n - 1, half - 1] -= lam
    a[n - 1, half] += lam


@njit(**_KW)
def _regularize_diagonal(a, n, half):
    """Floor the diagonal so fully masked rows/columns do not go singular."""
    max_diag = 0.0
    for i in range(n):
        if a[i, half] > max_diag:
            max_diag = a[i, half]
    if max_diag > 0.0:
        min_diag = max_diag * 1.0e-10
        for i in range(n):
            if a[i, half] < min_diag:
                a[i, half] = min_diag


@njit(**_KW)
def _build_model(ncols, nrows, sP, sL, zx, ziy, zw, m_zeta, model):
    # x outer so the zeta tensor, by far the largest array, is read
    # sequentially instead of with a large stride
    for x in range(ncols):
        for y in range(nrows):
            mz = m_zeta[x, y]
            acc = 0.0
            for m in range(mz):
                acc += sP[zx[x, y, m]] * sL[ziy[x, y, m]] * zw[x, y, m]
            model[y, x] = acc


@njit(**_KW)
def _residual_rms(ncols, nrows, delta_x, im, model, mask):
    tmp = 0.0
    isum = 0
    for y in range(nrows):
        for x in range(delta_x, ncols - delta_x):
            if mask[y, x]:
                resid = model[y, x] - im[y, x]
                tmp += resid * resid
                isum += 1
    return math.sqrt(tmp / isum)


@njit(**_KW)
def _reject_outliers(ncols, nrows, delta_x, im, model, mask, cutoff):
    for y in range(nrows):
        for x in range(delta_x, ncols - delta_x):
            if abs(model[y, x] - im[y, x]) < cutoff:
                mask[y, x] = 1
            else:
                mask[y, x] = 0


@njit(**_KW)
def _uncertainty(
    ncols, nrows, im, model, mask, zx, ziy, zw, m_zeta, unc, norm, norm_sq
):
    for x in range(ncols):
        unc[x] = 0.0
        norm[x] = 0.0
        norm_sq[x] = 0.0
    for y in range(nrows):
        for x in range(ncols):
            if not mask[y, x]:
                continue
            tmp = im[y, x] - model[y, x]
            t2 = tmp * tmp
            for m in range(m_zeta[x, y]):
                xx = zx[x, y, m]
                ww = zw[x, y, m]
                unc[xx] += t2 * ww
                norm[xx] += ww
                norm_sq[xx] += ww * ww
    for x in range(ncols):
        d = norm[x] - norm_sq[x] / norm[x]
        unc[x] = math.sqrt(unc[x] / d * nrows)


@njit(**_KW_STRICT)
def _select(arr, k):
    """Value at sorted position k, matching the C quickselect (no interpolation)."""
    return np.partition(arr, k)[k]


@njit(**_KW_STRICT)
def _slitdec_core(
    ncols,
    nrows,
    im,
    pix_unc,
    mask,
    ycen,
    slitcurve,
    slitdeltas,
    osample,
    lambda_sP,
    lambda_sL,
    maxiter,
    kappa,
    use_preset,
    sP,
    sL,
    model,
    unc,
    info,
):
    sP_stop = 5e-5  # 99th percentile spectrum change relative to median
    sP_change = math.inf
    success = 1.0
    status = 0.0

    ny = osample * (nrows + 1) + 1
    y_lower_lim = nrows // 2

    # Maximum horizontal shift in detector pixels due to slit curvature.
    # Smoothing the spectrum needs delta_x >= 1 to have any neighbours.
    delta_x = 0 if lambda_sP == 0 else 1
    for x in range(ncols):
        for y in range(-y_lower_lim, nrows - y_lower_lim + 1):
            y2 = float(y * y)
            y3 = y2 * y
            y4 = y3 * y
            y5 = y4 * y
            tmp = math.ceil(
                abs(
                    y * slitcurve[x, 1]
                    + y2 * slitcurve[x, 2]
                    + y3 * slitcurve[x, 3]
                    + y4 * slitcurve[x, 4]
                    + y5 * slitcurve[x, 5]
                )
            )
            if tmp > delta_x:
                delta_x = int(tmp)
    for iy in range(ny):
        tmp = math.ceil(abs(slitdeltas[iy]))
        if tmp > delta_x:
            delta_x = int(tmp)

    nx = 4 * delta_x + 1
    if nx > ncols:
        info[0] = 0.0
        info[1] = sP_change
        info[2] = -2.0  # curvature too large
        info[3] = 0.0
        info[4] = delta_x
        return -1

    mzz = 3 * (osample + 1)
    zx = np.empty((ncols, nrows, mzz), dtype=np.int32)
    ziy = np.empty((ncols, nrows, mzz), dtype=np.int32)
    zw = np.empty((ncols, nrows, mzz), dtype=np.float64)
    m_zeta = np.empty((ncols, nrows), dtype=np.int32)
    z_rng = np.empty((ncols, nrows, 4), dtype=np.int32)

    l_Aij = np.empty((ny, 4 * osample + 1), dtype=np.float64)
    l_bj = np.empty(ny, dtype=np.float64)
    p_Aij = np.empty((ncols, nx), dtype=np.float64)
    p_bj = np.empty(ncols, dtype=np.float64)

    nbuf = max(mzz, nx)
    buf_w = np.empty(nbuf, dtype=np.float64)
    buf_k = np.empty(nbuf, dtype=np.int32)

    sP_old = np.empty(ncols, dtype=np.float64)
    sP_diff = np.empty(ncols, dtype=np.float64)
    unc_norm = np.empty(ncols, dtype=np.float64)
    unc_norm_sq = np.empty(ncols, dtype=np.float64)

    # Split ycen into integer row offset and sub-pixel remainder
    ycen_offset = np.empty(ncols, dtype=np.int32)
    for x in range(ncols):
        ycen_offset[x] = int(ycen[x])
        ycen[x] = ycen[x] - ycen_offset[x]

    _zeta_tensors(
        ncols,
        nrows,
        ycen,
        ycen_offset,
        y_lower_lim,
        osample,
        slitcurve,
        slitdeltas,
        zx,
        ziy,
        zw,
        m_zeta,
        z_rng,
    )

    if use_preset:
        norm = 0.0
        for iy in range(ny):
            norm += sL[iy]
        norm /= osample
        for iy in range(ny):
            sL[iy] /= norm

    pct99 = int(0.99 * (ncols - 1))
    kmed = (ncols - 1) // 2

    it = 0
    while True:
        if not use_preset:
            _fill_sl(
                ncols,
                nrows,
                osample,
                im,
                mask,
                sP,
                zx,
                ziy,
                zw,
                m_zeta,
                z_rng,
                l_Aij,
                l_bj,
                buf_w,
                buf_k,
            )
            _mirror_bands(l_Aij, ny, 2 * osample)

            diag_tot = 0.0
            for iy in range(ny):
                diag_tot += l_Aij[iy, 2 * osample]
            _regularize_smooth(l_Aij, ny, 2 * osample, lambda_sL * diag_tot / ny)
            _regularize_diagonal(l_Aij, ny, 2 * osample)

            _bandsol(l_Aij, l_bj, ny, 4 * osample + 1)

            norm = 0.0
            for iy in range(ny):
                sL[iy] = l_bj[iy]
                norm += sL[iy]
            norm /= osample
            for iy in range(ny):
                sL[iy] /= norm

        _fill_sp(
            ncols,
            nrows,
            delta_x,
            im,
            mask,
            sL,
            zx,
            ziy,
            zw,
            m_zeta,
            z_rng,
            p_Aij,
            p_bj,
            buf_w,
            buf_k,
        )
        _mirror_bands(p_Aij, ncols, 2 * delta_x)

        if lambda_sP > 0.0:
            _regularize_smooth(p_Aij, ncols, 2 * delta_x, lambda_sP)
        _regularize_diagonal(p_Aij, ncols, 2 * delta_x)

        _bandsol(p_Aij, p_bj, ncols, nx)

        for x in range(ncols):
            sP_old[x] = sP[x]
        for x in range(ncols):
            sP[x] = p_bj[x]
        for x in range(ncols):
            sP_diff[x] = abs(sP[x] - sP_old[x])

        sP_change = _select(sP_diff, pct99)
        sP_med = abs(_select(sP, kmed))

        _build_model(ncols, nrows, sP, sL, zx, ziy, zw, m_zeta, model)

        dev = _residual_rms(ncols, nrows, delta_x, im, model, mask)
        if kappa > 0:
            _reject_outliers(ncols, nrows, delta_x, im, model, mask, kappa * dev)

        prev = it
        it += 1
        if prev == 0:
            continue  # always do at least 2 iterations
        if it <= maxiter and sP_change > sP_stop * sP_med:
            continue
        break

    if it >= maxiter:
        status = -1.0  # ran out of iterations
        success = 0.0
    else:
        status = 1.0

    _uncertainty(
        ncols, nrows, im, model, mask, zx, ziy, zw, m_zeta, unc, unc_norm, unc_norm_sq
    )

    # Columns within delta_x of the edge have incomplete support
    for x in range(delta_x):
        sP[x] = 0.0
        unc[x] = 0.0
    for x in range(ncols - delta_x, ncols):
        sP[x] = 0.0
        unc[x] = 0.0

    info[0] = success
    info[1] = sP_change
    info[2] = status
    info[3] = it
    info[4] = delta_x
    return 0


def slitdec(
    im,
    pix_unc,
    mask,
    ycen,
    slitcurve,
    slitdeltas,
    osample=6,
    lambda_sP=0.0,
    lambda_sL=1.0,
    maxiter=20,
    kappa=10.0,
    preset_slitfunc=None,
):
    """Slit decomposition with slit characterization (Numba backend).

    Drop-in replacement for :func:`pyreduce.cwrappers.slitdec`: same signature,
    same returned dict, same mask convention (0 = bad, 1 = good).

    Returns
    -------
    dict with keys spectrum, slitfunction, model, uncertainty, info, mask,
    return_code.
    """
    im = np.ascontiguousarray(im, dtype=np.float64)
    pix_unc = np.ascontiguousarray(pix_unc, dtype=np.float64)

    nrows, ncols = im.shape
    if pix_unc.shape != (nrows, ncols):
        raise ValueError("pix_unc must have same shape as im")
    if mask.shape != (nrows, ncols):
        raise ValueError("mask must have same shape as im")
    if ycen.shape[0] != ncols:
        raise ValueError("ycen must have length ncols")

    slitcurve = np.asarray(slitcurve, dtype=np.float64)
    n_coeffs = slitcurve.shape[1]
    if slitcurve.shape[0] != ncols or not (1 <= n_coeffs <= 6):
        raise ValueError("slitcurve must have shape (ncols, n) where 1 <= n <= 6")

    osample = int(osample)
    ny = osample * (nrows + 1) + 1

    slitdeltas = np.asarray(slitdeltas, dtype=np.float64).ravel()
    if slitdeltas.size == nrows:
        pos = np.arange(ny) * (nrows - 1.0) / (ny - 1.0)
        slitdeltas_ny = np.interp(pos, np.arange(nrows), slitdeltas)
    elif slitdeltas.size == ny:
        slitdeltas_ny = slitdeltas.copy()
    else:
        raise ValueError(
            "slitdeltas must have length nrows or ny = osample * (nrows + 1) + 1"
        )
    slitdeltas_ny = np.ascontiguousarray(slitdeltas_ny, dtype=np.float64)

    slitcurve_padded = np.zeros((ncols, 6), dtype=np.float64)
    slitcurve_padded[:, :n_coeffs] = slitcurve

    # ycen and mask are modified in place, so pass copies.
    mask_copy = np.ascontiguousarray(mask, dtype=np.uint8).copy()
    ycen_copy = np.ascontiguousarray(ycen, dtype=np.float64).copy()

    use_preset = 0
    if preset_slitfunc is not None:
        use_preset = 1
        ps = np.asarray(preset_slitfunc, dtype=np.float64).ravel()
        if ps.size == ny:
            sL = ps.copy()
        elif ps.size == nrows:
            pos = np.arange(ny) * (nrows - 1.0) / (ny - 1.0)
            sL = np.interp(pos, np.arange(nrows), ps)
        else:
            raise ValueError(
                "preset_slitfunc must have length nrows or ny = osample * (nrows + 1) + 1"
            )
        sL = np.ascontiguousarray(sL, dtype=np.float64)
    else:
        sL = np.full(ny, 1.0 / osample, dtype=np.float64)

    sP = np.ones(ncols, dtype=np.float64)
    model = np.zeros((nrows, ncols), dtype=np.float64)
    unc = np.zeros(ncols, dtype=np.float64)
    info = np.zeros(5, dtype=np.float64)

    return_code = _slitdec_core(
        ncols,
        nrows,
        im,
        pix_unc,
        mask_copy,
        ycen_copy,
        slitcurve_padded,
        slitdeltas_ny,
        osample,
        float(lambda_sP),
        float(lambda_sL),
        int(maxiter),
        float(kappa),
        use_preset,
        sP,
        sL,
        model,
        unc,
        info,
    )

    return {
        "spectrum": sP,
        "slitfunction": sL,
        "model": model,
        "uncertainty": unc,
        "info": info,
        "mask": mask_copy,
        "return_code": int(return_code),
    }
