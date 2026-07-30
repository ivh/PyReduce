"""
Pure NumPy/SciPy port of the slitdec extraction algorithm.

Same algorithm as ``clib/slitdec.c`` and ``numba_slitdec.py``: pixel-centric
SLE fills, dense merge windows keyed on the per-pixel zeta ranges, zeta only
(no xi tensor). Here the pixel loops are replaced by scatter/gather over the
zeta tensor in flat COO form (``pix``, ``src_x``, ``src_iy``, ``w``), so the
module needs neither a C compiler nor Numba -- only numpy and scipy.

Set ``PYREDUCE_USE_NUMPY=1`` to select it as the extraction backend.

Everything geometric is built once per call; per iteration only the
sP/sL-weighted values change. The two SLE fills merge each pixel's zeta list
into a dense window of uniform width K (window slots between actual keys are
zero and contribute exactly nothing, as in the C), then accumulate the pairs
of one band offset at a time with a bincount over the per-pixel window base.

Mask convention matches ``cwrappers``: 0 = bad pixel, 1 = good pixel. Masked
pixels are multiplied out rather than skipped, so a heavily masked frame costs
the same as a clean one.

Memory scales as ~50 bytes per zeta entry (~14 per detector pixel), i.e. of
the same order as the C's zeta tensor for the same swath.
"""

import numpy as np
from scipy.linalg import solve_banded, solveh_banded

_INT64_MAX = np.iinfo(np.int64).max

# Target number of subpixel candidates held live while building the geometry.
_CHUNK_ELEMENTS = 2_000_000


def _geometry(
    ncols, nrows, ycen, ycen_offset, y_lower_lim, osample, slitcurve, slitdeltas
):
    """Zeta tensor in flat COO form: (pix, src_x, src_iy, w).

    ``pix = xx * nrows + yy`` indexes the detector pixel in column-major
    ("transposed") order, matching the C's zeta layout, so that consecutive
    COO entries hit consecutive pixels and the per-iteration scatters stay
    cache friendly.

    Emits the same (column, weight) pairs as the C's three-branch ``zeta_add``
    sequence: A = (x + ix1, w - frac*w) and B = (x + ix2, frac*w), with B
    dropped by the w > 0 test when delta == 0. Order within a pixel differs,
    which only affects the rounding of the merged sums.
    """
    step = 1.0 / osample
    nk = osample + 1
    cap = 2 * ncols * nrows * nk
    pix = np.empty(cap, dtype=np.int64)
    src_x = np.empty(cap, dtype=np.int64)
    src_iy = np.empty(cap, dtype=np.int64)
    zw = np.empty(cap, dtype=np.float64)

    iy1_init = (osample - np.floor(ycen * osample) - osample).astype(np.int64)
    d1 = np.fmod(ycen, step)
    d1 = np.where(d1 == 0.0, step, d1)
    d2 = step - d1
    dy_start = ycen - np.floor((y_lower_lim + ycen) / step) * step - step

    yrow = np.arange(nrows)
    iy_base = (yrow + 1) * osample
    kk = np.arange(nk)

    total = 0
    chunk = max(1, _CHUNK_ELEMENTS // (nrows * nk))
    for x0 in range(0, ncols, chunk):
        x1 = min(x0 + chunk, ncols)
        nc = x1 - x0
        xs = np.arange(x0, x1)[:, None, None]
        off = ycen_offset[x0:x1, None, None]

        # dy must be accumulated, not evaluated in closed form: the closed
        # form drifts ~1e-12 from the C's sequential +=/-=, enough to flip
        # int(delta) when delta lands on an integer. Cumsum over the same
        # sequence of +-step, in the same order, is bit-exact. One -step per
        # row followed by osample+1 x +step, matching the C's loop nesting.
        inc = np.empty((nc, nrows * (osample + 2) + 1))
        inc[:, 0] = dy_start[x0:x1]
        inc[:, 1:] = step
        inc[:, 1 :: osample + 2] = -step
        np.cumsum(inc, axis=1, out=inc)
        dy = inc[:, 1:].reshape(nc, nrows, osample + 2)[:, :, 1:]

        # iy1/iy2 depend on x only and both step by osample per row, so the
        # (x, y, iy) triples form a regular grid of exactly osample+1 subpixels
        iy = iy1_init[x0:x1, None, None] + iy_base[None, :, None] + kk
        wgt = np.empty((nc, 1, nk))
        wgt[:] = step
        wgt[:, 0, 0] = d1[x0:x1]
        wgt[:, 0, osample] = d2[x0:x1]
        wgt = np.broadcast_to(wgt, (nc, nrows, nk))

        t = dy - ycen[x0:x1, None, None]
        delta = t * slitcurve[x0:x1, 5, None, None]
        for c in range(4, 0, -1):
            delta += slitcurve[x0:x1, c, None, None]
            delta *= t
        delta += slitdeltas[iy]

        ix1 = delta.astype(np.int64)
        ix2 = ix1 + np.sign(delta).astype(np.int64)
        frac = np.abs(delta - ix1)
        # Both columns must be in range, else emit neither; when ix1 == ix2
        # this reduces to zeta_add's own bounds check
        inb = (np.minimum(ix1, ix2) + xs >= 0) & (np.maximum(ix1, ix2) + xs < ncols)

        w_b = frac * wgt
        w_a = wgt - w_b
        for ix, ww in ((ix1, w_a), (ix2, w_b)):
            xx = xs + ix
            np.clip(xx, 0, ncols - 1, out=xx)
            # subpixel iy contributes to row yy of column xx, where
            # y + ycen_offset[x] == yy + ycen_offset[xx]
            yy = yrow[None, :, None] + off - ycen_offset[xx]
            keep = inb & (ww > 0) & (yy >= 0) & (yy < nrows)
            idx = np.flatnonzero(keep)
            n = idx.size
            sl = slice(total, total + n)
            xx *= nrows
            xx += yy
            np.take(xx.reshape(-1), idx, out=pix[sl])
            np.take(iy.reshape(-1), idx, out=src_iy[sl])
            np.take(ww.reshape(-1), idx, out=zw[sl])
            np.floor_divide(idx, nrows * nk, out=src_x[sl])
            src_x[sl] += x0
            total += n

    return pix[:total], src_x[:total], src_iy[:total], zw[:total]


def _window(pix, key, npix):
    """Per-pixel window base k0, uniform width K, and merge bin indices.

    Replaces the C's per-pixel ``z_rng`` scan. A window wider than a pixel's
    own key range costs nothing numerically: the extra slots stay zero. The
    width comes off the shifted keys rather than a second scatter-max, since
    the shift has to be computed for the bin indices anyway.
    """
    k0 = np.full(npix, _INT64_MAX, dtype=np.int64)
    np.minimum.at(k0, pix, key)
    # pixels with no zeta entries never contribute; keep their base in range
    np.copyto(k0, 0, where=k0 == _INT64_MAX)

    rel = key - k0[pix]
    width = int(rel.max()) + 1 if rel.size else 1
    rel *= npix
    rel += pix
    return k0, width, rel


def _merge(bin_idx, vals, width, npix):
    """Merged window W[k, pix]: entries sharing a key add into one slot."""
    W = np.bincount(bin_idx, weights=vals, minlength=width * npix)
    return W.reshape(width, npix)


def _fill_system(W, k0, n, imv, nband):
    """Accumulate the normal equations of one merged window into upper bands.

    ``Aup[i, d]`` holds ``A[i, i + d]``; the matrix is symmetric, so only the
    upper bands are built -- which is exactly what scipy consumes. Rows
    ``k0[p] + m`` past ``n - 1`` can only receive zero-weight window slots, so
    the slices simply truncate.
    """
    width, npix = W.shape
    Aup = np.zeros((n, nband))
    bj = np.zeros(n)
    buf = np.empty(npix)
    for m in range(width):
        np.multiply(imv, W[m], out=buf)
        cnt = np.bincount(k0, weights=buf, minlength=n)
        bj[m:n] += cnt[: n - m]
        for d in range(width - m):
            np.multiply(W[m], W[m + d], out=buf)
            cnt = np.bincount(k0, weights=buf, minlength=n)
            Aup[m:n, d] += cnt[: n - m]
    return Aup, bj


def _regularize_smooth(Aup, lam):
    """First-derivative smoothing penalty, upper-band form of the C's fill."""
    n = Aup.shape[0]
    Aup[0, 0] += lam
    Aup[1 : n - 1, 0] += lam * 2.0
    Aup[n - 1, 0] += lam
    Aup[: n - 1, 1] -= lam


def _regularize_diagonal(Aup):
    """Floor the diagonal so fully masked rows/columns do not go singular."""
    max_diag = Aup[:, 0].max()
    if max_diag > 0.0:
        np.maximum(Aup[:, 0], max_diag * 1.0e-10, out=Aup[:, 0])


def _solve(Aup, b, u):
    """Solve the symmetric band system given its upper bands.

    Cholesky (``solveh_banded``) replaces the C's unpivoted ``bandsol``; if the
    matrix is not positive definite it falls back to a pivoted LU on the
    mirrored band instead of propagating the garbage bandsol would produce.
    """
    n = b.size
    if u == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            return b / Aup[:, 0]

    ab = np.zeros((u + 1, n))
    for d in range(u + 1):
        ab[u - d, d:] = Aup[: n - d, d]
    try:
        return solveh_banded(ab, b, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        full = np.zeros((2 * u + 1, n))
        full[: u + 1] = ab
        for d in range(1, u + 1):
            full[u + d, : n - d] = Aup[: n - d, d]
        return solve_banded((u, u), full, b, check_finite=False)


def _select(arr, k):
    """Value at sorted position k, matching the C quickselect (no interpolation)."""
    return np.partition(arr, k)[k]


def _slitdec_core(
    ncols,
    nrows,
    imT,
    maskT,
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
    modelT,
    unc,
    info,
):
    """Iterate sL/sP to convergence.

    Images are transposed (ncols, nrows) here so that the flat pixel index
    runs along the slit first, as the C's zeta layout does.
    """
    sP_stop = 5e-5  # 99th percentile spectrum change relative to median
    sP_change = np.inf

    ny = osample * (nrows + 1) + 1
    y_lower_lim = nrows // 2
    npix = ncols * nrows

    # Maximum horizontal shift in detector pixels due to slit curvature.
    # Smoothing the spectrum needs delta_x >= 1 to have any neighbours.
    delta_x = 0 if lambda_sP == 0 else 1
    yv = np.arange(-y_lower_lim, nrows - y_lower_lim + 1, dtype=np.float64)[:, None]
    y2 = yv * yv
    y3 = y2 * yv
    y4 = y3 * yv
    y5 = y4 * yv
    shift = (
        yv * slitcurve[None, :, 1]
        + y2 * slitcurve[None, :, 2]
        + y3 * slitcurve[None, :, 3]
        + y4 * slitcurve[None, :, 4]
        + y5 * slitcurve[None, :, 5]
    )
    delta_x = max(delta_x, int(np.ceil(np.abs(shift).max())))
    delta_x = max(delta_x, int(np.ceil(np.abs(slitdeltas).max())))

    nx = 4 * delta_x + 1
    if nx > ncols:
        info[0] = 0.0
        info[1] = sP_change
        info[2] = -2.0  # curvature too large
        info[3] = 0.0
        info[4] = delta_x
        return -1

    # Split ycen into integer row offset and sub-pixel remainder
    ycen_offset = ycen.astype(np.int64)
    ycen -= ycen_offset

    pix, src_x, src_iy, zw = _geometry(
        ncols, nrows, ycen, ycen_offset, y_lower_lim, osample, slitcurve, slitdeltas
    )

    imv = imT.reshape(-1)
    maskv = maskT.reshape(-1)

    k0_iy, K, bin_iy = _window(pix, src_iy, npix)
    k0_x, Kx, bin_x = _window(pix, src_x, npix)
    # The C's band layout assumes K <= 2*osample+1 and Kx <= 2*delta_x+1 and
    # falls back to a key search beyond that; here the band simply widens.
    # Band 1 must exist for the smoothing penalty.
    nband_l = max(K, 2)
    nband_p = max(Kx, 2) if lambda_sP > 0.0 else max(Kx, 1)

    if use_preset:
        sL /= sL.sum() / osample

    pct99 = int(0.99 * (ncols - 1))
    kmed = (ncols - 1) // 2

    sP_pad = np.zeros(ncols + Kx)
    it = 0
    while True:
        maskf = maskv.astype(np.float64)

        if not use_preset:
            W = _merge(bin_iy, sP[src_x] * zw, K, npix)
            W *= maskf
            l_Aij, l_bj = _fill_system(W, k0_iy, ny, imv, nband_l)

            diag_tot = l_Aij[:, 0].sum()
            _regularize_smooth(l_Aij, lambda_sL * diag_tot / ny)
            _regularize_diagonal(l_Aij)

            sL[:] = _solve(l_Aij, l_bj, nband_l - 1)
            sL /= sL.sum() / osample

        # kept unmasked, because the model needs every pixel
        W = _merge(bin_x, sL[src_iy] * zw, Kx, npix)
        p_Aij, p_bj = _fill_system(W * maskf, k0_x, ncols, imv, nband_p)

        if lambda_sP > 0.0:
            _regularize_smooth(p_Aij, lambda_sP)
        _regularize_diagonal(p_Aij)

        sP_old = sP.copy()
        sP[:] = _solve(p_Aij, p_bj, nband_p - 1)

        sP_change = _select(np.abs(sP - sP_old), pct99)
        sP_med = abs(_select(sP, kmed))

        # The model is the same window contracted with the new spectrum:
        # model[p] = sum_m W[m, p] * sP[k0_x[p] + m]. Padding sP absorbs the
        # bases that run past the last column, where W is zero anyway.
        sP_pad[:ncols] = sP
        model = modelT.reshape(-1)
        np.multiply(W[0], sP_pad[k0_x], out=model)
        for m in range(1, Kx):
            model += W[m] * sP_pad[k0_x + m]

        core = slice(delta_x, ncols - delta_x)
        resid = modelT[core] - imT[core]
        good = maskT[core] != 0
        dev = np.sqrt(np.sum(resid * resid, where=good) / np.count_nonzero(good))
        if kappa > 0:
            maskT[core] = np.abs(resid) < kappa * dev

        prev = it
        it += 1
        if prev == 0:
            continue  # always do at least 2 iterations
        if it <= maxiter and sP_change > sP_stop * sP_med:
            continue
        break

    if it >= maxiter:
        success, status = 0.0, -1.0  # ran out of iterations
    else:
        success, status = 1.0, 1.0

    maskf = maskv.astype(np.float64)
    t2 = (imv - modelT.reshape(-1)) ** 2
    t2 *= maskf
    wm = zw * maskf[pix]
    unc[:] = np.bincount(src_x, weights=t2[pix] * zw, minlength=ncols)
    norm = np.bincount(src_x, weights=wm, minlength=ncols)
    norm_sq = np.bincount(src_x, weights=wm * zw, minlength=ncols)
    with np.errstate(divide="ignore", invalid="ignore"):
        unc[:] = np.sqrt(unc / (norm - norm_sq / norm) * nrows)

    # Columns within delta_x of the edge have incomplete support
    if delta_x > 0:
        sP[:delta_x] = sP[ncols - delta_x :] = 0.0
        unc[:delta_x] = unc[ncols - delta_x :] = 0.0

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
    """Slit decomposition with slit characterization (NumPy backend).

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

    # ycen and mask are modified in place, so work on copies. The transposed
    # layout puts the slit direction first, as the zeta tensor expects.
    ycen_copy = np.ascontiguousarray(ycen, dtype=np.float64).copy()
    maskT = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8).T)
    imT = np.ascontiguousarray(im.T)

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
    modelT = np.zeros((ncols, nrows), dtype=np.float64)
    unc = np.zeros(ncols, dtype=np.float64)
    info = np.zeros(5, dtype=np.float64)

    return_code = _slitdec_core(
        ncols,
        nrows,
        imT,
        maskT,
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
        modelT,
        unc,
        info,
    )

    return {
        "spectrum": sP,
        "slitfunction": sL,
        "model": np.ascontiguousarray(modelT.T),
        "uncertainty": unc,
        "info": info,
        "mask": np.ascontiguousarray(maskT.T),
        "return_code": int(return_code),
    }
