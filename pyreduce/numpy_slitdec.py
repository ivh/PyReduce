"""
Pure NumPy/SciPy port of the slitdec extraction algorithm.

Same algorithm as ``clib/slitdec.c`` and ``numba_slitdec.py``: pixel-centric
SLE fills, dense merge windows keyed on the per-pixel zeta ranges, zeta only
(no xi tensor). Here the pixel loops are replaced by dense array algebra, so
the module needs neither a C compiler nor Numba -- only numpy and scipy.

Set ``PYREDUCE_USE_NUMPY=1`` to select it as the extraction backend.

Everything geometric is built once per call and collapsed into a single dense
weight tensor ``T[m, j, p]``: the total zeta weight reaching detector pixel
``p`` from slit position ``k0_iy[p] + m`` and source column ``k0_x[p] + j``.
Both merge windows are then contractions of ``T`` -- no per-iteration scatter
over the millions of individual zeta entries::

    W_sL[m, p] = sum_j T[m, j, p] * sP[k0_x[p] + j]
    W_sP[j, p] = sum_m T[m, j, p] * sL[k0_iy[p] + m]

Pixels are held permuted into runs of equal ``k0``, which turns the normal
equation fills from indexed scatters into ``np.add.reduceat`` segment sums.

Mask convention matches ``cwrappers``: 0 = bad pixel, 1 = good pixel. Masked
pixels are multiplied out rather than skipped, so a heavily masked frame costs
the same as a clean one.

Memory is dominated by ``T`` at ``K * Kx`` doubles per detector pixel (~24 for
CRIRES-like geometry at osample=6), plus the transient zeta lists it is built
from.
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
    COO entries hit consecutive pixels.

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
        xsb = np.broadcast_to(xs, (nc, nrows, nk))
        for ix, ww in ((ix1, w_a), (ix2, w_b)):
            xx = xs + ix
            np.clip(xx, 0, ncols - 1, out=xx)
            # subpixel iy contributes to row yy of column xx, where
            # y + ycen_offset[x] == yy + ycen_offset[xx]
            yy = yrow[None, :, None] + off - ycen_offset[xx]
            keep = inb & (ww > 0) & (yy >= 0) & (yy < nrows)
            xx *= nrows
            xx += yy
            # boolean indexing, not flatnonzero + take: one sequential scan
            # instead of an index array plus four gathers, ~4x faster here
            n = int(np.count_nonzero(keep))
            sl = slice(total, total + n)
            pix[sl] = xx[keep]
            src_iy[sl] = iy[keep]
            src_x[sl] = xsb[keep]
            zw[sl] = ww[keep]
            total += n

    return pix[:total], src_x[:total], src_iy[:total], zw[:total]


def _window(pix, key, npix):
    """Per-pixel window base k0, uniform width, and the relative keys.

    Replaces the C's per-pixel ``z_rng`` scan. A window wider than a pixel's
    own key range costs nothing numerically: the extra slots stay zero. The
    width comes off the shifted keys rather than a second scatter-max, since
    the shift has to be computed for the tensor index anyway.
    """
    k0 = np.full(npix, _INT64_MAX, dtype=np.int64)
    np.minimum.at(k0, pix, key)
    # pixels with no zeta entries never contribute; keep their base in range
    np.copyto(k0, 0, where=k0 == _INT64_MAX)

    rel = key - k0[pix]
    width = int(rel.max()) + 1 if rel.size else 1
    return k0, width, rel


def _group(k0):
    """Permute pixels into ascending runs of equal k0.

    Returns the permutation, the distinct k0 values and the run starts, i.e.
    exactly what ``np.add.reduceat`` consumes. Grouping is what lets the
    normal-equation fills be segment sums instead of indexed scatters.

    Sorted narrow, on purpose: numpy counting-sorts 16-bit integers, which is
    an order of magnitude faster than the radix sort it uses on wider ones.
    """
    narrow = np.int16 if k0.max() < 0x8000 else np.int32
    order = np.argsort(k0.astype(narrow, copy=False), kind="stable")
    ks = k0[order]
    starts = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1])))
    return order, ks[starts], starts


def _fill_system(W, imv, starts, uk, n, nband):
    """Accumulate the normal equations of one merged window into upper bands.

    ``Aup[d, i]`` holds ``A[i, i + d]``; the matrix is symmetric, so only the
    upper bands are built -- which is exactly what scipy consumes. ``W`` and
    ``imv`` are in grouped-pixel order, so each band entry is a run sum over
    the pixels sharing a window base. Rows ``uk + m`` past ``n - 1`` can only
    receive zero-weight window slots and are dropped.
    """
    width, npix = W.shape
    Aup = np.zeros((nband, n))
    bj = np.zeros(n)
    buf = np.empty(npix)
    for m in range(width):
        cut = int(np.searchsorted(uk, n - m))
        if cut == 0:
            break
        rows = uk[:cut] + m
        np.multiply(imv, W[m], out=buf)
        bj[rows] += np.add.reduceat(buf, starts)[:cut]
        for d in range(width - m):
            np.multiply(W[m], W[m + d], out=buf)
            Aup[d, rows] += np.add.reduceat(buf, starts)[:cut]
    return Aup, bj


def _regularize_smooth(Aup, lam):
    """First-derivative smoothing penalty, upper-band form of the C's fill."""
    n = Aup.shape[1]
    Aup[0, 0] += lam
    Aup[0, 1 : n - 1] += lam * 2.0
    Aup[0, n - 1] += lam
    Aup[1, : n - 1] -= lam


def _regularize_diagonal(Aup):
    """Floor the diagonal so fully masked rows/columns do not go singular."""
    max_diag = Aup[0].max()
    if max_diag > 0.0:
        np.maximum(Aup[0], max_diag * 1.0e-10, out=Aup[0])


def _solve(Aup, b, u):
    """Solve the symmetric band system given its upper bands.

    Cholesky (``solveh_banded``) replaces the C's unpivoted ``bandsol``; if the
    matrix is not positive definite it falls back to a pivoted LU on the
    mirrored band instead of propagating the garbage bandsol would produce.
    """
    n = b.size
    if u == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            return b / Aup[0]

    ab = np.zeros((u + 1, n))
    for d in range(u + 1):
        ab[u - d, d:] = Aup[d, : n - d]
    try:
        return solveh_banded(ab, b, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        full = np.zeros((2 * u + 1, n))
        full[: u + 1] = ab
        for d in range(1, u + 1):
            full[u + d, : n - d] = Aup[d, : n - d]
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
    runs along the slit first, as the C's zeta layout does, and are then
    permuted into runs of equal sL window base.
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

    k0_iy, K, rel_iy = _window(pix, src_iy, npix)
    k0_x, Kx, rel_x = _window(pix, src_x, npix)
    # The C's band layout assumes K <= 2*osample+1 and Kx <= 2*delta_x+1 and
    # falls back to a key search beyond that; here the band simply widens.
    # Band 1 must exist for the smoothing penalty.
    nband_l = max(K, 2)
    nband_p = max(Kx, 2) if lambda_sP > 0.0 else max(Kx, 1)

    # Pixels are grouped by their sL window base for the rest of the call, so
    # the sL fill can use run sums. The sP system is grouped a second time on
    # top of that, since its bases run the other way.
    order, uk_l, starts_l = _group(k0_iy)
    rank = np.empty(npix, dtype=np.int64)
    rank[order] = np.arange(npix)
    # Group for the sP system in the original pixel order, where k0_x is
    # nearly sorted already, then compose: sorting the permuted copy costs
    # several times as much.
    order_x, uk_p, starts_p = _group(k0_x)
    order_p = rank[order_x]
    imv_p = imT.reshape(-1)[order_x]

    k0_iy = k0_iy[order]
    k0_x = k0_x[order]
    imv = imT.reshape(-1)[order]

    # T[m, j, p]: total zeta weight reaching pixel p from slit position
    # k0_iy[p] + m and source column k0_x[p] + j. Both merge windows are
    # contractions of it, so the per-entry lists are needed only to build it.
    np.take(rank, pix, out=pix)
    rel_x *= npix
    rel_x += pix
    rel_iy *= Kx * npix
    rel_iy += rel_x
    T = np.bincount(rel_iy, weights=zw, minlength=K * Kx * npix)
    T = T.reshape(K, Kx, npix)
    del pix, src_x, src_iy, zw, rel_iy, rel_x, rank
    Tsum = T.sum(axis=0)
    # A cell (m, j, p) can hold at most one zeta entry: j fixes the source
    # column and (p, m) then fix the source row and subpixel. So the squared
    # weights the uncertainty needs come straight off T, no second scatter.
    Tsq = np.einsum("mjp,mjp->jp", T, T)

    if use_preset:
        sL /= sL.sum() / osample

    pct99 = int(0.99 * (ncols - 1))
    kmed = (ncols - 1) // 2

    sP_pad = np.zeros(ncols + Kx)
    sL_pad = np.zeros(ny + K)
    sPg = np.empty((Kx, npix))
    sLg = np.empty((K, npix))
    model = np.empty(npix)
    it = 0
    while True:
        maskf = maskT.reshape(-1)[order].astype(np.float64)

        sP_pad[:ncols] = sP
        for j in range(Kx):
            np.take(sP_pad, k0_x + j, out=sPg[j])

        if not use_preset:
            W = np.einsum("mjp,jp->mp", T, sPg)
            W *= maskf
            l_Aij, l_bj = _fill_system(W, imv, starts_l, uk_l, ny, nband_l)

            diag_tot = l_Aij[0].sum()
            _regularize_smooth(l_Aij, lambda_sL * diag_tot / ny)
            _regularize_diagonal(l_Aij)

            sL[:] = _solve(l_Aij, l_bj, nband_l - 1)
            sL /= sL.sum() / osample

        sL_pad[:ny] = sL
        for m in range(K):
            np.take(sL_pad, k0_iy + m, out=sLg[m])
        # kept unmasked, because the model needs every pixel
        W = np.einsum("mjp,mp->jp", T, sLg)
        Wm = W[:, order_p]
        Wm *= maskf[order_p]
        p_Aij, p_bj = _fill_system(Wm, imv_p, starts_p, uk_p, ncols, nband_p)

        if lambda_sP > 0.0:
            _regularize_smooth(p_Aij, lambda_sP)
        _regularize_diagonal(p_Aij)

        sP_old = sP.copy()
        sP[:] = _solve(p_Aij, p_bj, nband_p - 1)

        sP_change = _select(np.abs(sP - sP_old), pct99)
        sP_med = abs(_select(sP, kmed))

        # The model is the same window contracted with the new spectrum:
        # model[p] = sum_j W[j, p] * sP[k0_x[p] + j]. Padding sP absorbs the
        # bases that run past the last column, where W is zero anyway.
        sP_pad[:ncols] = sP
        np.take(sP_pad, k0_x, out=sPg[0])
        np.multiply(W[0], sPg[0], out=model)
        for j in range(1, Kx):
            np.take(sP_pad, k0_x + j, out=sPg[j])
            np.multiply(W[j], sPg[j], out=W[j])
            model += W[j]
        modelT.reshape(-1)[order] = model

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

    # Grouped by source column via T's second axis: pixel p feeds column
    # k0_x[p] + j with total weight Tsum[j, p] (Tsq for the squared weights).
    maskf = maskT.reshape(-1)[order].astype(np.float64)
    t2 = (imv - model) ** 2
    t2 *= maskf
    acc = np.zeros((3, ncols + Kx))
    tmp = np.empty(npix)
    for j in range(Kx):
        col = k0_x + j
        np.multiply(t2, Tsum[j], out=tmp)
        acc[0] += np.bincount(col, weights=tmp, minlength=ncols + Kx)
        np.multiply(maskf, Tsum[j], out=tmp)
        acc[1] += np.bincount(col, weights=tmp, minlength=ncols + Kx)
        np.multiply(maskf, Tsq[j], out=tmp)
        acc[2] += np.bincount(col, weights=tmp, minlength=ncols + Kx)
    norm, norm_sq = acc[1, :ncols], acc[2, :ncols]
    with np.errstate(divide="ignore", invalid="ignore"):
        unc[:] = np.sqrt(acc[0, :ncols] / (norm - norm_sq / norm) * nrows)

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
