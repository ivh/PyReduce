"""
ANDES iq15 sampling test: measure FWHM of synthetic LFC peaks across the
detector for the new single-fiber "master_rotfix_iq15" optical models
(Y_iq15, J_iq15, H_iq15 bands of the E2E simulator). process_band() is
reused for other single-fiber LFC frames, see sampl_vis.py.

Adapted from andes_yjh_sampl.py. The old 75-fibre PyReduce traces do not
apply to the new optical models, but the iq15 HDF files carry the trace
directly (translation_x/translation_y per order), so the single fiber is
extracted with a simple aperture sum along the model trace. The trace is
refined per order by the median centroid offset of the LFC lines (the
affine translation refers to the field corner, not the fiber centre).

The FWHM (in pixels) of the Gaussian fits is a direct measure of the optical
design's spectral sampling; combined with the HDF wavelength model it gives
the resolving power R.
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import CubicSpline, griddata, make_lsq_spline
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import find_peaks
from scipy.stats import binned_statistic_2d

# --- Configuration ---
CHANNELS = ["Y", "J", "H"]

DATA_DIR = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
HDF_DIR = os.path.expanduser("~/ANDES/E2E/src/HDF")
LFC_DIR = os.path.expanduser("~/ANDES/E2E/iq15")
PLOT = int(os.environ.get("PYREDUCE_PLOT", "0"))

APERTURE_HALF = 5  # extraction half-height in pixels
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

# Empirical LSF: stack lines with different sub-pixel phases into an
# oversampled profile per cell (order x ELSF_XBIN columns).
ELSF_HALF = 8.0  # stacking half-window in px; LFC lines are >= 23 px apart
ELSF_XBIN = 512
ELSF_KNOT = 0.25  # spline knot spacing in px
ELSF_MAX_GAP = 1.5  # max sub-pixel phase gap between lines, in knot spacings
ELSF_MARGIN = 1.5  # recentring moves lines by <= 1 px, keep the spline inside the data
ELSF_MIN_LINES = 12
ELSF_ITER = 3


def load_trace_model(hdf_path: str, fiber: str = "fiber_1") -> dict:
    """Read trace and wavelength model per order from an ANDES E2E HDF file.

    Returns dict mapping order number m -> (x, y, wl) sorted by x,
    where x/y are detector pixels and wl is wavelength in microns.
    """
    model = {}
    with h5py.File(hdf_path, "r") as f:
        for key in f["CCD_1"][fiber]:
            if not key.startswith("order"):
                continue
            m = int(key.replace("order", ""))
            data = np.array(f[f"CCD_1/{fiber}/{key}"])
            tx = data["translation_x"]
            ty = data["translation_y"]
            wl = data["wavelength"]
            order = np.argsort(tx)
            model[m] = (tx[order], ty[order], wl[order])
    return model


def compute_resolving_power(
    x_peaks: np.ndarray,
    fwhm_px: np.ndarray,
    wl_model: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Convert FWHM in pixels to resolving power R = lambda / delta_lambda."""
    # The HDF has only ~15 samples per order; finite differences of those are
    # off by up to ~6% in dlambda/dx between samples.
    wl = CubicSpline(*wl_model)
    wl_at_peak = wl(x_peaks)
    delta_wl = fwhm_px * np.abs(wl(x_peaks, 1))
    R = wl_at_peak / delta_wl
    return R


def gaussian(x, amp, mu, sigma, baseline):
    return baseline + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_peaks(spec, prominence_frac=0.1, window=6, max_sigma=5.0):
    """Find LFC peaks and fit each with a Gaussian.

    Returns (x_peaks, fwhms, baselines).
    """
    empty = (np.array([]), np.array([]), np.array([]))
    flux = np.asarray(spec, dtype=float)
    mask = np.isfinite(flux)
    if mask.sum() < 50:
        return empty

    idx = np.arange(flux.size)
    valid = idx[mask]
    fmax = np.nanmax(flux[mask])
    if not np.isfinite(fmax) or fmax <= 0:
        return empty

    peaks, _ = find_peaks(
        np.where(mask, flux, 0.0),
        prominence=prominence_frac * fmax,
        distance=4,
    )
    lo, hi = valid.min() + window, valid.max() - window
    peaks = peaks[(peaks >= lo) & (peaks <= hi)]

    x_out, fwhm_out, base_out = [], [], []
    for p in peaks:
        xs = idx[p - window : p + window + 1].astype(float)
        ys = flux[p - window : p + window + 1]
        m = np.isfinite(ys)
        if m.sum() < 5:
            continue
        xs = xs[m]
        ys = ys[m]
        baseline0 = np.min(ys)
        amp0 = np.max(ys) - baseline0
        p0 = [amp0, float(p), window / 4, baseline0]
        try:
            popt, _ = curve_fit(gaussian, xs, ys, p0=p0, maxfev=2000)
        except (RuntimeError, ValueError):
            continue
        sigma = abs(popt[2])
        if not np.isfinite(sigma) or sigma <= 0 or sigma > max_sigma:
            continue
        x_out.append(popt[1])
        fwhm_out.append(sigma * SIGMA_TO_FWHM)
        base_out.append(popt[3])

    return np.array(x_out), np.array(fwhm_out), np.array(base_out)


def line_cutouts(spec, x_peaks, baselines, half=ELSF_HALF):
    """Baseline-subtracted pixel cutouts around each line, normalised to unit flux.

    Returns list of (pixel_x, values, centre).
    """
    out = []
    n = spec.size
    for mu, b in zip(x_peaks, baselines, strict=True):
        lo = int(np.ceil(mu - half))
        hi = int(np.floor(mu + half))
        if lo < 0 or hi >= n:
            continue
        ys = spec[lo : hi + 1] - b
        tot = ys.sum()
        if not np.isfinite(tot) or tot <= 0:
            continue
        out.append((np.arange(lo, hi + 1, dtype=float), ys / tot, mu))
    return out


def fit_profile_spline(dx, val, half, knot=ELSF_KNOT):
    k = 3
    inner = np.arange(-half + knot, half - knot / 2, knot)
    t = np.r_[[-half] * (k + 1), inner, [half] * (k + 1)]
    order = np.argsort(dx)
    return make_lsq_spline(dx[order], val[order], t, k=k)


def _scale_offset(model, ys):
    A = np.column_stack([model, np.ones_like(model)])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return coef, ys - A @ coef


def recentre_line(px, ys, mu, spl):
    """Shift, scale and offset one line cutout to best match the stacked profile."""
    half = spl.t[-1]

    def model(d):
        x = px - mu - d
        return np.where(np.abs(x) <= half, spl(np.clip(x, -half, half)), 0.0)

    res = minimize_scalar(
        lambda d: np.sum(_scale_offset(model(d), ys)[1] ** 2),
        bounds=(-0.5, 0.5),
        method="bounded",
    )
    (a, b), _ = _scale_offset(model(res.x), ys)
    return mu + res.x, (ys - b) / a


def max_phase_gap(x):
    """Largest gap in the sub-pixel phases of positions x (on the unit circle)."""
    ph = np.sort(np.mod(x, 1.0))
    return np.max(np.diff(np.r_[ph, ph[0] + 1.0]))


def empirical_lsf(cutouts, n_iter=ELSF_ITER, half=ELSF_HALF, knot=ELSF_KNOT):
    """Oversampled pixel-integrated LSF from stacked line cutouts.

    The line centres from the per-line Gaussian fits are biased with sub-pixel
    phase if the profile is not Gaussian, which would blur the stack, so each
    line is re-registered against the current stacked profile.

    Returns (spline, dx, val) of the final iteration, or None if the lines do
    not sample the sub-pixel phases densely enough for the knot spacing (the
    spline then rings), which happens where the comb spacing is close to an
    integer number of pixels.
    """
    pxs = [c[0] for c in cutouts]
    yss = [c[1] for c in cutouts]
    mus = np.array([c[2] for c in cutouts])
    if max_phase_gap(mus) > ELSF_MAX_GAP * knot:
        return None
    lim = half - ELSF_MARGIN
    for it in range(n_iter):
        dx = np.concatenate([px - m for px, m in zip(pxs, mus, strict=True)])
        val = np.concatenate(yss)
        keep = np.abs(dx) <= lim
        try:
            spl = fit_profile_spline(dx[keep], val[keep], lim, knot)
        except (ValueError, np.linalg.LinAlgError):
            return None
        if not np.all(np.isfinite(spl.c)):
            return None
        if it == n_iter - 1:
            break
        new = [
            recentre_line(px, ys, m, spl)
            for px, ys, m in zip(pxs, yss, mus, strict=True)
        ]
        mus = np.array([n[0] for n in new])
        mus -= np.median(mus - np.array([c[2] for c in cutouts]))
        yss = [n[1] for n in new]
    return spl, dx[keep], val[keep]


def profile_fwhm(spl):
    """FWHM from the half-maximum crossings of the profile."""
    xf = np.linspace(spl.t[0], spl.t[-1], 16001)
    yf = spl(xf)
    i = int(np.argmax(yf))
    hm = 0.5 * yf[i]
    left = np.nonzero(yf[:i] < hm)[0]
    right = np.nonzero(yf[i:] < hm)[0]
    if not left.size or not right.size:
        return np.nan
    il = left[-1]
    ir = i + right[0]
    xl = np.interp(hm, [yf[il], yf[il + 1]], [xf[il], xf[il + 1]])
    xr = np.interp(hm, [yf[ir], yf[ir - 1]], [xf[ir], xf[ir - 1]])
    return xr - xl


def plot_detector_map(
    pts, vals, shape, scatter_xy, levels, fmt, cmap, cbar_label, title, out_png
):
    """Linearly interpolated map of values given at detector positions pts.

    Does nothing if out_png is falsy.
    """
    if not out_png:
        return None
    nrow, ncol = shape
    grid_step = 32
    gx = np.arange(0, ncol, grid_step)
    gy = np.arange(0, nrow, grid_step)
    GX, GY = np.meshgrid(gx, gy)
    grid = griddata(pts, vals, (GX, GY), method="linear")

    fig, ax = plt.subplots(figsize=(9, 8))
    vmin, vmax = np.nanpercentile(grid, [2, 98])
    im = ax.imshow(
        grid,
        origin="lower",
        extent=(0, ncol, 0, nrow),
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    if callable(levels):
        levels = levels(vmin, vmax)
    cs = ax.contour(
        GX, GY, grid, levels=levels, colors="white", linewidths=0.7, alpha=0.8
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt=fmt)
    ax.scatter(*scatter_xy, s=1, color="k", alpha=0.15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label)
    ax.set_xlabel("x [pixel]  (dispersion)")
    ax.set_ylabel("y [pixel]  (cross-dispersion)")
    ax.set_title(title)
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"Saved map: {out_png}")
    if not PLOT:
        plt.close(fig)
    return fig


def fwhm_levels(vmin, vmax):
    return np.arange(np.floor(vmin * 10) / 10, vmax + 0.1, 0.1)


def r_levels(vmin, vmax):
    return np.arange(
        np.floor(vmin / 5000) * 5000, np.ceil(vmax / 5000) * 5000 + 1, 5000
    )


def trace_offset(img, x, y_model, half=10, min_flux=50.0):
    """Median offset between flux centroid and model trace at LFC line columns."""
    nrow = img.shape[0]
    offsets = []
    for xi, ym in zip(x, y_model, strict=False):
        lo = int(ym) - half
        hi = int(ym) + half + 1
        if lo < 0 or hi > nrow:
            continue
        col = img[lo:hi, xi].astype(float)
        tot = col.sum()
        if tot < min_flux:
            continue
        cen = (col * np.arange(lo, hi)).sum() / tot
        offsets.append(cen - ym)
    return np.median(offsets) if offsets else 0.0


def extract_order(img, x_samp, y_samp, half=APERTURE_HALF):
    """Aperture-sum extraction along the model trace of one order.

    Returns (spec, y_trace) over the full detector width; columns outside
    the model's x range or off the detector are NaN.
    """
    nrow, ncol = img.shape
    x = np.arange(ncol)
    # linear interpolation of the ~15 model samples is off by up to 4 px
    y_model = CubicSpline(x_samp, y_samp)(x)
    in_x = (x >= x_samp.min()) & (x <= x_samp.max())

    def on_detector(y):
        return in_x & (y >= half + 1) & (y <= nrow - half - 2)

    inside = on_detector(y_model)
    off = trace_offset(img, x[inside], y_model[inside])
    # check the aperture again after the shift, it can push it off the edge
    y_trace = y_model + off
    inside = on_detector(y_trace)
    y_trace[~inside] = np.nan
    xi = x[inside]

    spec = np.full(ncol, np.nan)
    for i in xi:
        yc = int(round(y_trace[i]))
        spec[i] = img[yc - half : yc + half + 1, i].sum()
    return spec, y_trace, off


def process_band(
    band: str,
    lfc_file: str,
    hdf_file: str,
    output_dir: str,
    prefix: str,
    fiber: int = 1,
    aperture_half: int = APERTURE_HALF,
    peak_window: int = 6,
    elsf_half: float = ELSF_HALF,
    elsf_xbin: int = ELSF_XBIN,
    elsf_knot: float = ELSF_KNOT,
    extract=extract_order,
    plots: bool = True,
) -> list[dict] | None:
    """Run the sampling analysis for one single-fiber LFC frame.

    extract(img, x_samp, y_samp, half) -> (spec, y_trace, offset) does the
    1D extraction of one order, by default a box sum along the model trace.
    Plots are written to output_dir as {prefix}_*.png unless plots is False.
    Returns the per-cell empirical LSF results, or None if the inputs are
    missing or no lines fit.
    """
    print(f"\n{'=' * 60}\nChannel {band}\n{'=' * 60}")

    if not os.path.exists(lfc_file):
        print(f"  Skipping: LFC file not found ({lfc_file})")
        return None
    if not os.path.exists(hdf_file):
        print(f"  Skipping: HDF file not found ({hdf_file})")
        return None
    if plots:
        os.makedirs(output_dir, exist_ok=True)

    model = load_trace_model(hdf_file, fiber=f"fiber_{fiber}")
    orders = sorted(model, reverse=True)
    print(f"  Loaded trace/wavelength model: {os.path.basename(hdf_file)}")
    print(f"  Orders: m = {orders[0]}..{orders[-1]} ({len(orders)} total)")

    img = fits.getdata(lfc_file).astype(float)

    # --- Extract the single fiber along the model trace ---
    print(f"Extracting fiber {fiber} from {os.path.basename(lfc_file)}...")
    spectra = {}  # m -> (spec, y_trace)
    for m in orders:
        x_samp, y_samp, _wl = model[m]
        spec, y_trace, off = extract(img, x_samp, y_samp, half=aperture_half)
        spectra[m] = (spec, y_trace)
        nvalid = np.isfinite(spec).sum()
        print(f"  m={m}: {nvalid} valid columns, trace offset {off:+.2f} px")

    # --- Peak finding + Gaussian FWHM analysis ---
    print("Fitting LFC peaks with Gaussians...")

    # per_order: list of (m, x_peaks, y_peaks, fwhms, resolving_power)
    per_order: list[tuple] = []
    baselines = {}
    for m in orders:
        spec, y_trace = spectra[m]
        xp, fw, baselines[m] = fit_peaks(spec, window=peak_window)
        yp = (
            np.interp(xp, np.arange(y_trace.size), y_trace) if xp.size else np.array([])
        )
        x_samp, _y_samp, wl_samp = model[m]
        rp = (
            compute_resolving_power(xp, fw, (x_samp, wl_samp))
            if xp.size
            else np.array([])
        )
        per_order.append((m, xp, yp, fw, rp))

    all_fw = np.concatenate([fw for _m, _xp, _yp, fw, _rp in per_order])
    print(f"\nFWHM summary (pixels) for {band}:")
    print(f"  {'npeaks':>6}  {'median':>7}  {'mean':>7}  {'std':>6}")
    if all_fw.size == 0:
        print(f"  {0:>6}      -       -       -")
        return None
    print(
        f"  {all_fw.size:>6}  {np.median(all_fw):>7.3f}  "
        f"{np.mean(all_fw):>7.3f}  {np.std(all_fw):>6.3f}"
    )

    if plots:
        # --- Diagnostic plots: spectrum sample + FWHM vs x ---
        fig, (ax_spec, ax_fwhm) = plt.subplots(2, 1, figsize=(11, 8))

        ref_order = orders[len(orders) // 2]
        ax_spec.plot(spectra[ref_order][0], lw=0.7, label=f"fiber {fiber}")
        ax_spec.set_title(f"{band} LFC spectrum, order m={ref_order}")
        ax_spec.set_xlabel("x [pixel]")
        ax_spec.set_ylabel("flux")
        ax_spec.legend(loc="upper right")

        for _m, xp, _yp, fw, _rp in per_order:
            ax_fwhm.scatter(xp, fw, s=6, alpha=0.5, color="tab:blue")
        ax_fwhm.axhline(2.0, ls=":", color="k", lw=0.8, label="Nyquist (2 px)")
        ax_fwhm.set_xlabel("x [pixel]")
        ax_fwhm.set_ylabel("FWHM [pixel]")
        ax_fwhm.set_title(f"{band} LFC line FWHM across the detector (all orders)")
        ax_fwhm.legend(loc="upper right")
        ax_fwhm.set_ylim(0, None)

        fig.tight_layout()
        out_png = os.path.join(output_dir, f"{prefix}_sampl_fwhm.png")
        fig.savefig(out_png, dpi=120)
        print(f"\nSaved diagnostic plot: {out_png}")
        if not PLOT:
            plt.close(fig)

    # --- 2D interpolated FWHM map across the full detector ---
    all_x = np.concatenate([xp for _m, xp, _yp, _fw, _rp in per_order if xp.size])
    all_y = np.concatenate([yp for _m, _xp, yp, _fw, _rp in per_order if yp.size])
    all_fw = np.concatenate([fw for _m, _xp, _yp, fw, _rp in per_order if fw.size])

    nrow, ncol = img.shape

    # Median-bin raw peak measurements to suppress per-peak fit noise.
    bin_size = 256
    nbx = ncol // bin_size
    nby = nrow // bin_size
    stat, xedges, yedges, _ = binned_statistic_2d(
        all_x,
        all_y,
        all_fw,
        statistic="median",
        bins=[nbx, nby],
        range=[[0, ncol], [0, nrow]],
    )
    xcen = 0.5 * (xedges[:-1] + xedges[1:])
    ycen = 0.5 * (yedges[:-1] + yedges[1:])

    valid = np.isfinite(stat)
    XB, YB = np.meshgrid(xcen, ycen, indexing="ij")
    pts = np.column_stack([XB[valid], YB[valid]])

    plot_detector_map(
        pts,
        stat[valid],
        img.shape,
        (all_x, all_y),
        fwhm_levels,
        "%.1f",
        "viridis",
        "FWHM [pixel]",
        f"{band} LFC FWHM map (linear interpolation)",
        plots and os.path.join(output_dir, f"{prefix}_sampl_fwhm_map.png"),
    )

    # --- 2D interpolated resolving power map ---
    all_rp = np.concatenate([rp for _m, _xp, _yp, _fw, rp in per_order if rp.size])
    good = np.isfinite(all_rp) & np.isfinite(all_x) & np.isfinite(all_y)
    if good.sum() > 100:
        stat_r, _, _, _ = binned_statistic_2d(
            all_x[good],
            all_y[good],
            all_rp[good],
            statistic="median",
            bins=[nbx, nby],
            range=[[0, ncol], [0, nrow]],
        )
        valid_r = np.isfinite(stat_r)
        plot_detector_map(
            np.column_stack([XB[valid_r], YB[valid_r]]),
            stat_r[valid_r],
            img.shape,
            (all_x[good], all_y[good]),
            r_levels,
            "%.0f",
            "inferno",
            "Resolving power R",
            f"{band} resolving power map (from LFC FWHM)",
            plots and os.path.join(output_dir, f"{prefix}_sampl_R_map.png"),
        )

        med_r = np.nanmedian(all_rp[good])
        mean_r = np.nanmean(all_rp[good])
        print(f"  Resolving power: median R = {med_r:.0f}, mean R = {mean_r:.0f}")

    # --- Empirical oversampled LSF per (order, x-bin) cell ---
    print(f"\nEmpirical LSF from stacked lines ({elsf_xbin} px cells)...")
    cells = []
    n_skipped = 0
    for m, xp, _yp, fw, rp in per_order:
        if not xp.size:
            continue
        spec, y_trace = spectra[m]
        x_samp, _y_samp, wl_samp = model[m]
        cut = line_cutouts(spec, xp, baselines[m], half=elsf_half)
        for x0 in range(0, ncol, elsf_xbin):
            sel = [c for c in cut if x0 <= c[2] < x0 + elsf_xbin]
            if len(sel) < ELSF_MIN_LINES:
                continue
            res = empirical_lsf(sel, half=elsf_half, knot=elsf_knot)
            if res is None:
                n_skipped += 1
                continue
            spl, dx, val = res
            fwhm = profile_fwhm(spl)
            if not np.isfinite(fwhm):
                n_skipped += 1
                continue
            try:
                g, _ = curve_fit(gaussian, dx, val, p0=[val.max(), 0.0, 1.0, 0.0])
                fwhm_gstack = abs(g[2]) * SIGMA_TO_FWHM
            except RuntimeError:
                g, fwhm_gstack = None, np.nan
            mus = np.array([c[2] for c in sel])
            in_cell = (xp >= x0) & (xp < x0 + elsf_xbin)
            xc = float(np.mean(mus))
            cells.append(
                {
                    "m": m,
                    "x0": x0,
                    "x": xc,
                    "y": float(np.interp(xc, np.arange(y_trace.size), y_trace)),
                    "n": len(sel),
                    "fwhm": fwhm,
                    "fwhm_gstack": fwhm_gstack,
                    "fwhm_gline": float(np.median(fw[in_cell])),
                    "R_gline": float(np.nanmedian(rp[in_cell])),
                    "R": compute_resolving_power(
                        np.array([xc]), np.array([fwhm]), (x_samp, wl_samp)
                    )[0],
                    "spl": spl,
                    "gauss": g,
                    "dx": dx,
                    "val": val,
                }
            )

    if len(cells) < 4:
        print("  Too few cells with enough lines; skipping eLSF output.")
        return cells

    c_x = np.array([c["x"] for c in cells])
    c_y = np.array([c["y"] for c in cells])
    c_fw = np.array([c["fwhm"] for c in cells])
    c_gs = np.array([c["fwhm_gstack"] for c in cells])
    c_gl = np.array([c["fwhm_gline"] for c in cells])
    c_r = np.array([c["R"] for c in cells])
    ok = np.isfinite(c_fw)

    print(
        f"  {ok.sum()} cells, {np.median([c['n'] for c in cells]):.0f} lines/cell, "
        f"{n_skipped} skipped (sub-pixel phase gap > {ELSF_MAX_GAP * elsf_knot:.2f} px)"
    )
    print(f"  {'FWHM [px]':<28} {'median':>7} {'p5':>6} {'p95':>6}")
    for label, arr in [
        ("eLSF half-max", c_fw),
        ("Gaussian fit to stack", c_gs),
        ("per-line Gaussian (median)", c_gl),
    ]:
        p5, p50, p95 = np.nanpercentile(arr, [5, 50, 95])
        print(f"  {label:<28} {p50:>7.3f} {p5:>6.3f} {p95:>6.3f}")
    ratio = c_fw / c_gl
    print(
        f"  eLSF / per-line Gaussian: median {np.nanmedian(ratio):.3f}, "
        f"range {np.nanmin(ratio):.3f}..{np.nanmax(ratio):.3f}"
    )
    print(f"  eLSF resolving power: median R = {np.nanmedian(c_r):.0f}")

    plot_detector_map(
        np.column_stack([c_x[ok], c_y[ok]]),
        c_fw[ok],
        img.shape,
        (c_x[ok], c_y[ok]),
        fwhm_levels,
        "%.1f",
        "viridis",
        "FWHM [pixel]",
        f"{band} empirical LSF FWHM map",
        plots and os.path.join(output_dir, f"{prefix}_elsf_fwhm_map.png"),
    )
    ok_r = ok & np.isfinite(c_r)
    plot_detector_map(
        np.column_stack([c_x[ok_r], c_y[ok_r]]),
        c_r[ok_r],
        img.shape,
        (c_x[ok_r], c_y[ok_r]),
        r_levels,
        "%.0f",
        "inferno",
        "Resolving power R",
        f"{band} resolving power map (from empirical LSF)",
        plots and os.path.join(output_dir, f"{prefix}_elsf_R_map.png"),
    )

    if plots:
        # eLSF vs Gaussian widths
        fig4, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))
        lim = [
            np.nanmin([c_fw, c_gl, c_gs]) - 0.05,
            np.nanmax([c_fw, c_gl, c_gs]) + 0.05,
        ]
        ax_a.scatter(c_gl, c_fw, s=12, label="vs per-line Gaussian (median)")
        ax_a.scatter(c_gs, c_fw, s=12, marker="x", label="vs Gaussian fit to stack")
        ax_a.plot(lim, lim, "k:", lw=0.8)
        ax_a.set_xlim(lim)
        ax_a.set_ylim(lim)
        ax_a.set_xlabel("Gaussian FWHM [pixel]")
        ax_a.set_ylabel("eLSF half-max FWHM [pixel]")
        ax_a.legend(loc="upper left")
        sc = ax_b.scatter(c_x, ratio, c=[c["m"] for c in cells], s=12, cmap="turbo")
        ax_b.axhline(1.0, ls=":", color="k", lw=0.8)
        ax_b.set_xlabel("x [pixel]")
        ax_b.set_ylabel("eLSF FWHM / per-line Gaussian FWHM")
        fig4.colorbar(sc, ax=ax_b).set_label("order m")
        fig4.suptitle(f"{band}: empirical LSF vs Gaussian FWHM per cell")
        fig4.tight_layout()
        out_cmp = os.path.join(output_dir, f"{prefix}_elsf_vs_gauss.png")
        fig4.savefig(out_cmp, dpi=120)
        if not PLOT:
            plt.close(fig4)
        print(f"Saved comparison: {out_cmp}")

        # Stacked profiles for a 3x3 selection of cells (top/mid/bottom order x left/mid/right)
        c_orders = sorted({c["m"] for c in cells}, reverse=True)
        pick_m = [c_orders[0], c_orders[len(c_orders) // 2], c_orders[-1]]
        fig5 = plt.figure(figsize=(14, 11))
        subfigs = fig5.subfigures(3, 3)
        xf = np.linspace(-elsf_half + ELSF_MARGIN, elsf_half - ELSF_MARGIN, 2001)
        for row, m in enumerate(pick_m):
            oc = sorted((c for c in cells if c["m"] == m), key=lambda c: c["x"])
            for col, c in enumerate([oc[0], oc[len(oc) // 2], oc[-1]]):
                ax_p, ax_res = subfigs[row, col].subplots(
                    2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
                )
                ax_p.plot(c["dx"], c["val"], ".", ms=2, alpha=0.4, color="0.5")
                ax_p.plot(xf, c["spl"](xf), color="tab:blue", label="eLSF")
                if c["gauss"] is not None:
                    ax_p.plot(
                        xf,
                        gaussian(xf, *c["gauss"]),
                        "--",
                        color="tab:red",
                        label="Gaussian",
                    )
                    ax_res.plot(
                        c["dx"],
                        c["val"] - gaussian(c["dx"], *c["gauss"]),
                        ".",
                        ms=2,
                        alpha=0.4,
                        color="0.5",
                    )
                    ax_res.plot(
                        xf, c["spl"](xf) - gaussian(xf, *c["gauss"]), color="tab:blue"
                    )
                ax_res.axhline(0, color="k", lw=0.5)
                ax_p.set_title(
                    f"m={m}, x={c['x']:.0f}, n={c['n']}\n"
                    f"eLSF {c['fwhm']:.2f}, G(stack) {c['fwhm_gstack']:.2f}, "
                    f"G(line) {c['fwhm_gline']:.2f} px",
                    fontsize=9,
                )
                ax_p.set_xlim(-2.2 * c["fwhm"], 2.2 * c["fwhm"])
                ax_res.set_xlabel("x - x0 [pixel]")
                if row == 0 and col == 0:
                    ax_p.legend(fontsize=8)
        fig5.suptitle(
            f"{band} stacked LFC line profiles (residuals: data - Gaussian)", y=1.03
        )
        out_prof = os.path.join(output_dir, f"{prefix}_elsf_profiles.png")
        fig5.savefig(out_prof, dpi=110, bbox_inches="tight")
        if not PLOT:
            plt.close(fig5)
        print(f"Saved profiles: {out_prof}")

    return cells


# --- Main loop over all iq15 bands ---
if __name__ == "__main__":
    processed = []
    for ch in CHANNELS:
        band = f"{ch}_iq15"
        if (
            process_band(
                band,
                lfc_file=os.path.join(LFC_DIR, f"{band}_LFC_single_1s.fits"),
                hdf_file=os.path.join(HDF_DIR, f"ANDES_{ch}_master_rotfix_iq15.hdf"),
                output_dir=os.path.join(DATA_DIR, "ANDES", "reduced", band),
                prefix=f"andes_{ch.lower()}_iq15",
            )
            is not None
        ):
            processed.append(band)

    print(f"\n{'=' * 60}")
    print(f"Done. Processed channels: {processed}")

    if PLOT:
        plt.show()
