"""
ANDES iq15 sampling test: measure FWHM of synthetic LFC peaks across the
detector for the new single-fiber "master_rotfix_iq15" optical models
(Y_iq15, J_iq15, H_iq15 bands of the E2E simulator).

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
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
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
    x_samp, wl_samp = wl_model
    wl_at_peak = np.interp(x_peaks, x_samp, wl_samp)
    # dλ/dx via finite differences of the model samples
    dwl_dx = np.gradient(wl_samp, x_samp)
    dwl_dx_at_peak = np.interp(x_peaks, x_samp, dwl_dx)
    delta_wl = fwhm_px * np.abs(dwl_dx_at_peak)
    R = wl_at_peak / delta_wl
    return R


def gaussian(x, amp, mu, sigma, baseline):
    return baseline + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_peaks(spec, prominence_frac=0.1, window=6):
    """Find LFC peaks and fit each with a Gaussian. Returns (x_peaks, fwhms)."""
    flux = np.asarray(spec, dtype=float)
    mask = np.isfinite(flux)
    if mask.sum() < 50:
        return np.array([]), np.array([])

    idx = np.arange(flux.size)
    valid = idx[mask]
    fmax = np.nanmax(flux[mask])
    if not np.isfinite(fmax) or fmax <= 0:
        return np.array([]), np.array([])

    peaks, _ = find_peaks(
        np.where(mask, flux, 0.0),
        prominence=prominence_frac * fmax,
        distance=4,
    )
    lo, hi = valid.min() + window, valid.max() - window
    peaks = peaks[(peaks >= lo) & (peaks <= hi)]

    x_out, fwhm_out = [], []
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
        p0 = [amp0, float(p), 1.5, baseline0]
        try:
            popt, _ = curve_fit(gaussian, xs, ys, p0=p0, maxfev=2000)
        except (RuntimeError, ValueError):
            continue
        sigma = abs(popt[2])
        if not np.isfinite(sigma) or sigma <= 0 or sigma > 5:
            continue
        x_out.append(popt[1])
        fwhm_out.append(sigma * SIGMA_TO_FWHM)

    return np.array(x_out), np.array(fwhm_out)


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
    y_model = np.interp(x, x_samp, y_samp, left=np.nan, right=np.nan)
    inside = (
        np.isfinite(y_model)
        & (x >= x_samp.min())
        & (x <= x_samp.max())
        & (y_model >= half + 1)
        & (y_model <= nrow - half - 2)
    )

    xi = x[inside]
    off = trace_offset(img, xi, y_model[inside])
    y_trace = y_model + off

    spec = np.full(ncol, np.nan)
    for i in xi:
        yc = int(round(y_trace[i]))
        spec[i] = img[yc - half : yc + half + 1, i].sum()
    return spec, y_trace, off


def process_channel(channel: str) -> bool:
    """Run the sampling analysis for a single iq15 band. Returns True on success."""
    band = f"{channel}_iq15"
    print(f"\n{'=' * 60}\nChannel {band}\n{'=' * 60}")

    output_dir = os.path.join(DATA_DIR, "ANDES", "reduced", band)
    lfc_file = os.path.join(LFC_DIR, f"{band}_LFC_single_1s.fits")
    hdf_file = os.path.join(HDF_DIR, f"ANDES_{channel}_master_rotfix_iq15.hdf")

    if not os.path.exists(lfc_file):
        print(f"  Skipping: LFC file not found ({lfc_file})")
        return False
    if not os.path.exists(hdf_file):
        print(f"  Skipping: HDF file not found ({hdf_file})")
        return False
    os.makedirs(output_dir, exist_ok=True)

    model = load_trace_model(hdf_file)
    orders = sorted(model, reverse=True)
    print(f"  Loaded trace/wavelength model: {os.path.basename(hdf_file)}")
    print(f"  Orders: m = {orders[0]}..{orders[-1]} ({len(orders)} total)")

    img = fits.getdata(lfc_file).astype(float)

    # --- Extract the single fiber along the model trace ---
    print(f"Extracting fiber 1 from {os.path.basename(lfc_file)}...")
    spectra = {}  # m -> (spec, y_trace)
    for m in orders:
        x_samp, y_samp, _wl = model[m]
        spec, y_trace, off = extract_order(img, x_samp, y_samp)
        spectra[m] = (spec, y_trace)
        nvalid = np.isfinite(spec).sum()
        print(f"  m={m}: {nvalid} valid columns, trace offset {off:+.2f} px")

    # --- Peak finding + Gaussian FWHM analysis ---
    print("Fitting LFC peaks with Gaussians...")

    # per_order: list of (m, x_peaks, y_peaks, fwhms, resolving_power)
    per_order: list[tuple] = []
    for m in orders:
        spec, y_trace = spectra[m]
        xp, fw = fit_peaks(spec)
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
        return False
    print(
        f"  {all_fw.size:>6}  {np.median(all_fw):>7.3f}  "
        f"{np.mean(all_fw):>7.3f}  {np.std(all_fw):>6.3f}"
    )

    # --- Diagnostic plots: spectrum sample + FWHM vs x ---
    fig, (ax_spec, ax_fwhm) = plt.subplots(2, 1, figsize=(11, 8))

    ref_order = orders[len(orders) // 2]
    ax_spec.plot(spectra[ref_order][0], lw=0.7, label="fiber 1")
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
    out_png = os.path.join(output_dir, f"andes_{channel.lower()}_iq15_sampl_fwhm.png")
    fig.savefig(out_png, dpi=120)
    print(f"\nSaved diagnostic plot: {out_png}")

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
    vals = stat[valid]

    grid_step = 32
    gx = np.arange(0, ncol, grid_step)
    gy = np.arange(0, nrow, grid_step)
    GX, GY = np.meshgrid(gx, gy)

    fwhm_map = griddata(pts, vals, (GX, GY), method="linear")

    fig2, ax_map = plt.subplots(figsize=(9, 8))
    vmin, vmax = np.nanpercentile(fwhm_map, [2, 98])
    im = ax_map.imshow(
        fwhm_map,
        origin="lower",
        extent=(0, ncol, 0, nrow),
        aspect="equal",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    cs = ax_map.contour(
        GX,
        GY,
        fwhm_map,
        levels=np.arange(1.6, 3.01, 0.1),
        colors="white",
        linewidths=0.7,
        alpha=0.8,
    )
    ax_map.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
    ax_map.scatter(all_x, all_y, s=1, color="k", alpha=0.15)

    cbar = fig2.colorbar(im, ax=ax_map, shrink=0.85)
    cbar.set_label("FWHM [pixel]")
    ax_map.set_xlabel("x [pixel]  (dispersion)")
    ax_map.set_ylabel("y [pixel]  (cross-dispersion)")
    ax_map.set_title(f"{band} LFC FWHM map (linear interpolation)")
    ax_map.set_xlim(0, ncol)
    ax_map.set_ylim(0, nrow)

    fig2.tight_layout()
    out_map_png = os.path.join(
        output_dir, f"andes_{channel.lower()}_iq15_sampl_fwhm_map.png"
    )
    fig2.savefig(out_map_png, dpi=120)
    print(f"Saved FWHM map: {out_map_png}")

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
        pts_r = np.column_stack([XB[valid_r], YB[valid_r]])
        vals_r = stat_r[valid_r]
        r_map = griddata(pts_r, vals_r, (GX, GY), method="linear")

        fig3, ax_r = plt.subplots(figsize=(9, 8))
        rvmin, rvmax = np.nanpercentile(r_map, [2, 98])
        im_r = ax_r.imshow(
            r_map,
            origin="lower",
            extent=(0, ncol, 0, nrow),
            aspect="equal",
            cmap="inferno",
            vmin=rvmin,
            vmax=rvmax,
            interpolation="bilinear",
        )
        r_levels = np.arange(
            np.floor(rvmin / 5000) * 5000,
            np.ceil(rvmax / 5000) * 5000 + 1,
            5000,
        )
        cs_r = ax_r.contour(
            GX,
            GY,
            r_map,
            levels=r_levels,
            colors="white",
            linewidths=0.7,
            alpha=0.8,
        )
        ax_r.clabel(cs_r, inline=True, fontsize=8, fmt="%.0f")
        ax_r.scatter(all_x[good], all_y[good], s=1, color="k", alpha=0.15)

        cbar_r = fig3.colorbar(im_r, ax=ax_r, shrink=0.85)
        cbar_r.set_label("Resolving power R")
        ax_r.set_xlabel("x [pixel]  (dispersion)")
        ax_r.set_ylabel("y [pixel]  (cross-dispersion)")
        ax_r.set_title(f"{band} resolving power map (from LFC FWHM)")
        ax_r.set_xlim(0, ncol)
        ax_r.set_ylim(0, nrow)

        fig3.tight_layout()
        out_r_png = os.path.join(
            output_dir, f"andes_{channel.lower()}_iq15_sampl_R_map.png"
        )
        fig3.savefig(out_r_png, dpi=120)
        print(f"Saved R map: {out_r_png}")

        # Print R summary
        med_r = np.nanmedian(all_rp[good])
        mean_r = np.nanmean(all_rp[good])
        print(f"  Resolving power: median R = {med_r:.0f}, mean R = {mean_r:.0f}")

    return True


# --- Main loop over all iq15 bands ---
processed = []
for ch in CHANNELS:
    if process_channel(ch):
        processed.append(ch)

print(f"\n{'=' * 60}")
print(f"Done. Processed channels: {processed}")

if PLOT:
    plt.show()
