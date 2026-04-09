"""
ANDES_YJH sampling test: extract individual fibers from a custom-illuminated
LFC frame and measure FWHM of the synthetic LFC peaks across the detector.

Re-uses the tracing from andes_yjh.py (loads the traces saved by that script).
The trace file contains both grouped and per-fiber traces, so we can select
individual fibers 1, 38 and 75 by their fiber_idx.

{channel}_LFC_tcb.fits has only fibers 1, 38 and 75 illuminated with sharp
synthetic LFC lines. Extracting those three fibers gives us a sampling probe
near the bottom, middle and top of each spectral order.

The FWHM (in pixels) of the Gaussian fits is a direct measure of the optical
design's spectral sampling. This script runs the same analysis for all three
ANDES_YJH bands (Y, J, H).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import binned_statistic_2d

from pyreduce.configuration import load_config
from pyreduce.pipeline import Pipeline

# --- Configuration ---
INSTRUMENT_NAME = "ANDES_YJH"
CHANNELS = ["Y", "J", "H"]
# Fibers to extract (1-based, as stored in Trace.fiber_idx).
FIBERS_TO_EXTRACT = [1, 38, 75]

DATA_DIR = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
PLOT = int(os.environ.get("PYREDUCE_PLOT", "0"))

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


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


def process_channel(channel: str) -> bool:
    """Run the sampling analysis for a single ANDES_YJH band. Returns True on success."""
    print(f"\n{'=' * 60}\nChannel {channel}\n{'=' * 60}")

    raw_dir = os.path.join(DATA_DIR, "ANDES", channel)
    output_dir = os.path.join(DATA_DIR, "ANDES", "reduced", channel)
    file_even = os.path.join(raw_dir, f"{channel}_FF_even_1s.fits")
    file_odd = os.path.join(raw_dir, f"{channel}_FF_odd_1s.fits")
    lfc_file = os.path.join(raw_dir, f"{channel}_LFC_tcb.fits")

    if not os.path.exists(lfc_file):
        print(f"  Skipping: LFC file not found ({lfc_file})")
        return False

    # --- Create Pipeline ---
    config = load_config(None, INSTRUMENT_NAME)
    # Narrow aperture so we don't bleed over unlit neighbours (~2.2 px spacing).
    config["science"]["extraction_height"] = 3
    # Disable outlier rejection: the LFC peaks are sharp and bright, the rejector
    # can clip their cores and distort the measured line profiles.
    config["science"]["extraction_reject"] = 0

    pipe = Pipeline(
        instrument=INSTRUMENT_NAME,
        channel=channel,
        output_dir=output_dir,
        target="ANDES_sampling",
        config=config,
        plot=PLOT,
    )

    # --- Load traces from the previous andes_yjh.py run ---
    print("Loading traces from previous run...")
    try:
        trace_objects = pipe._run_step("trace", None, load_only=True)
    except FileNotFoundError:
        print(f"  Skipping: trace file not found for channel {channel}")
        print(f"  Run examples/andes_yjh.py with channel={channel} first.")
        return False
    pipe._data["trace"] = trace_objects

    ind = [t for t in trace_objects if t.group is None and t.fiber_idx is not None]
    orders = sorted({t.m for t in ind}, reverse=True)
    print(f"  Loaded {len(trace_objects)} traces ({len(ind)} individual fibers)")
    print(f"  Orders: m = {orders[0]}..{orders[-1]} ({len(orders)} total)")
    print(
        f"  Used source flats: {os.path.basename(file_even)}, {os.path.basename(file_odd)}"
    )

    # --- Select only the requested fibers for the science step ---
    pipe.instrument.config.fibers.use["science"] = list(FIBERS_TO_EXTRACT)

    # --- Extract the LFC frame ---
    print(f"Extracting fibers {FIBERS_TO_EXTRACT} from {os.path.basename(lfc_file)}...")
    result = pipe.extract([lfc_file]).run()
    _heads, all_spectra = result["science"]
    spectra = all_spectra[0]
    print(f"  Extracted {len(spectra)} spectra")

    # --- Peak finding + Gaussian FWHM analysis ---
    print("Fitting LFC peaks with Gaussians...")

    trace_lookup = {
        (t.m, t.fiber_idx): t
        for t in trace_objects
        if t.group is None and t.fiber_idx is not None
    }

    per_fiber: dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = {
        f: [] for f in FIBERS_TO_EXTRACT
    }
    for s in spectra:
        if s.fiber_idx not in per_fiber:
            continue
        xp, fw = fit_peaks(s.spec)
        t = trace_lookup.get((s.m, s.fiber_idx))
        yp = t.y_at_x(xp) if t is not None and xp.size else np.array([])
        per_fiber[s.fiber_idx].append((s.m, xp, yp, fw))

    # Per-fiber summary
    print(f"\nFWHM summary (pixels) for {channel}:")
    print(f"  {'fiber':>5}  {'npeaks':>6}  {'median':>7}  {'mean':>7}  {'std':>6}")
    for fiber in FIBERS_TO_EXTRACT:
        all_fw_f = (
            np.concatenate([fw for _, _, _, fw in per_fiber[fiber]])
            if per_fiber[fiber]
            else np.array([])
        )
        if all_fw_f.size == 0:
            print(f"  {fiber:>5}  {0:>6}      -       -       -")
            continue
        print(
            f"  {fiber:>5}  {all_fw_f.size:>6}  {np.median(all_fw_f):>7.3f}  "
            f"{np.mean(all_fw_f):>7.3f}  {np.std(all_fw_f):>6.3f}"
        )

    # --- Diagnostic plots: spectrum sample + FWHM vs x ---
    fig, (ax_spec, ax_fwhm) = plt.subplots(2, 1, figsize=(11, 8))

    ref_order = orders[len(orders) // 2]
    for fiber in FIBERS_TO_EXTRACT:
        match = [s for s in spectra if s.fiber_idx == fiber and s.m == ref_order]
        if not match:
            continue
        ax_spec.plot(match[0].spec, lw=0.7, label=f"fiber {fiber}")
    ax_spec.set_title(f"{channel}-band LFC spectrum, order m={ref_order}")
    ax_spec.set_xlabel("x [pixel]")
    ax_spec.set_ylabel("flux")
    ax_spec.legend(loc="upper right")

    colors = {1: "tab:blue", 38: "tab:orange", 75: "tab:green"}
    for fiber in FIBERS_TO_EXTRACT:
        all_x_f, all_fw_f = [], []
        for _m, xp, _yp, fw in per_fiber[fiber]:
            all_x_f.append(xp)
            all_fw_f.append(fw)
        if not all_x_f:
            continue
        all_x_f = np.concatenate(all_x_f)
        all_fw_f = np.concatenate(all_fw_f)
        ax_fwhm.scatter(
            all_x_f,
            all_fw_f,
            s=6,
            alpha=0.5,
            color=colors.get(fiber, "k"),
            label=f"fiber {fiber}",
        )
    ax_fwhm.axhline(2.0, ls=":", color="k", lw=0.8, label="Nyquist (2 px)")
    ax_fwhm.set_xlabel("x [pixel]")
    ax_fwhm.set_ylabel("FWHM [pixel]")
    ax_fwhm.set_title(f"{channel}-band LFC line FWHM across the detector (all orders)")
    ax_fwhm.legend(loc="upper right")
    ax_fwhm.set_ylim(0, None)

    fig.tight_layout()
    out_png = os.path.join(output_dir, f"andes_{channel.lower()}_sampl_fwhm.png")
    fig.savefig(out_png, dpi=120)
    print(f"\nSaved diagnostic plot: {out_png}")

    # --- 2D interpolated FWHM map across the full detector ---
    all_x = np.concatenate(
        [xp for lst in per_fiber.values() for _m, xp, _yp, _fw in lst if xp.size]
    )
    all_y = np.concatenate(
        [yp for lst in per_fiber.values() for _m, _xp, yp, _fw in lst if yp.size]
    )
    all_fw = np.concatenate(
        [fw for lst in per_fiber.values() for _m, _xp, _yp, fw in lst if fw.size]
    )

    with fits.open(lfc_file) as hdu:
        nrow, ncol = hdu[0].data.shape

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
    ax_map.set_title(f"{channel}-band LFC FWHM map (linear interpolation)")
    ax_map.set_xlim(0, ncol)
    ax_map.set_ylim(0, nrow)

    fig2.tight_layout()
    out_map_png = os.path.join(
        output_dir, f"andes_{channel.lower()}_sampl_fwhm_map.png"
    )
    fig2.savefig(out_map_png, dpi=120)
    print(f"Saved FWHM map: {out_map_png}")

    return True


# --- Main loop over all ANDES_YJH bands ---
processed = []
for ch in CHANNELS:
    if process_channel(ch):
        processed.append(ch)

print(f"\n{'=' * 60}")
print(f"Done. Processed channels: {processed}")

if PLOT:
    plt.show()
