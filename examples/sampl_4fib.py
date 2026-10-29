"""
ANDES sampling / resolving-power maps from per-line Gaussian fits, for all
arms and optical-model variants, with the LFC in 4 fibers spread along the
slit (1, 23, 44, 66) instead of fiber 16 only. Every value is still a
single-fiber measurement, the extra fibers just put 4x as many lines on the
detector. The iq15 YJH models have one fiber only.

Frames from ~/ANDES/E2E/src/scripts/lfc_sampl_4fib.py. Unlike sampl_iq15.py /
sampl_vis.py, lines are kept by the photon-noise error of the fitted FWHM
instead of a prominence cut at 10% of the order maximum, which threw away
everything down the blaze wings.

Per-line results are cached as {tag}_lines.npz (--refit redoes the fits) and
written to {tag}_lines.csv.

Usage: sampl_4fib.py [--refit] [tag ...]   (default: all models)
"""

import argparse
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.ticker import MaxNLocator
from sampl_iq15 import (
    DATA_DIR,
    HDF_DIR,
    PLOT,
    SIGMA_TO_FWHM,
    compute_resolving_power,
    extract_order,
    gaussian,
    load_trace_model,
    plot_detector_map,
)
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import binned_statistic_2d

LFC_DIR = os.path.expanduser("~/ANDES/E2E/sampl_4fib")
OUT_DIR = os.path.join(DATA_DIR, "ANDES", "reduced", "sampl_4fib")

FIBERS_66 = [1, 23, 44, 66]
# lines are ~4-6 px FWHM in UBVRIZ, about twice the YJH iq15 sampling
VIS = {"aperture_half": 8, "peak_window": 10}
IQ15 = {"aperture_half": 5, "peak_window": 6}

# tag -> (HDF model, fibers, extraction params)
MODELS = {
    "U": ("ANDES_U_v88", FIBERS_66, VIS),
    "B": ("ANDES_B_v88", FIBERS_66, VIS),
    "V": ("ANDES_V_v88", FIBERS_66, VIS),
    "R0": ("ANDES_123_R3", FIBERS_66, VIS),
    "R1": ("ANDES_full_F18A33_win_jmr_MC_T0019_Rband_p0", FIBERS_66, VIS),
    "R2": ("Andes_full_F18A33_win_jmr_MC_T0108_Rband_P0_cfg1", FIBERS_66, VIS),
    "IZ0": ("ANDES_123_IZ3", FIBERS_66, VIS),
    "IZ1": ("Andes_F18A33_VM246aa_win_jmr9_MC_T0045_IZband_P0_cf1", FIBERS_66, VIS),
    "IZ2": ("Andes_full_F18A33_win_jmr_MC_T0028_IZband_P0", FIBERS_66, VIS),
    "Y_iq15": ("ANDES_Y_master_rotfix_iq15", [1], IQ15),
    "J_iq15": ("ANDES_J_master_rotfix_iq15", [1], IQ15),
    "H_iq15": ("ANDES_H_master_rotfix_iq15", [1], IQ15),
}

GROUPS = {
    "R_models": ["R0", "R1", "R2"],
    "IZ_models": ["IZ0", "IZ1", "IZ2"],
    "UBVRIZ": ["U", "B", "V", "R0", "IZ0"],
    "YJH_iq15": ["Y_iq15", "J_iq15", "H_iq15"],
}

MIN_PROMINENCE = 20.0  # e-, just above the photon noise of the inter-line floor
MAX_FWHM_RELERR = 0.03
BIN_SIZE = 256


def lfc_file(tag):
    return os.path.join(LFC_DIR, f"{tag}_LFC_{len(MODELS[tag][1])}fib_1s.fits")


def photon_sigma_err(xs, popt):
    """Photon-noise error of the Gaussian sigma at the fitted parameters."""
    amp, mu, sig, base = popt
    g = np.exp(-0.5 * ((xs - mu) / sig) ** 2)
    jac = np.column_stack(
        [
            g,
            amp * g * (xs - mu) / sig**2,
            amp * g * (xs - mu) ** 2 / sig**3,
            np.ones_like(xs),
        ]
    )
    var = np.maximum(gaussian(xs, *popt), 1.0)
    try:
        cov = np.linalg.inv(jac.T @ (jac / var[:, None]))
    except np.linalg.LinAlgError:
        return np.nan
    return np.sqrt(cov[2, 2])


def fit_lines(spec, window, max_sigma=5.0):
    """Gaussian fit of every LFC line.

    The fit is unweighted as in sampl_iq15.fit_peaks: the profiles are not
    Gaussian, and photon-noise weighting gives ~7% smaller FWHMs (H_iq15), so
    weights would break comparability with the earlier results. Photon noise
    only decides which lines are kept.

    Returns (x, fwhm, fwhm_err, amp) of the lines whose relative FWHM error
    is below MAX_FWHM_RELERR.
    """
    flux = np.asarray(spec, dtype=float)
    good = np.isfinite(flux)
    out = np.empty((0, 4))
    if good.sum() < 50:
        return out.T
    idx = np.arange(flux.size)
    peaks, _ = find_peaks(
        np.where(good, flux, 0.0), prominence=MIN_PROMINENCE, distance=4
    )
    lo, hi = idx[good].min() + window, idx[good].max() - window
    peaks = peaks[(peaks >= lo) & (peaks <= hi)]

    rows = []
    for p in peaks:
        xs = idx[p - window : p + window + 1].astype(float)
        ys = flux[p - window : p + window + 1]
        if not np.all(np.isfinite(ys)):
            continue
        b0 = ys.min()
        p0 = [ys.max() - b0, float(p), window / 4, b0]
        try:
            popt, _ = curve_fit(gaussian, xs, ys, p0=p0, maxfev=2000)
        except (RuntimeError, ValueError):
            continue
        sigma = abs(popt[2])
        err = photon_sigma_err(xs, popt)
        if not (0 < sigma <= max_sigma) or not np.isfinite(err):
            continue
        if err / sigma > MAX_FWHM_RELERR or abs(popt[1] - p) > 2:
            continue
        rows.append((popt[1], sigma * SIGMA_TO_FWHM, err * SIGMA_TO_FWHM, popt[0]))
    return (np.array(rows) if rows else out).T


def measure(tag):
    """Fit all lines of all fibers of one frame; returns a dict of arrays."""
    hdf, fibers, par = MODELS[tag]
    img = fits.getdata(lfc_file(tag)).astype(float)
    cols = {k: [] for k in ("fiber", "m", "x", "y", "fwhm", "fwhm_err", "amp", "R")}
    for fib in fibers:
        model = load_trace_model(
            os.path.join(HDF_DIR, f"{hdf}.hdf"), fiber=f"fiber_{fib}"
        )
        for m in sorted(model):
            x_samp, y_samp, wl_samp = model[m]
            spec, y_trace, _off = extract_order(
                img, x_samp, y_samp, half=par["aperture_half"]
            )
            x, fw, fe, amp = fit_lines(spec, par["peak_window"])
            if not x.size:
                continue
            cols["fiber"].append(np.full(x.size, fib))
            cols["m"].append(np.full(x.size, m))
            cols["x"].append(x)
            cols["y"].append(np.interp(x, np.arange(y_trace.size), y_trace))
            cols["fwhm"].append(fw)
            cols["fwhm_err"].append(fe)
            cols["amp"].append(amp)
            cols["R"].append(compute_resolving_power(x, fw, (x_samp, wl_samp)))
    res = {k: np.concatenate(v) for k, v in cols.items()}
    res["shape"] = np.array(img.shape)
    return res


def load_or_measure(tag, refit):
    cache = os.path.join(OUT_DIR, f"{tag}_lines.npz")
    if not refit and os.path.exists(cache):
        return dict(np.load(cache))
    if not os.path.exists(lfc_file(tag)):
        print(f"{tag}: no frame {lfc_file(tag)}, skipped")
        return None
    res = measure(tag)
    np.savez(cache, **res)
    return res


def binned_map(res, key, bin_size=BIN_SIZE):
    """Median of key in bin_size^2 detector cells; returns (points, values)."""
    nrow, ncol = res["shape"]
    v = res[key]
    ok = np.isfinite(v) & np.isfinite(res["y"])
    stat, xe, ye, _ = binned_statistic_2d(
        res["x"][ok],
        res["y"][ok],
        v[ok],
        statistic="median",
        bins=[ncol // bin_size, nrow // bin_size],
        range=[[0, ncol], [0, nrow]],
    )
    XB, YB = np.meshgrid(
        0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), indexing="ij"
    )
    valid = np.isfinite(stat)
    return np.column_stack([XB[valid], YB[valid]]), stat[valid]


def color_range(vals):
    # the far blaze wings leave a few outlying cells, which would squash the scale
    return tuple(np.nanpercentile(vals, [1, 99]))


def nice_levels(vmin, vmax):
    return MaxNLocator(nbins=10).tick_values(vmin, vmax)


def plot_band(tag, res):
    out = os.path.join(OUT_DIR, tag)
    os.makedirs(out, exist_ok=True)
    prefix = os.path.join(out, f"andes_{tag.lower()}_4fib")
    fibers = MODELS[tag][1]

    fig, ax = plt.subplots(figsize=(11, 5))
    for fib in fibers:
        s = res["fiber"] == fib
        ax.scatter(res["x"][s], res["fwhm"][s], s=3, alpha=0.4, label=f"fiber {fib}")
    ax.axhline(2.0, ls=":", color="k", lw=0.8, label="Nyquist (2 px)")
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("FWHM [pixel]")
    ax.set_ylim(0, None)
    ax.set_title(f"{tag} LFC line FWHM across the detector (all orders)")
    ax.legend(loc="upper right", markerscale=4)
    fig.tight_layout()
    fig.savefig(f"{prefix}_fwhm.png", dpi=120)
    if not PLOT:
        plt.close(fig)

    scatter = (res["x"], res["y"])
    pts, val = binned_map(res, "fwhm")
    plot_detector_map(
        pts, val, res["shape"], scatter, nice_levels, "%.2f", "viridis",
        "FWHM [pixel]", f"{tag} LFC FWHM map, fibers {fibers}",
        f"{prefix}_fwhm_map.png", vrange=color_range(val),
    )  # fmt: skip
    pts, val = binned_map(res, "R")
    plot_detector_map(
        pts, val, res["shape"], scatter, nice_levels, "%.0f", "inferno",
        "Resolving power R", f"{tag} resolving power map, fibers {fibers}",
        f"{prefix}_R_map.png", vrange=color_range(val),
    )  # fmt: skip


def write_csv(tag, res):
    path = os.path.join(OUT_DIR, f"{tag}_lines.csv")
    cols = ["fiber", "m", "x", "y", "fwhm", "fwhm_err", "R"]
    np.savetxt(
        path,
        np.column_stack([res[c] for c in cols]),
        delimiter=",",
        header=",".join(cols),
        comments="",
        fmt=["%d", "%d", "%.3f", "%.3f", "%.4f", "%.4f", "%.0f"],
    )


def plot_comparison(results, title, out_png):
    """Binned FWHM and R maps of several models on shared colour scales."""
    keys = [
        ("fwhm", "FWHM [pixel]", "viridis", "{:.2f} px", "%.2f"),
        ("R", "R", "inferno", "{:.0f}", "%.0f"),
    ]
    fig, axs = plt.subplots(
        2, len(results), figsize=(5 * len(results), 9.5), layout="constrained"
    )
    for row, (key, label, cmap, fmt, cfmt) in enumerate(keys):
        maps = {t: binned_map(r, key) for t, r in results.items()}
        vmin, vmax = color_range(np.concatenate([v for _, v in maps.values()]))
        levels = nice_levels(vmin, vmax)
        for col, (tag, (pts, val)) in enumerate(maps.items()):
            nrow, ncol = results[tag]["shape"]
            gx, gy = np.meshgrid(np.arange(0, ncol, 32), np.arange(0, nrow, 32))
            grid = griddata(pts, val, (gx, gy), method="linear")
            ax = axs[row, col]
            im = ax.imshow(
                grid, origin="lower", extent=(0, ncol, 0, nrow),
                cmap=cmap, vmin=vmin, vmax=vmax,
            )  # fmt: skip
            cs = ax.contour(
                gx, gy, grid, levels=levels, colors="white", linewidths=0.6, alpha=0.8
            )
            ax.clabel(cs, inline=True, fontsize=7, fmt=cfmt)
            ax.scatter(*pts.T, s=2, color="k", alpha=0.3)
            med = np.nanmedian(results[tag][key])
            ax.set_title(f"{tag}: median " + fmt.format(med))
            ax.set_xlabel("x [pixel]")
            ax.set_ylabel("y [pixel]")
        fig.colorbar(im, ax=axs[row, :], shrink=0.85, label=label)
    fig.suptitle(f"ANDES optical models, {title}, per-line Gaussian fits")
    fig.savefig(out_png, dpi=110)
    print(f"Saved comparison: {out_png}")
    if not PLOT:
        plt.close(fig)


def run(args):
    tag, refit = args
    return tag, load_or_measure(tag, refit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="*", default=list(MODELS))
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    with multiprocessing.Pool(int(os.environ.get("NPROC", 6))) as pool:
        for tag, res in pool.imap_unordered(run, [(t, args.refit) for t in args.tags]):
            if res is not None:
                results[tag] = res
    results = {t: results[t] for t in MODELS if t in results}

    print(f"\n{'=' * 78}\nPer-line Gaussian fits (median, p5-p95)")
    print(
        f"  {'model':<7} {'lines':>6} {'bins':>5}  {'FWHM [px]':>17}"
        f"  {'R':>18}  R per fiber"
    )
    for tag, res in results.items():
        plot_band(tag, res)
        write_csv(tag, res)
        f5, f50, f95 = np.nanpercentile(res["fwhm"], [5, 50, 95])
        r5, r50, r95 = np.nanpercentile(res["R"], [5, 50, 95]) / 1e3
        per_fib = " ".join(
            f"{np.nanmedian(res['R'][res['fiber'] == f]) / 1e3:.1f}k"
            for f in MODELS[tag][1]
        )
        nbins = len(binned_map(res, "fwhm")[1])
        print(
            f"  {tag:<7} {res['x'].size:>6} {nbins:>5}  {f50:5.2f} ({f5:.2f}-{f95:.2f})"
            f"  {r50:5.1f}k ({r5:.0f}-{r95:.0f}k)  {per_fib}"
        )

    for name, group in GROUPS.items():
        sub = {t: results[t] for t in group if t in results}
        if len(sub) >= 2:
            plot_comparison(
                sub,
                name.replace("_", " "),
                os.path.join(OUT_DIR, f"andes_{name}_4fib.png"),
            )

    if PLOT:
        plt.show()
