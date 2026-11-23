"""
Compare the PyReduce SOXS reduction (examples/soxs_example.py) with soxspipe.

Needs both reductions of the public commissioning set: PyReduce output in
$REDUCE_DATA/SOXS/reduced/{VIS,NIR}, and a soxspipe workspace after
`soxspipe reduce all` (path in $SOXSPIPE_WS). Writes PNGs to
$REDUCE_DATA/SOXS/compare:

    tilt_{arm}.png        slit tilt dx/dy: PyReduce (multi-pinhole, sky lines)
                          vs soxspipe (2D map, per-line pinhole fits)
    arc_R_{arm}.png       resolving power of pinhole arc lines
    sky_fwhm_{arm}.png    sky-line widths: soxspipe sky model vs PyReduce
                          box extraction with and without the tilt
    spectra_{arm}.png     optimal extraction of the stare frames, both pipelines

Usage:
    uv run python examples/soxs_compare_soxspipe.py
"""

import os
import warnings
from copy import deepcopy
from glob import glob
from os.path import join

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit

from pyreduce.extract import extract
from pyreduce.instruments.instrument_info import load_instrument
from pyreduce.slit_curve import Curvature
from pyreduce.spectra import Spectra
from pyreduce.trace_model import load_traces
from pyreduce.wavelength_calibration import LineList

warnings.filterwarnings("ignore")

data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
SOXSPIPE_WS = os.environ.get(
    "SOXSPIPE_WS", os.path.expanduser("~/soxs_archive_test/ws")
)
RAW = join(data_dir, "SOXS", "raw")
# soxspipe's sky-line catalogue (air, nm), shipped with the package
SKYLINES = os.environ.get(
    "SOXS_SKYLINES",
    os.path.expanduser(
        "~/soxspipe/soxspipe/resources/static_calibrations/soxs/skylines.fits"
    ),
)
OUT = join(data_dir, "SOXS", "compare")
os.makedirs(OUT, exist_ok=True)
inst = load_instrument("SOXS")

FWHM = 2 * np.sqrt(2 * np.log(2))

# frames used for the sky-line measurements (narrowest slit available)
SKY_FRAMES = {
    "VIS": ["SOXS.2025-10-13T04:46:31.367.fits"],  # Feige 110 stare, 1.0" slit, 300 s
    "NIR": ["SOXS.2026-01-11T08:03:47.636.fits"],  # CD-32 5613 stare, 5.0" slit, 30 s
}
# frames summed for the tilt from sky lines: all Feige 110 1.0"/1.5" exposures,
# and the four 150 s LTT 3218 nod frames
TILT_SKY_FRAMES = {
    "VIS": [
        "SOXS.2025-10-13T04:40:51.497.fits",
        "SOXS.2025-10-13T04:46:31.367.fits",
        "SOXS.2025-10-13T05:03:57.361.fits",
        "SOXS.2025-10-13T05:09:33.518.fits",
        "SOXS.2025-10-13T05:15:27.853.fits",
        "SOXS.2025-10-13T05:21:01.993.fits",
    ],
    "NIR": [
        "SOXS.2026-01-11T04:07:44.547.fits",
        "SOXS.2026-01-11T04:13:18.845.fits",
        "SOXS.2026-01-11T06:06:25.639.fits",
        "SOXS.2026-01-11T06:12:01.468.fits",
    ],
}
STARE = {
    "VIS": [
        ("SOXS.2025-10-13T04:35:11.440", "20251013T043511", '5.0"'),
        ("SOXS.2025-10-13T04:40:51.497", "20251013T044051", '1.5"'),
        ("SOXS.2025-10-13T04:46:31.367", "20251013T044631", '1.0"'),
    ],
    "NIR": [("SOXS.2026-01-11T08:03:47.636", "20260111T080347", '5.0"')],
}


# --- coordinate conversions soxspipe (trimmed raw) -> PyReduce ---
def sp_image(arm, img):
    return np.rot90(img, 1) if arm == "VIS" else np.flipud(img)


def sp_xy(arm, xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return (ys, 807 - xs) if arm == "VIS" else (xs, 2039 - ys)


def sp_file(pattern):
    files = sorted(glob(join(SOXSPIPE_WS, pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    return files[0]


def red_dir(arm):
    return join(data_dir, "SOXS", "reduced", arm)


def traces_of(arm):
    traces, _ = load_traces(join(red_dir(arm), f"soxs_{arm.lower()}.traces.fits"))
    return traces


def calibrated(arm, files):
    """Bias-subtracted, flat-fielded sum of raw frames in the PyReduce frame."""
    norm = np.load(join(red_dir(arm), f"soxs_{arm.lower()}.flat_norm.npz"))["norm"]
    bias = 0
    if arm == "VIS":
        bias = fits.getdata(join(red_dir(arm), "soxs_vis.bias.fits"))
    total = 0
    for f in files:
        im, head = inst.load_fits(join(RAW, f), arm)
        total = total + (np.ma.filled(im.astype(float), np.nan) - bias)
    return total / norm, head


def gauss_lin(x, a, mu, sig, c0, c1):
    return a * np.exp(-0.5 * ((x - mu) / sig) ** 2) + c0 + c1 * (x - mu)


def fit_line(x, y, mu0, sig0):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 6:
        return None
    a0 = np.max(y) - np.median(y)
    try:
        p, cov = curve_fit(
            gauss_lin, x, y, p0=[a0, mu0, sig0, np.median(y), 0], maxfev=4000
        )
    except (RuntimeError, ValueError):
        return None
    if not (
        p[0] > 0 and abs(p[1] - mu0) < 2 * sig0 and 0.2 * sig0 < abs(p[2]) < 4 * sig0
    ):
        return None
    perr = np.sqrt(np.diag(cov))
    if not np.isfinite(perr[2]) or perr[2] > 0.15 * abs(p[2]):
        return None
    return p[1], abs(p[2])


# --- 1. slit tilt ---
def soxspipe_map_tilt(arm, traces):
    """dx/dy of the iso-wavelength lines along each trace, from the soxspipe 2D map."""
    img = sp_file(
        f"reduced/*/soxs-spat-solution/*{arm}*{'1X1_1_' if arm == 'VIS' else ''}SSOL*_IMAGE.fits"
    )
    with fits.open(img) as h:
        wave = sp_image(arm, h["WAVELENGTH"].data.astype(float))
    wave[wave <= 0] = np.nan
    gy, gx = np.gradient(wave)
    out = {}
    for t in traces:
        x = np.arange(t.column_range[0] + 20, t.column_range[1] - 20, 20)
        y = np.round(t.y_at_x(x)).astype(int)
        out[t.m] = (x, -gy[y, x] / gx[y, x])
    return out


def soxspipe_pinhole_tilts(arm):
    """dx/dy of each arc line from a straight-line fit through its 9 pinhole images."""
    f = sp_file(
        f"qc/*/soxs-spat-solution/*{arm}*{'1X1_1_' if arm == 'VIS' else ''}SSOL*FITTED_LINES.fits"
    )
    t = Table.read(f)
    t = t[~t["sigma_clipped"].astype(bool)]
    xp, yp = sp_xy(arm, t["observed_x"], t["observed_y"])
    out = {}
    keys = np.array(
        [f"{o}_{w}" for o, w in zip(t["order"], t["wavelength"], strict=True)]
    )
    for k in np.unique(keys):
        s = keys == k
        if s.sum() < 6:
            continue
        slope = np.polyfit(yp[s], xp[s], 1)[0]
        out.setdefault(int(float(k.split("_")[0])), []).append((np.mean(xp[s]), slope))
    return {m: np.array(v) for m, v in out.items()}


def pyreduce_tilts(traces, image, curve_height, window, threshold):
    """Per-line c1 measured by PyReduce on `image`, and a clipped quadratic in x per order."""
    module = Curvature(
        deepcopy(traces),
        curve_height=curve_height,
        extraction_height=1,
        window_width=window,
        peak_threshold=threshold,
        curve_degree=1,
    )
    module._fix_inputs(image)
    peaks, coeffs, _, _, _ = module._determine_curvature_all_lines(image)
    out = {}
    for t, pk, cs in zip(module.traces, peaks, coeffs, strict=True):
        c1 = cs[:, 0] if len(pk) else np.array([])
        keep = np.isfinite(c1) & (c1 != 0)
        fit = None
        for _ in range(5):
            if keep.sum() < 8:
                fit = None
                break
            fit = np.polyfit(pk[keep], c1[keep], 2)
            res = c1 - np.polyval(fit, pk)
            mad = 1.4826 * np.median(np.abs(res[keep]))
            keep = keep & (np.abs(res) < 4 * mad)
        out[t.m] = (pk, c1, keep, fit)
    return out


def plot_tilt(arm):
    traces = traces_of(arm)
    smap = soxspipe_map_tilt(arm, traces)
    spin = soxspipe_pinhole_tilts(arm)
    sky_image, _ = calibrated(arm, TILT_SKY_FRAMES[arm])
    # 1.0-1.5" VIS sky lines are ~4 px wide, the 5" NIR ones ~15 px with flat tops
    if arm == "VIS":
        sky = pyreduce_tilts(traces, np.nan_to_num(sky_image), 36, 6, 5)
    else:
        sky = pyreduce_tilts(traces, np.nan_to_num(sky_image), 28, 15, 5)

    n = len(traces)
    ncols = 4 if arm == "VIS" else 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False
    )
    summary = []
    for ax, t in zip(axes.flat, sorted(traces, key=lambda t: t.m), strict=False):
        m = t.m
        if m in spin:
            ax.plot(
                spin[m][:, 0],
                spin[m][:, 1],
                ".",
                color="0.6",
                ms=3,
                label="soxspipe pinhole sets",
            )
        x, s = smap[m]
        ax.plot(x, s, "-", color="C0", lw=2, label="soxspipe 2D map")
        xs = np.arange(*t.column_range)
        c1 = np.array([t.slit_at_x(xx)[1] for xx in xs])
        ax.plot(
            xs, c1, "-", color="C3", lw=2, label="PyReduce, multi-pinhole (smoothed)"
        )
        pk, cs, keep, fit = sky[m]
        if keep.any():
            ax.plot(
                pk[keep], cs[keep], "x", color="C2", ms=4, label="PyReduce, sky lines"
            )
        if fit is not None:
            ax.plot(xs, np.polyval(fit, xs), "--", color="C2", lw=1.5)
        ax.set_title(f"{arm} order {m}")
        lo, hi = np.nanpercentile(s, [2, 98])
        pad = max(0.05, 0.5 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("dx/dy")
        # differences at the trace columns of the soxspipe map
        c1_at = np.interp(x, xs, c1)
        sky_dev = np.nan
        if keep.sum() >= 8:
            sky_dev = np.nanmedian(cs[keep] - np.interp(pk[keep], x, s))
        summary.append(
            (
                m,
                np.nanmedian(s),
                np.nanmedian(c1_at),
                np.nanmedian(c1_at - s),
                np.nanmax(np.abs(c1_at - s)),
                sky_dev,
                int(keep.sum()),
            )
        )
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"SOXS {arm}: slit tilt dx/dy (PyReduce frame)")
    fig.tight_layout()
    fig.savefig(join(OUT, f"tilt_{arm}.png"), dpi=110)
    plt.close(fig)
    return summary


# --- 2. resolving power of the pinhole arc lines ---
def plot_arc_R(arm):
    traces = traces_of(arm)
    spec = fits.getdata(join(red_dir(arm), f"soxs_{arm.lower()}.wavecal_master.fits"))
    lines = LineList.load(join(red_dir(arm), f"soxs_{arm.lower()}.linelist.npz"))
    lines = lines.data[lines["flag"]]
    pr = []
    for ln in lines:
        i = int(ln["order"])
        t = traces[i]
        x = np.arange(spec.shape[1])
        win = np.abs(x - ln["posm"]) < 6
        res = fit_line(x[win].astype(float), spec[i][win], ln["posm"], 1.0)
        if res is None:
            continue
        mu, sig = res
        wl = t.wlen(np.array([mu - 0.5, mu, mu + 0.5]))
        disp = abs(wl[2] - wl[0])
        pr.append((wl[1] / 10, wl[1] / (FWHM * sig * disp), FWHM * sig, t.m))
    pr = np.array(pr)

    pattern = f"qc/*/soxs-disp-solution/*{arm}*{'1X1_1_' if arm == 'VIS' else ''}DSOL*FITTED_LINES.fits"
    st = Table.read(sp_file(pattern))
    st = st[
        ~st["sigma_clipped"].astype(bool) & np.isfinite(st["R_pin"]) & (st["R_pin"] > 0)
    ]

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].plot(
        st["wavelength"],
        st["R_pin"],
        ".",
        color="C0",
        label=f"soxspipe (median {np.median(st['R_pin']):.0f})",
    )
    ax[0].plot(
        pr[:, 0],
        pr[:, 1],
        ".",
        color="C3",
        label=f"PyReduce (median {np.median(pr[:, 1]):.0f})",
    )
    ax[0].set_xlabel("wavelength [nm]")
    ax[0].set_ylabel("R = lambda / FWHM")
    ax[0].set_title(f"{arm}: single-pinhole arc lines")
    ax[0].legend()
    ax[1].plot(
        st["wavelength"],
        st["fwhm_pin_px"],
        ".",
        color="C0",
        label="soxspipe (2D, along dispersion)",
    )
    ax[1].plot(pr[:, 0], pr[:, 2], ".", color="C3", label="PyReduce (extracted 1D)")
    ax[1].set_xlabel("wavelength [nm]")
    ax[1].set_ylabel("FWHM [px]")
    ax[1].set_ylim(0, 4)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(join(OUT, f"arc_R_{arm}.png"), dpi=110)
    plt.close(fig)
    return np.median(st["R_pin"]), np.median(pr[:, 1]), len(st), len(pr)


# --- 3. sky-line widths with and without the tilt ---
def soxspipe_orders(arm, stamp):
    f = sp_file(f"reduced/*/soxs-stare/{stamp}_*EXTRACTED_ORDERS.fits")
    return Table.read(f)


def line_shape(x, y, xc, halfwin):
    """Centre, FWHM and mean 20-80% edge width of the line nearest xc (any shape)."""
    win = np.abs(x - xc) < halfwin
    x, y = x[win], y[win]
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 8:
        return None
    base = np.percentile(y, 10)
    noise = 1.4826 * np.median(np.abs(np.diff(y[y < np.percentile(y, 50)]))) + 1e-12
    y = y - base
    core = np.abs(x - xc) < halfwin / 2
    i = int(np.nonzero(core)[0][np.argmax(y[core])])
    peak = y[i]
    if peak < 10 * noise or i == 0 or i == len(y) - 1:
        return None

    def crossing(level, side):
        idx = (
            np.nonzero(y[:i] < level)[0]
            if side < 0
            else i + np.nonzero(y[i:] < level)[0]
        )
        if len(idx) == 0:
            return None
        j = idx[-1] if side < 0 else idx[0]
        k = j + 1 if side < 0 else j - 1
        return np.interp(level, [y[j], y[k]], [x[j], x[k]])

    c = {f: (crossing(f * peak, -1), crossing(f * peak, +1)) for f in (0.2, 0.5, 0.8)}
    if any(v is None for pair in c.values() for v in pair):
        return None
    fwhm = c[0.5][1] - c[0.5][0]
    edge = 0.5 * ((c[0.8][0] - c[0.2][0]) + (c[0.2][1] - c[0.8][1]))
    return 0.5 * (c[0.5][0] + c[0.5][1]), abs(fwhm), abs(edge)


# expected sky-line width (px) and the extraction height for the box spectra
SKY_WIDTH = {"VIS": 4, "NIR": 18}
SKY_HEIGHT = {"VIS": 40, "NIR": 30}
C_KMS = 299792.458


def plot_sky(arm):
    traces = traces_of(arm)
    image, _ = calibrated(arm, SKY_FRAMES[arm])
    image = np.ma.masked_invalid(image)
    notilt = [deepcopy(t) for t in traces]
    for t in notilt:
        t.slit = None
        t.slitdelta = None
    kw = {
        "extraction_height": SKY_HEIGHT[arm],
        "extraction_type": "simple",
        "collapse_function": "sum",
    }
    with_tilt = extract(image, deepcopy(traces), **kw)
    without = extract(image, notilt, **kw)

    stamp = [s for f, s, _ in STARE[arm] if f + ".fits" == SKY_FRAMES[arm][0]][0]
    sp = soxspipe_orders(arm, stamp)
    cat = Table.read(SKYLINES)
    cat = cat[cat["ISOLATED"].astype(bool)]
    w = SKY_WIDTH[arm]

    rows = []
    for t, a, b in zip(traces, with_tilt, without, strict=True):
        o = sp[sp["order"] == t.m]
        wl_s = np.asarray(o["wavelengthMean"], float)
        sky_s = np.asarray(o["skyFlux"], float)
        x = np.arange(len(a.spec), dtype=float)
        wl_p = t.wlen(x) / 10
        ok = np.isfinite(wl_p) & np.isfinite(a.spec) & np.isfinite(b.spec)
        x0, x1 = x[ok].min() + 3 * w, x[ok].max() - 3 * w
        srt = np.argsort(wl_p[ok])
        for w0 in np.asarray(cat["WAVELENGTH"], float):
            xc = np.interp(w0, wl_p[ok][srt], x[ok][srt], left=np.nan, right=np.nan)
            if (
                not (x0 < xc < x1)
                or len(wl_s) == 0
                or not (wl_s.min() < w0 < wl_s.max())
            ):
                continue
            disp_p = abs(np.interp(xc + 0.5, x, wl_p) - np.interp(xc - 0.5, x, wl_p))
            ra = line_shape(x, a.spec, xc, 2.5 * w)
            rb = line_shape(x, b.spec, xc, 2.5 * w)
            # soxspipe sky model, measured in nm and converted to PyReduce pixels
            rs = line_shape(wl_s, sky_s, w0, 2.5 * w * disp_p)
            if ra is None or rb is None or rs is None:
                continue
            wl_a = np.interp(ra[0], x, wl_p)
            rows.append(
                (
                    t.m,
                    w0,
                    rs[1] / disp_p,
                    ra[1],
                    rb[1],
                    rs[2] / disp_p,
                    ra[2],
                    rb[2],
                    (wl_a - w0) / w0 * C_KMS,
                    (rs[0] - w0) / w0 * C_KMS,
                    disp_p,
                )
            )
    rows = np.array(rows)

    fig, ax = plt.subplots(1, 3, figsize=(19, 4.6))
    labels = [
        "soxspipe sky model",
        "PyReduce box, tilt-corrected",
        "PyReduce box, no tilt",
    ]
    colors = ["C0", "C3", "0.5"]
    slit = [s for f, _, s in STARE[arm] if f + ".fits" == SKY_FRAMES[arm][0]][0]
    if arm == "VIS":
        for j, (lab, c) in enumerate(zip(labels, colors, strict=True)):
            r = rows[:, 1] / (rows[:, 2 + j] * rows[:, 10])
            ax[0].plot(
                rows[:, 1],
                r,
                ".",
                color=c,
                label=f"{lab} (median R {np.median(r):.0f})",
            )
        ax[0].set_ylabel("R = lambda / FWHM")
    else:
        for j, (lab, c) in enumerate(zip(labels, colors, strict=True)):
            ax[0].plot(
                rows[:, 1],
                rows[:, 2 + j],
                ".",
                color=c,
                label=f"{lab} (median {np.median(rows[:, 2 + j]):.1f} px)",
            )
        ax[0].set_ylabel("FWHM [px] (slit image)")
    ax[0].set_title(
        f"{arm}: {len(rows)} isolated sky lines, {slit} slit, box over {SKY_HEIGHT[arm]} px"
    )
    for j, (lab, c) in enumerate(zip(labels, colors, strict=True)):
        ax[1].plot(
            rows[:, 1],
            rows[:, 5 + j],
            ".",
            color=c,
            label=f"{lab} (median {np.median(rows[:, 5 + j]):.2f} px)",
        )
    ax[1].set_ylabel("20-80% edge width [px]")
    ax[1].set_title("line edge sharpness")
    ax[2].plot(
        rows[:, 1],
        rows[:, 9],
        ".",
        color="C0",
        label=f"soxspipe (median {np.median(rows[:, 9]):+.1f} km/s)",
    )
    ax[2].plot(
        rows[:, 1],
        rows[:, 8],
        ".",
        color="C3",
        label=f"PyReduce (median {np.median(rows[:, 8]):+.1f} km/s)",
    )
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("line centre - catalogue [km/s]")
    ax[2].set_title("wavelength zero point on sky lines")
    for a_ in ax:
        a_.set_xlabel("wavelength [nm]")
        a_.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(join(OUT, f"sky_fwhm_{arm}.png"), dpi=110)
    plt.close(fig)
    return rows


# --- 4. extracted spectra ---
def pyreduce_orders(arm, files):
    """Blaze-divided sum of PyReduce science spectra: {m: (wl_nm, flux, sigma)}."""
    traces = traces_of(arm)
    blaze = np.load(join(red_dir(arm), f"soxs_{arm.lower()}.flat_norm.npz"))["blaze"]
    blaze = {
        t.m: np.where(blaze[i] > 0.05 * np.nanmax(blaze[i]), blaze[i], np.nan)
        for i, t in enumerate(traces)
    }
    out = {}
    for f in files:
        spectra = Spectra.read(
            f,
            raw=True,
            continuum_normalization=False,
            barycentric_correction=False,
            radial_velocity_correction=False,
        )
        for sp in spectra.data:
            flux, var = sp.spec / blaze[sp.m], (sp.sig / blaze[sp.m]) ** 2
            if sp.m in out:
                out[sp.m] = (out[sp.m][0], out[sp.m][1] + flux, out[sp.m][2] + var)
            else:
                out[sp.m] = (sp.wave / 10, flux, var)
    return {m: (w, f, np.sqrt(v)) for m, (w, f, v) in out.items()}


def snr_per_nm(flux, sigma, wl):
    return flux / sigma / np.sqrt(np.abs(np.gradient(wl)))


def der_snr(flux, wl):
    """Empirical SNR per sqrt(nm) from the scatter (DER_SNR, Stoehr et al. 2008)."""
    f = np.asarray(flux, float)
    good = np.isfinite(f)
    f, wl = f[good], np.asarray(wl, float)[good]
    if len(f) < 50:
        return np.nan
    noise = 1.482602 / np.sqrt(6) * np.median(np.abs(2 * f[2:-2] - f[:-4] - f[4:]))
    return np.median(f) / noise / np.sqrt(np.median(np.abs(np.gradient(wl))))


def soxspipe_order(sp, m, wl_range):
    """Flux, sigma and wavelength of order m (stare table) or of the merged nod spectrum."""
    if "order" in sp.colnames:
        o = sp[sp["order"] == m]
        return (
            np.asarray(o["wavelengthMean"], float),
            np.asarray(o["extractedFluxOptimal"], float),
            np.sqrt(np.asarray(o["varianceSpectrum"], float)),
        )
    wl = np.asarray(sp["WAVE"], float)
    sel = (wl > wl_range[0]) & (wl < wl_range[1])
    return (
        wl[sel],
        np.asarray(sp["FLUX_COUNTS"], float)[sel],
        np.sqrt(np.asarray(sp["VARIANCE"], float))[sel],
    )


def plot_spectra(arm):
    """Stare (VIS) or A-B nod (NIR) spectra of both pipelines, scaled per order, and their SNR.

    The per-order scale absorbs the different flat-field normalisations (PyReduce
    divides by its extracted flat, which keeps the lamp spectrum).
    """
    if arm == "VIS":
        sets = [
            ([f"{red_dir(arm)}/{f}.vis.science.fits"], soxspipe_orders(arm, st), slit)
            for f, st, slit in STARE[arm]
        ]
    else:
        merged = Table.read(
            sp_file("reduced/*/soxs-nod/20260111T080803_*EXTRACTED_MERGED.fits")
        )
        sets = [
            (
                sorted(
                    glob(join(red_dir(arm), "SOXS.2026-01-11T08*_nod.nir.science.fits"))
                ),
                merged,
                '5.0" nod',
            )
        ]

    ms = sorted(t.m for t in traces_of(arm))
    stats = []
    fig, axes = plt.subplots(
        len(sets) + 1, 1, figsize=(16, 3.2 * (len(sets) + 1)), squeeze=False
    )
    for k, (files, sp, slit) in enumerate(sets):
        pr = pyreduce_orders(arm, files)
        ax = axes[k, 0]
        top = 0
        for i, m in enumerate(ms):
            wl, f, e = pr[m]
            ok = np.isfinite(f) & np.isfinite(wl)
            wl_s, f_s, e_s = soxspipe_order(
                sp, m, (np.nanmin(wl[ok]), np.nanmax(wl[ok]))
            )
            f_p_nm = f / np.abs(np.gradient(wl))
            f_s_nm = f_s / np.abs(np.gradient(wl_s))
            ref = np.interp(wl, wl_s, f_s_nm, left=np.nan, right=np.nan)
            centre = ok & (np.abs(np.arange(len(wl)) - len(wl) / 2) < len(wl) / 4)
            scale = np.nanmedian(f_p_nm[centre] / ref[centre])
            ax.plot(
                wl_s,
                f_s_nm * scale,
                lw=0.5,
                color="C0",
                label="soxspipe" if i == 0 else None,
            )
            ax.plot(
                wl[ok],
                f_p_nm[ok],
                lw=0.5,
                color="C3",
                alpha=0.7,
                label="PyReduce" if i == 0 else None,
            )
            top = max(top, np.nanpercentile(f_s_nm * scale, 99))
            snr_p = snr_per_nm(f, e, wl)
            snr_s = snr_per_nm(f_s, e_s, wl_s)
            axes[-1, 0].plot(
                wl[ok],
                snr_p[ok],
                lw=0.4,
                color="C3",
                alpha=0.5,
                label="PyReduce" if (k, i) == (0, 0) else None,
            )
            axes[-1, 0].plot(
                wl_s,
                snr_s,
                lw=0.4,
                color="C0",
                alpha=0.5,
                label="soxspipe" if (k, i) == (0, 0) else None,
            )
            stats.append(
                (
                    slit,
                    m,
                    np.nanmedian(snr_p[centre]),
                    np.nanmedian(snr_s),
                    der_snr(f[centre], wl[centre]),
                    der_snr(f_s, wl_s),
                )
            )
        ax.set_ylim(0, 1.3 * top)
        ax.set_title(
            f"{arm} {slit}: PyReduce (blaze-divided) and soxspipe, per nm, soxspipe scaled per order"
        )
        ax.legend(fontsize=8)
    axes[-1, 0].set_ylabel("reported SNR per sqrt(nm)")
    axes[-1, 0].set_xlabel("wavelength [nm]")
    axes[-1, 0].set_yscale("log")
    axes[-1, 0].set_ylim(5, None)
    axes[-1, 0].set_title("SNR from each pipeline's own uncertainties (all sets above)")
    axes[-1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(join(OUT, f"spectra_{arm}.png"), dpi=100)
    plt.close(fig)
    return stats


# --- 5. PyReduce uncertainties against the scatter between independent nod pairs ---
def nod_pair_noise(arm):
    """Ratio of the empirical scatter between two A-B pairs to the reported uncertainty.

    Same star, same slit, a few minutes apart: (pair 1 - pair 2) after scaling
    contains no spectral structure, only noise (and seeing/slit-loss changes, which
    the scaling removes to first order).
    """
    pattern = {
        "VIS": "SOXS.2025-10-13T04:5*_nod.vis.science.fits",
        "NIR": "SOXS.2026-01-11T08*_nod.nir.science.fits",
    }
    files = sorted(glob(join(red_dir(arm), pattern[arm])))
    if arm == "NIR":
        first, second = [files[0], files[1]], [files[2], files[3]]
    else:
        first, second = [files[0]], [files[1]]  # A and B of the same 5" pair
    p1, p2 = pyreduce_orders(arm, first), pyreduce_orders(arm, second)
    out = []
    for m in sorted(p1):
        _, f1, e1 = p1[m]
        _, f2, e2 = p2[m]
        n = len(f1)
        c = np.isfinite(f1) & np.isfinite(f2) & (np.abs(np.arange(n) - n / 2) < n / 4)
        k = np.nanmedian(f1[c]) / np.nanmedian(f2[c])
        d = f1[c] - k * f2[c]
        # high-pass: remove slow residual trends with a running median over 25 px
        from scipy.ndimage import median_filter

        d = d - median_filter(d, 25)
        sig = np.sqrt(e1[c] ** 2 + (k * e2[c]) ** 2)
        out.append((m, 1.4826 * np.median(np.abs(d / sig))))
    return out


if __name__ == "__main__":
    for arm in ("VIS", "NIR"):
        print(f"\n===== {arm} =====")
        print(
            "order  soxspipe-map  PyReduce-MPH  MPH-map  max|MPH-map|  sky-map  nsky   (dx/dy)"
        )
        for row in plot_tilt(arm):
            print("%5d  %12.4f  %12.4f  %7.4f  %12.4f  %7.4f  %4d" % row)
        r_sp, r_pr, n_sp, n_pr = plot_arc_R(arm)
        print(
            f"arc lines: R soxspipe {r_sp:.0f} ({n_sp} lines), PyReduce {r_pr:.0f} ({n_pr} lines)"
        )
        rows = plot_sky(arm)
        if len(rows):
            r = rows[:, 1:2] / (rows[:, 2:5] * rows[:, 10:11])
            print(
                f"sky lines ({len(rows)}), soxspipe / PyReduce tilt / PyReduce no tilt:"
            )
            print(
                "   FWHM px   "
                + " / ".join(f"{v:.2f}" for v in np.median(rows[:, 2:5], axis=0))
            )
            print(
                "   R         " + " / ".join(f"{v:.0f}" for v in np.median(r, axis=0))
            )
            print(
                "   edge px   "
                + " / ".join(f"{v:.2f}" for v in np.median(rows[:, 5:8], axis=0))
            )
            for m in np.unique(rows[:, 0]).astype(int):
                s_ = rows[:, 0] == m
                print(
                    f"   order {m:2d}: n={s_.sum():3d}  dv soxspipe {np.median(rows[s_, 9]):+6.1f}  "
                    f"PyReduce {np.median(rows[s_, 8]):+6.1f} km/s"
                )
        print(
            "SNR per sqrt(nm), PyReduce / soxspipe: reported, empirical (DER_SNR, biased high by resampling)"
        )
        for row in plot_spectra(arm):
            print("%-9s  order %2d   %5.0f / %5.0f   %5.0f / %5.0f" % row)
        ratios = nod_pair_noise(arm)
        print(
            "PyReduce: scatter between independent nod spectra / reported uncertainty, per order:"
        )
        print("   " + "  ".join(f"{m}:{r:.2f}" for m, r in ratios))
    print(f"\nplots in {OUT}")
