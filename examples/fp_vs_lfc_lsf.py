"""
How well can the LSF be recovered from a resolved Fabry-Perot comb, compared
with a laser frequency comb, and how much does it matter what we assume for
the intrinsic FP line shape? ANDES R band, single fiber, E2E simulations.

Frames (fiber 16, HDF ANDES_123_R3, ~1.5-1.7e7 photons per order), from ~/ANDES/E2E/src:

    # LFC, 100 lines per order (see sampl_vis.py)
    uv run andes-sim simulate --hdf HDF/ANDES_123_R3.hdf --source lfc --fiber 16 \\
        --flux 50 --output-dir ../R --output-name R0_LFC_fiber16_1s.fits
    # FP, pure Airy with the CU baseline etalon: 10 mm gap, F = 20.7 (Airy fit
    # to the CU model spectrum SED/FP_simulation_RIZ_finesse_23.csv)
    uv run andes-sim simulate --band R --source fp --fiber 16 --finesse 20.7 \\
        --fp-gap 9.993082 --scaling 1.5e5 --output-dir ../R \\
        --output-name R0_FPcu_fiber16_1s.fits
    # FP, CU-like profile: Airy(F_mirror 23.2) (x) defect Gaussian (sigma
    # 7.6e-3 FSR at 680 nm), the model that fits the CU spectrum to 0.1 % rms;
    # and the same with a one-sided beam-divergence box (0.0147 FSR at 680 nm,
    # theta_max = 1 mrad, a ~75 m/s mean blueshift)
    uv run python scripts/fp_profile_csv.py ../R/FPcu_R.csv --sigma 7.6e-3
    uv run python scripts/fp_profile_csv.py ../R/FPdiv_R.csv --sigma 7.6e-3 --box 0.0147
    uv run andes-sim simulate --band R --source ../R/FPcu_R.csv --flux-unit ph/s \\
        --scaling 1 --fiber 16 --output-dir ../R --output-name R0_FPcuprof_fiber16_1s.fits
    uv run andes-sim simulate --band R --source ../R/FPdiv_R.csv --flux-unit ph/s \\
        --scaling 1 --fiber 16 --output-dir ../R --output-name R0_FPdivprof_fiber16_1s.fits
    # FP, CU baseline per FP DAR v3.0 Table 2 + Fig. 5: F_mirror 19.3, defects
    # to F_eff 18.4, plus 2.4 mrad beam divergence (Fig. 5 linewidths, F_eff ~10)
    uv run python scripts/fp_profile_csv.py ../R/FPbase_R.csv --finesse 19.3 \\
        --sigma 0.0048 --box 0.0869
    uv run andes-sim simulate --band R --source ../R/FPbase_R.csv --flux-unit ph/s \\
        --scaling 1 --fiber 16 --output-dir ../R --output-name R0_FPbaseprof_fiber16_1s.fits
    # delta comb at the FP peak wavelengths (a CSV in ph/s becomes discrete
    # lines in pyechelle), same photon budget as the FP: the reference LSF ...
    uv run andes-sim simulate --band R --source ../R/FPpeaks_comb_R.csv --fiber 16 \\
        --flux-unit ph/s --scaling 1.875 --output-dir ../R \\
        --output-name R0_FPcomb_fiber16_1s.fits
    # ... and 20x brighter in order 90 only, to check the reference itself
    uv run andes-sim simulate --band R --source ../R/FPpeaks_comb_R.csv --fiber 16 \\
        --flux-unit ph/s --scaling 37.5 --wl-min 676 --wl-max 688 --output-dir ../R \\
        --output-name R0_FPcombx20_o90_fiber16.fits
    # the fiber's flat (trace offset, slit function)
    uv run andes-sim simulate --band R --source flat --fiber 16 --flux 300 \\
        --output-dir ../R --output-name R0_FLAT_fiber16_1s.fits

All frames are extracted along the HDF model trace (offset measured on the
fiber's flat, vertical slit, no curvature), with a box sum (--extraction box,
the default) or slit decomposition with the flat's slit function (flat). With
its own slit function per frame (own), slitdec changes the extracted line
shape between noise realisations, see extract_one(); even with the flat's
slit function the slitdec spectrum biases line-shape fits in an S/N dependent
way, so box is the clean case.

Per block the same non-parametric LSF model (linear hats on a sub-pixel grid,
pixel-integrated) is fitted to each frame:

    spec(j) = sum_i A_i * sum_s w_is * L(j - x_i - v_is)

Combs: one delta per line. FP: each line is its own period of the assumed
intrinsic profile, sampled finely (v_is, w_is). Line positions x_i come from
the HDF transformation at the fiber centre, exactly as pyechelle places
photons, so no fit has free line positions: an unmodelled shift or asymmetry
of the FP lines goes into the LSF centroid. Blocks are the wavelength
intervals in which pyechelle uses one tabulated ZEMAX PSF (nearest-PSF
lookup), so the true LSF changes only smoothly inside a block.

Everything is compared with the equal-budget comb (formal centroid error
~0.7 m/s per block with box extraction, small against the FP's ~2 m/s).

Usage:
    uv run --isolated --python 3.13 --with h5py --with-editable . python examples/fp_vs_lfc_lsf.py \\
        [--extraction box|flat|own] [--reextract] [--orders 90 85] [--summarize]
"""

import argparse
import functools
import json
import multiprocessing
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from sampl_iq15 import (
    DATA_DIR,
    HDF_DIR,
    PLOT,
    extract_order,
    load_trace_model,
    max_phase_gap,
)
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from pyreduce import extract as pr_extract

E2E_R = os.path.expanduser("~/ANDES/E2E/R")
# optical model: R0 is the simulator default, R1 the less well aligned
# thermal variant T0019 (slightly lower resolution); set FPLSF_MODEL=R1 to
# analyse the R1_* frames, simulated with --hdf HDF/<model>.hdf
MODELS = {"R0": "ANDES_123_R3", "R1": "ANDES_full_F18A33_win_jmr_MC_T0019_Rband_p0"}
MODEL = os.environ.get("FPLSF_MODEL", "R0")
HDF = os.path.join(HDF_DIR, f"{MODELS[MODEL]}.hdf")
FIBER = 16
# name -> (file, line kind, orders with light or None for all)
FRAMES = {
    "LFC": (f"{MODEL}_LFC_fiber{FIBER}_1s.fits", "lfc", None),
    "COMB": (f"{MODEL}_FPcomb_fiber{FIBER}_1s.fits", "comb", None),
    "FP": (f"{MODEL}_FPcu_fiber{FIBER}_1s.fits", "fp", None),
    "FPCU": (f"{MODEL}_FPcuprof_fiber{FIBER}_1s.fits", "fp", None),
    "FPDIV": (f"{MODEL}_FPdivprof_fiber{FIBER}_1s.fits", "fp", None),
    "FPBASE": (f"{MODEL}_FPbaseprof_fiber{FIBER}_1s.fits", "fp", None),
    "TRUTH": (f"{MODEL}_FPcombx20_o90_fiber{FIBER}.fits", "comb", [90]),
}
FLAT = f"{MODEL}_FLAT_fiber{FIBER}_1s.fits"
LFC_LINES = os.path.join(E2E_R, "R_LFC_lines.txt")
FP_GAP_NM = 9.993082e6

# Intrinsic FP profiles (finesse, defect sigma, divergence box), widths in
# units of the FSR at LAM_REF, scaling as 1/lambda (scripts/fp_profile_csv.py).
# finesse "best": the pure Airy that best fits (finesse, sigma, box) per block.
LAM_REF = 680.0
F_AIRY = 20.7
CU = (23.2, 7.6e-3, 0.0)
DIV = (23.2, 7.6e-3, 0.0147)
# CU baseline etalon per FP DAR v3.0: Table 2 optics (F_mirror 19.3, defects to
# F_eff 18.4) plus the collimation term that gives the Fig. 5 linewidths
# (2.1-3.75 pm over RIZ, F_eff ~10): divergence box 0.0869 FSR at 680 nm,
# theta_max 2.4 mrad (250 um fibre, ~50 mm collimator), ~400 m/s mean blueshift
BASE = (19.3, 4.8e-3, 0.0869)
SWEEP = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5]
# study -> {fit name: (frame, assumed profile)}
STUDIES = {
    "reference": {"LFC (100/order)": ("LFC", None), "COMB": ("COMB", None)},
    "finesse": {
        f"F x{f:g}": ("FP", (F_AIRY * f, 0.0, 0.0)) for f in SWEEP
    },  # truth: Airy F=20.7
    "cu": {
        "exact": ("FPCU", CU),
        "best-fit Airy": ("FPCU", ("best", *CU)),
        "Airy, F_mirror": ("FPCU", (CU[0], 0.0, 0.0)),
    },
    "div": {
        "exact": ("FPDIV", DIV),
        "symmetric part": ("FPDIV", CU),
        "best-fit Airy (sym.)": ("FPDIV", ("best", *CU)),
    },
    "baseline": {
        "exact": ("FPBASE", BASE),
        "divergence x0.9": ("FPBASE", (*BASE[:2], 0.9 * BASE[2])),
        "divergence x1.1": ("FPBASE", (*BASE[:2], 1.1 * BASE[2])),
        "divergence x0.8": ("FPBASE", (*BASE[:2], 0.8 * BASE[2])),
        "divergence x1.2": ("FPBASE", (*BASE[:2], 1.2 * BASE[2])),
        "no collimation (Table 2)": ("FPBASE", (*BASE[:2], 0.0)),
        "best-fit Airy": ("FPBASE", ("best", *BASE)),
        "Airy F=11": ("FPBASE", (11.0, 0.0, 0.0)),
    },
}
FITS = {f"{s}/{n}": v for s, d in STUDIES.items() for n, v in d.items()}
REF = "reference/COMB"

OUT_DIR = os.path.join(
    DATA_DIR, "ANDES", "reduced", "fp_lsf_R" if MODEL == "R0" else f"fp_lsf_{MODEL}"
)
APERTURE_HALF = 8
OSAMPLE = 10
LAMBDA_SF = 0.1

# LSF window half-width in px, cut to half the FP/comb line spacing per block:
# a near-regular comb only constrains the LSF at harmonics of 1/spacing, so
# any support beyond +-spacing/2 is degenerate with the neighbouring lines
LSF_HALF = 10.0
KNOT = 0.25
REG = 1e-3  # second-difference penalty, relative to the mean data-term diagonal
N_ITER = 30
# a delta comb whose spacing is close to an integer number of pixels leaves
# sub-pixel phases unsampled; such blocks are no reference
MAX_PHASE_GAP = 1.5 * KNOT
MOM_HALF = 6.0  # window for the LSF centroid
MAX_CHI2 = 5.0
V_STEP = 0.02  # px, sampling of one FP period
C_MS = 299792458.0
N_PSI = 8192


def extract_one(img, y_trace, mode, slitf=None):
    """1D extraction along a fixed trace (NaN where the aperture is off the detector).

    mode "flat": slit decomposition with the slit function preset (from the
    flat), which makes it a fixed linear operator, the same for all frames.
    "own": slitdec fits its own slit function to each frame; with tilted line
    images under a vertical-slit model that changes the extracted line shape
    by up to ~0.1 px between noise realisations. "box": plain sum over the
    aperture rows.
    Returns (spec, unc, slitf).
    """
    valid = np.isfinite(y_trace)
    cols = np.nonzero(valid)[0]
    if mode == "box":
        spec = np.full(img.shape[1], np.nan)
        for i in cols:
            yc = int(round(y_trace[i]))
            spec[i] = img[yc - APERTURE_HALF : yc + APERTURE_HALF + 1, i].sum()
        return spec, np.sqrt(np.fmax(spec, 0)), None
    ycen = np.interp(np.arange(img.shape[1]), cols, y_trace[cols])
    spec, slitf, mask, unc = pr_extract.extract_spectrum(
        img,
        ycen + 0.5,
        (APERTURE_HALF, APERTURE_HALF),
        [cols[0], cols[-1] + 1],
        osample=OSAMPLE,
        lambda_sf=LAMBDA_SF,
        reject_threshold=0,
        preset_slitfunc=slitf if mode == "flat" else None,
    )
    bad = mask | ~valid
    return np.where(bad, np.nan, spec), np.where(bad, np.nan, unc), slitf


def extract_frames(reextract, mode):
    """{frame: {m: (spec, unc)}}, cached in OUT_DIR; missing frames are skipped.

    The trace (model trace + offset measured on the flat) and, for mode
    "flat", the slit function per order come from the fiber's flat frame.
    """
    model = load_trace_model(HDF, fiber=f"fiber_{FIBER}")
    flat = None
    out = {}
    for name, (fname, _kind, lit) in FRAMES.items():
        fname = os.path.join(E2E_R, fname)
        cache = os.path.join(OUT_DIR, f"{name}_fib{FIBER}_{mode}.npz")
        if os.path.exists(cache) and not reextract:
            d = np.load(cache)
            ms = sorted({int(k[4:]) for k in d.files if k.startswith("spec")})
            out[name] = {m: (d[f"spec{m}"], d[f"unc{m}"]) for m in ms}
            continue
        if not os.path.exists(fname):
            print(f"  {name}: {fname} not found, skipping")
            continue
        if flat is None:
            fimg = fits.getdata(os.path.join(E2E_R, FLAT)).astype(float)
            flat = {}
            for m in model:
                x_samp, y_samp, _wl = model[m]
                _box, y_trace, off = extract_order(
                    fimg, x_samp, y_samp, half=APERTURE_HALF
                )
                _s, _u, slitf = extract_one(fimg, y_trace, "own")
                flat[m] = (y_trace, slitf)
                print(f"  flat m={m}: trace offset {off:+.2f} px")
        img = fits.getdata(fname).astype(float)
        res = {}
        for m in sorted(lit or model, reverse=True):
            spec, unc, _ = extract_one(img, flat[m][0], mode, flat[m][1])
            res[m] = (spec, unc)
        np.savez(
            cache,
            **{f"spec{m}": s for m, (s, _u) in res.items()},
            **{f"unc{m}": u for m, (_s, u) in res.items()},
        )
        out[name] = res
    return out


class OrderGeometry:
    """Fiber-centre x(lambda) of one order, as pyechelle computes it.

    pyechelle splines each affine parameter against wavelength, maps the slit
    point (0.5, 0.5) and bins with floor(), so pixel j covers [j, j+1) and
    its centre is j + 0.5.
    """

    def __init__(self, h5, m):
        d = h5[f"CCD_1/fiber_{FIBER}/order{m}"][()]
        o = np.argsort(d["wavelength"])
        wl = d["wavelength"][o] * 1e3  # nm
        sp = {k: CubicSpline(wl, d[k][o]) for k in d.dtype.names if k != "wavelength"}
        lam = np.linspace(wl[0], wl[-1], 40001)
        rot, sx, sy, sh = (
            sp[k](lam) for k in ("rotation", "scale_x", "scale_y", "shear")
        )
        x = (
            0.5 * sx * np.cos(rot)
            - 0.5 * sy * np.sin(rot + sh)
            + sp["translation_x"](lam)
            - 0.5
        )
        if np.any(np.diff(x) <= 0):
            raise ValueError(f"order {m}: x(lambda) not monotonic")
        self.lam, self.x = lam, x
        psf = h5[f"CCD_1/fiber_{FIBER}/psf_order_{m}"]
        pw = np.sort([psf[k].attrs["wavelength"] for k in psf]) * 1e3
        # pyechelle: idx = int((wl - pw[0]) / (pw[1] - pw[0]))
        self.psf_edges = pw[0] + np.arange(len(pw)) * (pw[1] - pw[0])

    def x_of(self, lam):
        return np.interp(lam, self.lam, self.x, left=np.nan, right=np.nan)

    def lam_of(self, x):
        return np.interp(x, self.x, self.lam, left=np.nan, right=np.nan)

    def dlam_dx(self, x):
        return np.interp(x, self.x[1:], np.diff(self.lam) / np.diff(self.x))


PSI = (np.arange(N_PSI) - N_PSI // 2) / N_PSI  # phase within one FSR


def airy_psi(finesse):
    return 1 / (1 + (2 * finesse / np.pi) ** 2 * np.sin(np.pi * PSI) ** 2)


@functools.lru_cache(maxsize=4096)
def fp_template(finesse, sigma, box, lam_c):
    """One FSR of Airy (x) Gaussian (x) one-sided box on PSI.

    PSI > 0 is the blue side (larger 2d/lambda), where divergence moves the
    light. sigma and box are FSR units at LAM_REF and scale as 1/lambda.
    """
    s = LAM_REF / lam_c
    sigma, box = sigma * s, box * s
    if sigma <= 0 and box <= 0:
        return airy_psi(finesse)
    ker = np.exp(-0.5 * (PSI / max(sigma, 1e-7)) ** 2)
    if box > 1 / N_PSI:
        ker = np.convolve(np.where((PSI >= 0) & (PSI < box), 1.0, 0.0), ker, "same")
    ker = np.roll(ker / ker.sum(), -(N_PSI // 2))
    return np.real(np.fft.ifft(np.fft.fft(airy_psi(finesse)) * np.fft.fft(ker)))


@functools.lru_cache(maxsize=4096)
def best_airy_finesse(finesse, sigma, box, lam_c):
    """Finesse of the pure Airy (free amplitude) closest to the given profile."""
    t = fp_template(finesse, sigma, box, lam_c)

    def cost(f):
        a = airy_psi(f)
        return np.sum((t - a * (a @ t) / (a @ a)) ** 2)

    return minimize_scalar(cost, bounds=(5, 60), method="bounded").x


def resolve_profile(prof, lam_c):
    if prof[0] == "best":
        return (round(best_airy_finesse(*prof[1:], round(lam_c, 1)), 4), 0.0, 0.0)
    return prof


def line_sources(kind, geo, lo, hi, prof=None):
    """[(x_i, v_i, w_i)] for all lines whose light can reach pixels lo..hi."""
    pad = LSF_HALF + 30
    lam_lo = geo.lam_of(max(lo - pad, geo.x[0]))
    lam_hi = geo.lam_of(min(hi + pad, geo.x[-1]))
    if kind == "lfc":
        lam = np.loadtxt(LFC_LINES, comments="#")
        lam = lam[(lam > lam_lo) & (lam < lam_hi)]
        return [(geo.x_of(lk), np.zeros(1), np.ones(1)) for lk in lam]
    k = np.arange(np.ceil(2 * FP_GAP_NM / lam_hi), np.floor(2 * FP_GAP_NM / lam_lo) + 1)
    if kind == "comb":
        return [(geo.x_of(2 * FP_GAP_NM / kk), np.zeros(1), np.ones(1)) for kk in k]
    lam_c = round(float(geo.lam_of(0.5 * (lo + hi))), 1)
    t = fp_template(*resolve_profile(prof, lam_c), lam_c)
    out = []
    for kk in k:
        xi = geo.x_of(2 * FP_GAP_NM / kk)
        # one FSR, between the phases half-way to the neighbouring peaks
        xa = geo.x_of(2 * FP_GAP_NM / (kk + 0.5))
        xb = geo.x_of(2 * FP_GAP_NM / (kk - 0.5))
        if not (np.isfinite(xi) and np.isfinite(xa) and np.isfinite(xb)):
            continue
        xs = np.arange(xa, xb, V_STEP)
        lam_s = geo.lam_of(xs)
        psi = 2 * FP_GAP_NM / lam_s - kk
        # source is ph/s per log-lambda sample, so photons per px scale with dlnlam/dx
        w = np.interp(psi, PSI, t, period=1) * geo.dlam_dx(xs) / lam_s
        out.append((xi, xs - xi, w / w.sum()))
    return out


def design(lines, pix, half):
    """G[i, j, k]: hat k of the LSF, convolved with line i's source, at pixel j."""
    knots = np.arange(-half, half + KNOT / 2, KNOT)
    nk = knots.size
    G = np.zeros((len(lines), pix.size, nk))
    for i, (xi, v, w) in enumerate(lines):
        u = pix[:, None] - xi - v[None, :]
        t = (u + half) / KNOT
        k0 = np.floor(t).astype(int)
        f = t - k0
        jj = np.broadcast_to(np.arange(pix.size)[:, None], u.shape)
        ww = np.broadcast_to(w[None, :], u.shape)
        for kk, wk in ((k0, ww * (1 - f)), (k0 + 1, ww * f)):
            ok = (kk >= 0) & (kk < nk)
            G[i] += np.bincount(
                (jj[ok] * nk + kk[ok]), weights=wk[ok], minlength=pix.size * nk
            ).reshape(pix.size, nk)
    return G, knots


def fit_lsf(G, knots, y, ivar):
    """Alternating LSQ for line amplitudes and LSF hats; NaN result if it diverges."""
    try:
        return _fit_lsf(G, knots, y, ivar)
    except np.linalg.LinAlgError:
        nan = np.full(knots.size, np.nan)
        return {
            "c": nan,
            "mu": np.nan,
            "mu_err": np.nan,
            "fwhm": np.nan,
            "chi2": np.nan,
            "nlines": G.shape[0],
        }


def _fit_lsf(G, knots, y, ivar):
    nk = knots.size
    c = np.exp(-0.5 * (knots / 1.9) ** 2)
    c /= c.sum() * KNOT
    D = np.diff(np.eye(nk), 2, axis=0)
    sw = np.sqrt(ivar)
    for _ in range(N_ITER):
        P = G @ c  # (nline, npix)
        A, *_ = np.linalg.lstsq((P * sw).T, y * sw, rcond=None)
        M = np.tensordot(A, G, axes=(0, 0))  # (npix, nk)
        H = (M * ivar[:, None]).T @ M
        cov = np.linalg.inv(H + REG * np.mean(np.diag(H)) * D.T @ D)
        c_new = cov @ (M.T @ (ivar * y))
        s = c_new.sum() * KNOT
        c_new /= s
        done = np.max(np.abs(c_new - c)) < 1e-7 * c.max()
        c = c_new
        if done:
            break
    A *= s
    model = np.tensordot(A, G, axes=(0, 0)) @ c
    chi2 = np.sum(ivar * (y - model) ** 2) / (y.size - nk - A.size)
    cov /= s**2
    win = np.abs(knots) <= min(MOM_HALF, knots[-1])
    mu = np.sum(knots[win] * c[win]) / np.sum(c[win])
    g = np.where(win, (knots - mu) / np.sum(c[win]), 0.0)
    return {
        "c": c,
        "mu": mu,
        "mu_err": np.sqrt(g @ cov @ g),
        "fwhm": half_max_width(knots, c),
        "chi2": chi2,
        "nlines": A.size,
    }


def half_max_width(knots, c):
    xf = np.linspace(knots[0], knots[-1], 20001)
    yf = np.interp(xf, knots, c)
    above = np.nonzero(yf >= 0.5 * yf.max())[0]
    return xf[above[-1]] - xf[above[0]]


def analyse_block(spectra, geo, m, lo, hi):
    pix = np.arange(lo, hi)
    res = {"m": m, "x": 0.5 * (lo + hi), "lam": float(geo.lam_of(0.5 * (lo + hi)))}
    res["ms_per_px"] = C_MS * geo.dlam_dx(res["x"]) / res["lam"]
    xfp = np.sort([ln[0] for ln in line_sources("comb", geo, lo, hi)])
    spacing = np.min(np.diff(xfp[(xfp >= lo) & (xfp < hi)]))
    res["half"] = half = min(LSF_HALF, np.floor(0.5 * spacing / KNOT) * KNOT)
    fits_ = dict(FITS)
    if m in spectra.get("TRUTH", {}):
        fits_["TRUTH"] = ("TRUTH", None)
    for name, (frame, prof) in fits_.items():
        if m not in spectra.get(frame, {}):
            continue
        spec, unc = spectra[frame][m]
        y = spec[lo:hi]
        # slitdec's uncertainty is 0 where its model is 0 (between comb lines)
        s = np.sqrt(np.fmax(unc[lo:hi] ** 2, np.fmax(y, 0) + 1.0))
        ok = np.isfinite(y) & np.isfinite(s)
        if ok.sum() < 0.9 * y.size:
            return None
        lines = line_sources(FRAMES[frame][1], geo, lo, hi, prof)
        lines = [ln for ln in lines if np.isfinite(ln[0])]
        G, knots = design(lines, pix[ok], half)
        keep = G.reshape(len(lines), -1).any(axis=1)
        res[name] = fit_lsf(G[keep], knots, y[ok], 1.0 / s[ok] ** 2)
        xl = np.array([ln[0] for ln in lines])
        res[name]["phase_gap"] = max_phase_gap(xl[(xl >= lo) & (xl < hi)])
        if prof is not None:
            res[name]["profile"] = resolve_profile(prof, round(res["lam"], 1))
    res["knots"] = knots
    if res[REF]["phase_gap"] > MAX_PHASE_GAP:
        print(f"  m={m} x={res['x']:.0f}: comb phase gap too large, dropped")
        return None
    return res


def analyse_order(args):
    m, spectra = args
    blocks = []
    with h5py.File(HDF) as h5:
        geo = OrderGeometry(h5, m)
    edges_x = geo.x_of(geo.psf_edges)
    ncol = spectra["FP"][m][0].size
    for lo, hi in zip(edges_x[:-1], edges_x[1:], strict=True):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        lo, hi = int(np.ceil(lo)), int(np.floor(hi))
        if lo < LSF_HALF or hi > ncol - LSF_HALF or hi - lo < 100:
            continue
        b = analyse_block(spectra, geo, m, lo, hi)
        if b is not None:
            blocks.append(b)
    print(f"  m={m}: {len(blocks)} blocks")
    return blocks


def stats(blocks, key, ref=REF):
    """Summary of fit `key` against the reference, all centroids in m/s."""
    # a fit that lands in a wrong alternating-LSQ solution shows chi2_red in the
    # hundreds (all others ~1.4); count it as failed
    blocks = [
        b
        for b in blocks
        if np.isfinite(b[key]["mu"])
        and np.isfinite(b[ref]["mu"])
        and b[key]["chi2"] < MAX_CHI2
    ]
    sc = np.array([b["ms_per_px"] for b in blocks])
    lam = np.array([b["lam"] for b in blocks])
    d = np.array([b[key]["mu"] - b[ref]["mu"] for b in blocks]) * sc
    s = np.array([np.hypot(b[key]["mu_err"], b[ref]["mu_err"]) for b in blocks]) * sc
    slope = np.polyfit(lam - lam.mean(), d, 1, w=1 / s)[0]
    dfw = np.array([b[key]["fwhm"] - b[ref]["fwhm"] for b in blocks])
    shape = np.array(
        [
            np.sqrt(np.mean((b[key]["c"] - b[ref]["c"]) ** 2)) / b[ref]["c"].max()
            for b in blocks
        ]
    )
    sig = np.array([b[key]["mu_err"] for b in blocks]) * sc
    return {
        "n": len(blocks),
        "sigma_mu": float(np.median(sig)),
        "d_mu_mean": float(np.average(d, weights=1 / s**2)),
        "d_mu_std": float(np.std(d - np.average(d, weights=1 / s**2))),
        "pull_rms": float(
            np.sqrt(np.mean(((d - np.average(d, weights=1 / s**2)) / s) ** 2))
        ),
        "chroma_100nm": float(slope * 100),
        "d_fwhm": float(np.median(dfw)),
        "d_fwhm_rel": float(
            np.median(dfw / np.array([b[ref]["fwhm"] for b in blocks]))
        ),
        "shape_rms": float(np.median(shape)),
        "chi2": float(np.median([b[key]["chi2"] for b in blocks])),
    }


def summarize(blocks):
    print(
        f"\n{'=' * 96}\n{len(blocks)} blocks, fiber {FIBER}, R band ({MODELS[MODEL]}), extraction {TAG}"
    )
    print("fit - COMB (equal-photon delta comb at the FP peaks)")
    hdr = (
        f"  {'fit':<34} {'sig_mu':>7} {'<d_mu>':>8} {'std':>6} {'pull':>5}"
        f" {'chrom.':>7} {'dFWHM':>7} {'shape':>6} {'chi2':>5} {'n':>4}"
    )
    out = {}
    for study, fits_ in STUDIES.items():
        print(f"\n[{study}]\n{hdr}")
        for name in fits_:
            key = f"{study}/{name}"
            if key not in blocks[0]:
                continue
            if key == REF:
                continue
            st = stats(blocks, key)
            out[key] = st
            print(
                f"  {key:<34} {st['sigma_mu']:7.2f} {st['d_mu_mean']:+8.2f}"
                f" {st['d_mu_std']:6.2f} {st['pull_rms']:5.2f} {st['chroma_100nm']:+7.2f}"
                f" {100 * st['d_fwhm_rel']:+6.2f}% {100 * st['shape_rms']:5.2f}% {st['chi2']:5.2f} {st['n']:4d}"
            )
    print(
        "  (m/s: formal error, mean offset, scatter about it, pull of the scatter,"
        " linear trend per 100 nm; FWHM and shape rms relative to COMB)"
    )
    tb = [b for b in blocks if "TRUTH" in b]
    if tb:
        print(f"\nCheck against TRUTH (20x comb, {len(tb)} blocks of order 90)")
        for key in [REF, "finesse/F x1", "reference/LFC (100/order)"]:
            st = stats(tb, key, ref="TRUTH")
            out[f"truth:{key}"] = st
            print(
                f"  {key:<34} <d_mu> {st['d_mu_mean']:+6.2f}  std {st['d_mu_std']:5.2f}"
                f"  pull {st['pull_rms']:5.2f}  dFWHM {st['d_fwhm']:+.3f} px"
            )
    with open(os.path.join(OUT_DIR, f"summary_{TAG}.json"), "w") as f:
        json.dump(out, f, indent=1)


def plot_overview(blocks):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    lam = np.array([b["lam"] for b in blocks])
    for ax, study in zip(axs, ["finesse", "cu", "baseline"], strict=True):
        for name in STUDIES[study]:
            key = f"{study}/{name}"
            if key not in blocks[0]:
                continue
            d = [(b[key]["mu"] - b[REF]["mu"]) * b["ms_per_px"] for b in blocks]
            ax.plot(lam, d, ".", ms=3, label=name)
        ax.set_title(study)
        ax.set_xlabel("wavelength [nm]")
        ax.legend(fontsize=7)
    axs[0].set_ylabel("LSF centroid, fit - COMB [m/s]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"studies_{TAG}.png"), dpi=110)
    print(f"\nPlots in {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reextract", action="store_true")
    ap.add_argument("--orders", type=int, nargs="*", default=None)
    ap.add_argument("--extraction", choices=["flat", "box", "own"], default="box")
    ap.add_argument(
        "--summarize", action="store_true", help="only re-summarise saved blocks"
    )
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    global TAG
    TAG = args.extraction
    saved = os.path.join(OUT_DIR, f"blocks_{TAG}.npy")
    if args.summarize:
        blocks = list(np.load(saved, allow_pickle=True))
        summarize(blocks)
        plot_overview(blocks)
        return

    spectra = extract_frames(args.reextract, args.extraction)
    orders = args.orders or sorted(spectra["FP"])
    tasks = [(m, {f: {m: s[m]} for f, s in spectra.items() if m in s}) for m in orders]
    with multiprocessing.Pool(int(os.environ.get("NPROC", 8))) as pool:
        blocks = [b for bl in pool.map(analyse_order, tasks) for b in bl]
    np.save(saved, blocks, allow_pickle=True)
    summarize(blocks)
    plot_overview(blocks)
    if PLOT:
        plt.show()


TAG = "box"  # extraction mode, set in main()

if __name__ == "__main__":
    main()
