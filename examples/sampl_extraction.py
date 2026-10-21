"""
Does the 1D extraction change the measured sampling? Repeats the LFC
FWHM / empirical-LSF test of sampl_iq15.py and sampl_vis.py with PyReduce's
slit-decomposition extraction (extract_spectrum, charslit backend) and each
of charslit's pixel weightings, against the plain box sum along the trace.

Needs charslit with the weights= option (CharSlit commit 328e699 or later):

    uv build --wheel -o <dir> ~/CharSlit.git
    uv run --with h5py --with <dir>/charslit-*.whl python examples/sampl_extraction.py
"""

import functools
import multiprocessing
import os
import types

import charslit
import matplotlib.pyplot as plt
import numpy as np
from sampl_iq15 import DATA_DIR, HDF_DIR, LFC_DIR, PLOT, extract_order, process_band
from sampl_vis import FIBER, MODELS, hdf_file, lfc_file, params

from pyreduce import extract as pr_extract

OUT_DIR = os.path.join(DATA_DIR, "ANDES", "reduced", "extraction_cmp")
WEIGHTS = ["none", "unc", "model", "flux"]
OSAMPLE = 10
LAMBDA_SF = 0.1

FRAMES = [
    (
        f"{ch}_iq15",
        os.path.join(LFC_DIR, f"{ch}_iq15_LFC_single_1s.fits"),
        os.path.join(HDF_DIR, f"ANDES_{ch}_master_rotfix_iq15.hdf"),
        {"fiber": 1},
    )
    for ch in ["Y", "J", "H"]
] + [
    (tag, lfc_file(tag), hdf_file(tag), {"fiber": FIBER, **params(tag)})
    for tag in MODELS
]


def use_charslit_weights(weights):
    # PyReduce's charslit wrapper does not pass weights= through, so swap in a
    # backend whose slitdec has it bound.
    os.environ["PYREDUCE_EXTRACTION"] = "charslit"
    pr_extract._backend_cache["charslit"] = types.SimpleNamespace(
        slitdec=functools.partial(charslit.slitdec, weights=weights)
    )


def slitdec_extract(img, x_samp, y_samp, half):
    """Slit-decomposition extraction along the same refined trace as the box sum."""
    box, y_trace, off = extract_order(img, x_samp, y_samp, half=half)
    valid = np.isfinite(box)
    cols = np.nonzero(valid)[0]
    if cols.size < 2:
        return box, y_trace, off
    ycen = np.interp(np.arange(img.shape[1]), cols, y_trace[cols])
    # extract_spectrum puts the trace at floor(ycen); the box sum centres on
    # round(y), so shift by half a pixel to cut the same rows.
    spec, _slitf, mask, _unc = pr_extract.extract_spectrum(
        img,
        ycen + 0.5,
        (half, half),
        [cols[0], cols[-1] + 1],
        osample=OSAMPLE,
        lambda_sf=LAMBDA_SF,
        reject_threshold=0,
    )
    spec = np.where(mask | ~valid, np.nan, spec)
    return spec, y_trace, off


def summary(cells):
    a = {
        k: np.array([c[k] for c in cells])
        for k in ("fwhm", "fwhm_gline", "R", "R_gline")
    }
    return {k: np.nanmedian(v) for k, v in a.items()}


def matched_ratio(cells, ref, key):
    """Per-cell ratio cells/ref of key, matched by (order, x-bin)."""
    rmap = {(c["m"], c["x0"]): c for c in ref}
    pairs = [(c, rmap[(c["m"], c["x0"])]) for c in cells if (c["m"], c["x0"]) in rmap]
    x = np.array([c["x"] for c, _ in pairs])
    r = np.array([c[key] / r[key] for c, r in pairs])
    return x, r


def run(task):
    (band, lfc, hdf, kw), method = task
    if method == "box":
        extract = extract_order
    else:
        use_charslit_weights(method)
        extract = slitdec_extract
    cells = process_band(
        f"{band} [{method}]",
        lfc_file=lfc,
        hdf_file=hdf,
        output_dir=os.path.join(OUT_DIR, band, method),
        prefix=f"{band.lower()}_{method}",
        extract=extract,
        plots=False,
        **kw,
    )
    return band, method, cells


if __name__ == "__main__":
    tasks = [(frame, method) for frame in FRAMES for method in ["box", *WEIGHTS]]
    results = {band: {} for band, *_ in FRAMES}  # band -> method -> cells
    with multiprocessing.Pool(int(os.environ.get("NPROC", 8))) as pool:
        for band, method, cells in pool.imap_unordered(run, tasks):
            if cells:
                results[band][method] = cells
    results = {
        band: {m: res[m] for m in ["box", *WEIGHTS] if m in res}
        for band, res in results.items()
    }

    print(f"\n{'=' * 78}\nMedians over cells; ratio = median per-cell eLSF FWHM / box")
    print(
        f"  {'band':<7} {'method':<6} {'eLSF FWHM':>9} {'Gauss FWHM':>10} "
        f"{'eLSF/box':>8} {'R eLSF':>8} {'R Gauss':>8}"
    )
    for band, res in results.items():
        for method, cells in res.items():
            s = summary(cells)
            _, r = matched_ratio(cells, res["box"], "fwhm")
            print(
                f"  {band:<7} {method:<6} {s['fwhm']:>9.3f} {s['fwhm_gline']:>10.3f} "
                f"{np.nanmedian(r):>8.4f} {s['R'] / 1e3:>7.1f}k {s['R_gline'] / 1e3:>7.1f}k"
            )

    fig, axs = plt.subplots(
        len(results), 2, figsize=(13, 2.6 * len(results)), sharex="row"
    )
    for row, (band, res) in enumerate(results.items()):
        for col, key in enumerate(["fwhm", "fwhm_gline"]):
            ax = axs[row, col]
            for method in WEIGHTS:
                if method in res:
                    x, r = matched_ratio(res[method], res["box"], key)
                    ax.plot(x, r, ".", ms=4, label=method)
            ax.axhline(1, color="k", lw=0.6)
            name = "eLSF" if key == "fwhm" else "per-line Gaussian"
            ax.set_title(f"{band}: {name} FWHM, slitdec / box", fontsize=9)
            if row == len(results) - 1:
                ax.set_xlabel("x [pixel]")
    axs[0, 0].legend(title="weights", fontsize=8, ncol=4)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "extraction_fwhm_ratio.png")
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")

    if PLOT:
        plt.show()
