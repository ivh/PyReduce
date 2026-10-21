"""
ANDES UBVRIZ sampling test for all optical models (HDF files) of the visual
arms, using the same Gaussian and empirical-LSF analysis as sampl_iq15.py,
plus side-by-side comparisons of the models on common colour scales.

The single-fiber LFC frames are generated with the E2E simulator, from
~/ANDES/E2E/src:

    uv run andes-sim simulate --hdf HDF/<model>.hdf --source lfc --fiber 16 \\
        --flux 50 --output-dir ../<band> --output-name <tag>_LFC_fiber16_1s.fits

Usage: sampl_vis.py [tag ...]   (default: all models)
"""

import multiprocessing
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sampl_iq15 import DATA_DIR, HDF_DIR, PLOT, process_band
from scipy.interpolate import griddata

E2E_DIR = os.path.expanduser("~/ANDES/E2E")
OUT_DIR = os.path.join(DATA_DIR, "ANDES", "reduced")
FIBER = 16  # centre of slit A
# lines are ~4-6 px FWHM here, about twice the YJH iq15 sampling
VIS_PARAMS = {
    "aperture_half": 8,
    "peak_window": 10,
    "elsf_half": 14.0,
    "elsf_xbin": 1024,
    "elsf_knot": 0.4,
}

# tag -> (band directory under E2E_DIR, HDF model)
MODELS = {
    "U": ("U", "ANDES_U_v88"),
    "B": ("B", "ANDES_B_v88"),
    "V": ("V", "ANDES_V_v88"),
    "R0": ("R", "ANDES_123_R3"),
    "R1": ("R", "ANDES_full_F18A33_win_jmr_MC_T0019_Rband_p0"),
    "R2": ("R", "Andes_full_F18A33_win_jmr_MC_T0108_Rband_P0_cfg1"),
    "IZ0": ("IZ", "ANDES_123_IZ3"),
    "IZ1": ("IZ", "Andes_F18A33_VM246aa_win_jmr9_MC_T0045_IZband_P0_cf1"),
    "IZ2": ("IZ", "Andes_full_F18A33_win_jmr_MC_T0028_IZband_P0"),
}

# the IZ comb lines are ~100 px apart (R: ~57 px), so wider cells for enough lines
PARAM_OVERRIDES = {tag: {"elsf_xbin": 2048} for tag in ["IZ0", "IZ1", "IZ2"]}

# comparison figures: name -> tags ("UBVRIZ" uses the simulator's default
# model of each band)
GROUPS = {
    "R_models": ["R0", "R1", "R2"],
    "IZ_models": ["IZ0", "IZ1", "IZ2"],
    "UBVRIZ": ["U", "B", "V", "R0", "IZ0"],
}


def lfc_file(tag):
    band_dir, _hdf = MODELS[tag]
    return os.path.join(E2E_DIR, band_dir, f"{tag}_LFC_fiber{FIBER}_1s.fits")


def hdf_file(tag):
    return os.path.join(HDF_DIR, f"{MODELS[tag][1]}.hdf")


def params(tag):
    return {**VIS_PARAMS, **PARAM_OVERRIDES.get(tag, {})}


def plot_comparison(results, keys, title, out_png):
    """FWHM and R maps of several models on shared colour scales.

    keys: [(cell key, colorbar label, cmap, title value format), ...], one row each.
    """
    ext = 9232
    fig, axs = plt.subplots(
        len(keys), len(results), figsize=(5 * len(results), 4.75 * len(keys))
    )
    for row, (key, label, cmap, fmt) in enumerate(keys):
        allv = np.concatenate([[c[key] for c in cells] for cells in results.values()])
        vmin, vmax = np.nanpercentile(allv, [2, 98])
        for col, (tag, cells) in enumerate(results.items()):
            x = np.array([c["x"] for c in cells])
            y = np.array([c["y"] for c in cells])
            v = np.array([c[key] for c in cells])
            gx, gy = np.meshgrid(np.arange(0, ext, 64), np.arange(0, ext, 64))
            grid = griddata(np.column_stack([x, y]), v, (gx, gy), method="linear")
            ax = axs[row, col]
            im = ax.imshow(
                grid,
                origin="lower",
                extent=(0, ext, 0, ext),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.scatter(x, y, s=1, color="k", alpha=0.3)
            ax.set_title(f"{tag}: median " + fmt.format(np.nanmedian(v)))
            ax.set_xlabel("x [pixel]")
            ax.set_ylabel("y [pixel]")
        fig.colorbar(im, ax=axs[row, :], shrink=0.85, label=label)
    fig.suptitle(f"ANDES optical models, fiber {FIBER}, {title}")
    fig.savefig(out_png, dpi=110)
    print(f"Saved comparison: {out_png}")
    if not PLOT:
        plt.close(fig)


def run(tag):
    cells = process_band(
        tag,
        lfc_file=lfc_file(tag),
        hdf_file=hdf_file(tag),
        output_dir=os.path.join(OUT_DIR, f"{tag}_sampl"),
        prefix=f"andes_{tag.lower()}_fib{FIBER}",
        fiber=FIBER,
        **params(tag),
    )
    return tag, cells


if __name__ == "__main__":
    tags = sys.argv[1:] or list(MODELS)
    results = {}
    with multiprocessing.Pool(int(os.environ.get("NPROC", 6))) as pool:
        for tag, cells in pool.imap_unordered(run, tags):
            if cells:
                results[tag] = cells
    results = {t: results[t] for t in MODELS if t in results}

    print(f"\n{'=' * 60}\nSummary, fiber {FIBER} (medians over cells; p5-p95)")
    print(
        f"  {'model':<6} {'cells':>5}  {'FWHM eLSF':>17}  {'FWHM Gauss':>10}"
        f"  {'ratio':>5}  {'R eLSF':>15}  {'R Gauss':>7}"
    )
    for tag, cells in results.items():
        fw = np.array([c["fwhm"] for c in cells])
        gl = np.array([c["fwhm_gline"] for c in cells])
        r = np.array([c["R"] for c in cells])
        rg = np.array([c["R_gline"] for c in cells])
        p5, p50, p95 = np.nanpercentile(fw, [5, 50, 95])
        r5, r50, r95 = np.nanpercentile(r, [5, 50, 95]) / 1e3
        print(
            f"  {tag:<6} {len(cells):>5}  {p50:5.2f} ({p5:.2f}-{p95:.2f})"
            f"  {np.nanmedian(gl):>10.2f}  {np.nanmedian(fw / gl):5.3f}"
            f"  {r50:5.1f}k ({r5:.0f}-{r95:.0f}k)  {np.nanmedian(rg) / 1e3:6.1f}k"
        )

    for name, group in GROUPS.items():
        sub = {t: results[t] for t in group if t in results}
        if len(sub) < 2:
            continue
        plot_comparison(
            sub,
            [
                ("fwhm", "eLSF FWHM [pixel]", "viridis", "{:.2f} px"),
                ("R", "R (eLSF)", "inferno", "{:.0f}"),
            ],
            "empirical LSF",
            os.path.join(OUT_DIR, f"andes_{name}_fib{FIBER}_elsf.png"),
        )
        plot_comparison(
            sub,
            [
                ("fwhm_gline", "Gaussian FWHM [pixel]", "viridis", "{:.2f} px"),
                ("R_gline", "R (Gaussian)", "inferno", "{:.0f}"),
            ],
            "per-line Gaussian fits (median per cell)",
            os.path.join(OUT_DIR, f"andes_{name}_fib{FIBER}_gauss.png"),
        )

    if PLOT:
        plt.show()
