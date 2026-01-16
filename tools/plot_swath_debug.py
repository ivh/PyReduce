#!/usr/bin/env python
# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Visualize swath extraction debug output from ProgressPlot."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_swath_debug.py <npz_file>")
        print(
            "Example: plot_swath_debug.py ~/REDUCE_DATA/debug/swath_trace1_swath0.npz"
        )
        sys.exit(1)

    npz_file = Path(sys.argv[1])
    if not npz_file.exists():
        print(f"File not found: {npz_file}")
        sys.exit(1)

    data = np.load(npz_file, allow_pickle=True)

    swath_img = data["swath_img"]
    ycen = data["ycen"]
    spec = data["spec"]
    slitf = data["slitf"]
    model = data["model"]
    mask = data["mask"]
    unc = data["unc"]
    info = data["info"]

    resid = swath_img - model
    rel_resid = resid / np.maximum(model, 1)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[1, 1, 1, 1.2],
        width_ratios=[1, 1, 1, 0.8],
        hspace=0.08,
        wspace=0.25,
    )

    vmin, vmax = np.nanpercentile(swath_img, [5, 95])

    nrows = swath_img.shape[0]
    x_trace = np.arange(len(ycen))
    y_trace = nrows / 2 + ycen

    # Row 0: Swath image, Model, Residual
    ax_swath = fig.add_subplot(gs[0, 0])
    ax_swath.imshow(swath_img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax_swath.plot(x_trace, y_trace, "r-", lw=1, alpha=0.7)
    ax_swath.set_title("Swath image")
    ax_swath.set_ylabel("y")
    ax_swath.tick_params(labelbottom=False)

    ax_model = fig.add_subplot(gs[0, 1], sharey=ax_swath)
    ax_model.imshow(model, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax_model.plot(x_trace, y_trace, "r-", lw=1, alpha=0.7)
    ax_model.set_title("Model")
    ax_model.tick_params(labelbottom=False, labelleft=False)

    ax_resid = fig.add_subplot(gs[0, 2], sharex=ax_model, sharey=ax_swath)
    rlim = np.nanpercentile(np.abs(resid), 99)
    ax_resid.imshow(
        resid, aspect="auto", origin="lower", cmap="bwr", vmin=-rlim, vmax=rlim
    )
    ax_resid.plot(x_trace, y_trace, "r-", lw=1, alpha=0.7)
    ax_resid.set_title("Residual")
    ax_resid.tick_params(labelbottom=False, labelleft=False)

    # Row 1: Relative residual and Mask
    ax_rel_resid = fig.add_subplot(gs[1, 0], sharey=ax_swath)
    rlim_rel = np.nanpercentile(np.abs(rel_resid), 99)
    ax_rel_resid.imshow(
        rel_resid,
        aspect="auto",
        origin="lower",
        cmap="bwr",
        vmin=-rlim_rel,
        vmax=rlim_rel,
    )
    ax_rel_resid.set_title("Rel. residual")
    ax_rel_resid.set_ylabel("y")
    ax_rel_resid.tick_params(labelbottom=False)

    ax_mask = fig.add_subplot(gs[1, 1], sharex=ax_model, sharey=ax_swath)
    ax_mask.imshow(mask, aspect="auto", origin="lower", cmap="gray")
    ax_mask.set_title("Mask (white=good)")
    ax_mask.tick_params(labelbottom=False, labelleft=False)

    # Row 2: Uncertainty (1D per column)
    ax_unc = fig.add_subplot(gs[2, 0])
    ax_unc.plot(unc, "b-", lw=1)
    ax_unc.set_title("Uncertainty")
    ax_unc.set_ylabel("unc")
    ax_unc.set_xlabel("x")

    # Slit function panel (rightmost column, top 3 rows)
    ax_slit = fig.add_subplot(gs[0:3, 3])
    y_sl = np.linspace(0, nrows, len(slitf))
    ax_slit.plot(slitf, y_sl, "b-", lw=2)
    ax_slit.set_title("Slit function")
    ax_slit.set_xlabel("contribution")
    ax_slit.set_ylabel("y")
    ax_slit.yaxis.set_label_position("right")
    ax_slit.yaxis.tick_right()

    # Spectrum panel (bottom row)
    ax_spec = fig.add_subplot(gs[3, :3])
    x = np.arange(len(spec))
    ax_spec.plot(x, spec, "b-", lw=1.5, alpha=0.8)
    ax_spec.set_xlabel("x [pixel]")
    ax_spec.set_ylabel("flux")
    ax_spec.set_xlim(0, len(spec) - 1)
    ax_spec.set_title("Spectrum")

    # Info panel (bottom right)
    ax_info = fig.add_subplot(gs[3, 3])
    ax_info.axis("off")
    info_text = f"chi2: {info[1]:.3f}\niter: {int(info[0])}"
    ax_info.text(
        0.1,
        0.5,
        info_text,
        transform=ax_info.transAxes,
        fontsize=12,
        verticalalignment="center",
    )
    ax_info.set_title("Info")

    fig.suptitle(f"Swath Debug: {npz_file.name}", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
