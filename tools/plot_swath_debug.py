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
    input_mask = data["input_mask"]
    output_mask = data["output_mask"]
    unc = data["unc"]
    info = data["info"]

    resid = swath_img - model
    rel_resid = resid / np.ma.masked_less(model, 1)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[1, 1, 1.2],
        width_ratios=[1, 1, 1, 0.8],
        hspace=0.08,
        wspace=0.25,
    )

    vmin, vmax = np.nanpercentile(swath_img, [5, 95])

    nrows = swath_img.shape[0]
    x_trace = np.arange(len(ycen))
    y_trace = nrows / 2 + ycen

    # Row 0: Image, Residual, Mask
    ax_swath = fig.add_subplot(gs[0, 0])
    ax_swath.imshow(swath_img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax_swath.plot(x_trace, y_trace, "r-", lw=1, alpha=0.7)
    ax_swath.set_title("Image")
    ax_swath.set_ylabel("y")
    ax_swath.tick_params(labelbottom=False)

    rlim = np.nanpercentile(np.abs(resid), 99)
    ax_resid = fig.add_subplot(gs[0, 1], sharex=ax_swath, sharey=ax_swath)
    ax_resid.imshow(
        resid, aspect="auto", origin="lower", cmap="bwr", vmin=-rlim, vmax=rlim
    )
    ax_resid.plot(x_trace, y_trace, "r-", lw=1, alpha=0.7)
    ax_resid.set_title("Residual")
    ax_resid.tick_params(labelbottom=False, labelleft=False)

    ax_mask = fig.add_subplot(gs[0, 2], sharex=ax_swath, sharey=ax_swath)
    # Show masks: white=input mask, red=newly rejected, black=good
    new_bad = output_mask & ~input_mask
    mask_rgb = np.zeros((*swath_img.shape, 3), dtype=np.float32)
    mask_rgb[input_mask, :] = [1, 1, 1]  # white
    mask_rgb[new_bad, :] = [1, 0, 0]  # red
    ax_mask.imshow(mask_rgb, aspect="auto", origin="lower", interpolation="nearest")
    ax_mask.set_title("Mask (white=input, red=new)")
    ax_mask.tick_params(labelbottom=False, labelleft=False)

    # Row 1: Model, Rel. residual, Uncertainty
    ax_model = fig.add_subplot(gs[1, 0], sharex=ax_swath, sharey=ax_swath)
    ax_model.imshow(model, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax_model.plot(x_trace, y_trace, "r-", lw=1, alpha=0.7)
    ax_model.set_title("Model")
    ax_model.set_ylabel("y")
    ax_model.tick_params(labelbottom=False)

    rlim_rel = np.percentile(np.ma.compressed(np.abs(rel_resid)), 99)
    ax_rel_resid = fig.add_subplot(gs[1, 1], sharex=ax_swath, sharey=ax_swath)
    ax_rel_resid.imshow(
        rel_resid,
        aspect="auto",
        origin="lower",
        cmap="bwr",
        vmin=-rlim_rel,
        vmax=rlim_rel,
    )
    ax_rel_resid.set_title("Rel. residual")
    ax_rel_resid.tick_params(labelbottom=False, labelleft=False)

    ax_unc = fig.add_subplot(gs[1, 2], sharex=ax_swath)
    ax_unc.plot(unc, "b-", lw=1)
    ax_unc.set_title("Uncertainty")
    ax_unc.set_ylabel("unc")
    ax_unc.tick_params(labelbottom=False)

    # Slit function panel (rightmost column, top 2 rows)
    ax_slit = fig.add_subplot(gs[0:2, 3])
    y_sl = np.linspace(0, nrows, len(slitf))
    ax_slit.plot(slitf, y_sl, "b-", lw=2)
    ax_slit.set_title("Slit function")
    ax_slit.set_xlabel("contribution")
    ax_slit.set_ylabel("y")
    ax_slit.yaxis.set_label_position("right")
    ax_slit.yaxis.tick_right()

    # Spectrum panel (bottom row)
    ax_spec = fig.add_subplot(gs[2, :3])
    x = np.arange(len(spec))
    ax_spec.plot(x, spec, "b-", lw=1.5, alpha=0.8)
    ax_spec.set_xlabel("x [pixel]")
    ax_spec.set_ylabel("flux")
    ax_spec.set_xlim(0, len(spec) - 1)
    ax_spec.set_title("Spectrum")

    # Info panel (bottom right)
    ax_info = fig.add_subplot(gs[2, 3])
    ax_info.axis("off")
    info_text = f"chi2: {info[1]:.3f}\niter: {int(info[3])}"
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
