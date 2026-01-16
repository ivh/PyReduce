#!/usr/bin/env python
# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Visualize 1D vs 2D extraction comparison from test_cwrappers.py output."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    reduce_data = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
    npz_file = Path(reduce_data) / "debug" / "1d_vs_2d_extraction.npz"

    if not npz_file.exists():
        print(f"File not found: {npz_file}")
        print(
            "Run: pytest test/test_cwrappers.py::test_1d_vs_2d_extraction --instrument=UVES"
        )
        return

    data = np.load(npz_file)

    swath_img = data["swath_img"]
    ycen = data["ycen"]
    sp_1d, sl_1d, model_1d, mask_1d = (
        data["sp_1d"],
        data["sl_1d"],
        data["model_1d"],
        data["mask_1d"],
    )
    sp_2d, sl_2d, model_2d, mask_2d = (
        data["sp_2d"],
        data["sl_2d"],
        data["model_2d"],
        data["mask_2d"],
    )

    resid_1d = (swath_img - model_1d) / np.maximum(model_1d, 1)
    resid_2d = (swath_img - model_2d) / np.maximum(model_2d, 1)

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

    # ycen trace position: center of swath + fractional offset
    nrows = swath_img.shape[0]
    x_trace = np.arange(len(ycen))
    y_trace = nrows / 2 + ycen

    # Row 0: Swath image and Models
    ax_swath = fig.add_subplot(gs[0, 0])
    ax_swath.imshow(swath_img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax_swath.plot(x_trace, y_trace, "r-", lw=1, alpha=0.7)
    ax_swath.set_title("Swath image")
    ax_swath.set_ylabel("y")
    ax_swath.tick_params(labelbottom=False)

    ax_model_1d = fig.add_subplot(gs[0, 1], sharey=ax_swath)
    ax_model_1d.imshow(model_1d, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax_model_1d.plot(x_trace, y_trace, "r-", lw=1, alpha=0.7)
    ax_model_1d.set_title("Model 1D")
    ax_model_1d.tick_params(labelbottom=False, labelleft=False)

    ax_model_2d = fig.add_subplot(gs[0, 2], sharex=ax_model_1d, sharey=ax_swath)
    ax_model_2d.imshow(model_2d, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax_model_2d.plot(x_trace, y_trace, "r-", lw=1, alpha=0.7)
    ax_model_2d.set_title("Model 2D")
    ax_model_2d.tick_params(labelbottom=False, labelleft=False)

    # Row 1: Relative residuals
    rlim = np.nanpercentile(np.abs(resid_1d), 99)
    ax_resid_1d = fig.add_subplot(gs[1, 1], sharex=ax_model_1d, sharey=ax_swath)
    ax_resid_1d.imshow(
        resid_1d, aspect="auto", origin="lower", cmap="bwr", vmin=-rlim, vmax=rlim
    )
    ax_resid_1d.set_title("Rel. resid 1D")
    ax_resid_1d.set_ylabel("y")
    ax_resid_1d.tick_params(labelbottom=False)

    ax_resid_2d = fig.add_subplot(gs[1, 2], sharex=ax_model_1d, sharey=ax_swath)
    ax_resid_2d.imshow(
        resid_2d, aspect="auto", origin="lower", cmap="bwr", vmin=-rlim, vmax=rlim
    )
    ax_resid_2d.set_title("Rel. resid 2D")
    ax_resid_2d.tick_params(labelbottom=False, labelleft=False)

    # Row 2: Masks
    ax_mask_1d = fig.add_subplot(gs[2, 1], sharex=ax_model_1d, sharey=ax_swath)
    ax_mask_1d.imshow(mask_1d, aspect="auto", origin="lower", cmap="gray")
    ax_mask_1d.set_title("Mask 1D")
    ax_mask_1d.set_ylabel("y")
    ax_mask_1d.set_xlabel("x")

    ax_mask_2d = fig.add_subplot(gs[2, 2], sharex=ax_model_1d, sharey=ax_swath)
    ax_mask_2d.imshow(mask_2d, aspect="auto", origin="lower", cmap="gray")
    ax_mask_2d.set_title("Mask 2D")
    ax_mask_2d.set_xlabel("x")
    ax_mask_2d.tick_params(labelleft=False)

    # Mask difference
    ax_mask_diff = fig.add_subplot(gs[2, 0], sharey=ax_swath)
    mask_diff = mask_1d.astype(int) - mask_2d.astype(int)
    ax_mask_diff.imshow(
        mask_diff, aspect="auto", origin="lower", cmap="bwr", vmin=-1, vmax=1
    )
    ax_mask_diff.set_title("Mask diff (1D-2D)")
    ax_mask_diff.set_ylabel("y")
    ax_mask_diff.set_xlabel("x")

    # Slit function panel (rightmost column, top 3 rows)
    ax_slit = fig.add_subplot(gs[0:3, 3])
    y_sl = np.linspace(0, nrows, len(sl_1d))
    ax_slit.plot(sl_1d, y_sl, "b-", lw=2, label="1D")
    ax_slit.plot(sl_2d, y_sl, "r--", lw=2, label="2D")
    ax_slit.set_title("Slit function")
    ax_slit.set_xlabel("contribution")
    ax_slit.set_ylabel("y")
    ax_slit.legend(loc="upper right")
    ax_slit.yaxis.set_label_position("right")
    ax_slit.yaxis.tick_right()

    # Spectrum panel (bottom row)
    ax_spec = fig.add_subplot(gs[3, :3])
    x = np.arange(len(sp_1d))
    ax_spec.plot(x, sp_1d, "b-", lw=1.5, label="1D", alpha=0.8)
    ax_spec.plot(x, sp_2d, "r--", lw=1.5, label="2D", alpha=0.8)
    ax_spec.set_xlabel("x [pixel]")
    ax_spec.set_ylabel("flux")
    ax_spec.set_xlim(0, len(sp_1d) - 1)
    ax_spec.legend(loc="upper right")
    ax_spec.set_title("Spectrum")

    # Spectrum difference panel
    ax_spec_diff = fig.add_subplot(gs[3, 3])
    rel_diff = (sp_1d - sp_2d) / np.maximum(sp_1d, 1) * 100
    ax_spec_diff.plot(x, rel_diff, "k-", lw=1)
    ax_spec_diff.axhline(0, color="gray", ls="--", lw=0.5)
    ax_spec_diff.set_xlabel("x [pixel]")
    ax_spec_diff.set_ylabel("diff [%]")
    ax_spec_diff.set_title("Rel. diff (1D-2D)")
    ax_spec_diff.yaxis.set_label_position("right")
    ax_spec_diff.yaxis.tick_right()

    fig.suptitle("1D vs 2D Extraction Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
