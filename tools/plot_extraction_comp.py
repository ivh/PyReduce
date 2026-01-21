#!/usr/bin/env python
# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Visualize extraction comparison from test output npz files.

Supports:
- 1d_vs_2d_extraction.npz (from test_cwrappers.py::test_1d_vs_2d_extraction)
- numba_vs_c_extraction.npz (from test_numbaextract.py::test_numba_vs_c_extraction)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_comparison_data(npz_file):
    """Load npz file and return normalized data dict with keys: sp_a, sp_b, etc."""
    data = np.load(npz_file)
    keys = list(data.keys())

    # Detect file type and map to generic names
    if "sp_1d" in keys:
        # 1D vs 2D comparison
        return {
            "swath_img": data["swath_img"],
            "ycen": data["ycen"],
            "sp_a": data["sp_1d"],
            "sl_a": data["sl_1d"],
            "model_a": data["model_1d"],
            "mask_a": data["mask_1d"],
            "sp_b": data["sp_2d"],
            "sl_b": data["sl_2d"],
            "model_b": data["model_2d"],
            "mask_b": data["mask_2d"],
            "label_a": "1D",
            "label_b": "2D",
            "title": "1D vs 2D Extraction Comparison",
        }
    elif "sp_c" in keys:
        # C vs Numba comparison
        return {
            "swath_img": data["swath_img"],
            "ycen": data["ycen"],
            "sp_a": data["sp_c"],
            "sl_a": data["sl_c"],
            "model_a": data["model_c"],
            "mask_a": data["mask_c"],
            "sp_b": data["sp_numba"],
            "sl_b": data["sl_numba"],
            "model_b": data["model_numba"],
            "mask_b": data["mask_numba"],
            "label_a": "C",
            "label_b": "Numba",
            "title": "C vs Numba Extraction Comparison",
        }
    else:
        raise ValueError(f"Unknown npz format. Keys: {keys}")


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_extraction_comp.py <npz_file>")
        print()
        print("Example npz files (in $REDUCE_DATA/debug/):")
        print(
            "  1d_vs_2d_extraction.npz   - from test_cwrappers.py::test_1d_vs_2d_extraction"
        )
        print(
            "  numba_vs_c_extraction.npz - from test_numbaextract.py::test_numba_vs_c_extraction"
        )
        return

    npz_file = Path(sys.argv[1])
    if not npz_file.exists():
        print(f"File not found: {npz_file}")
        return

    d = load_comparison_data(npz_file)

    swath_img = d["swath_img"]
    ycen = d["ycen"]
    sp_a, sl_a, model_a, mask_a = d["sp_a"], d["sl_a"], d["model_a"], d["mask_a"]
    sp_b, sl_b, model_b, mask_b = d["sp_b"], d["sl_b"], d["model_b"], d["mask_b"]
    label_a, label_b = d["label_a"], d["label_b"]

    resid_a = (swath_img - model_a) / np.maximum(model_a, 1)
    resid_b = (swath_img - model_b) / np.maximum(model_b, 1)

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

    ax_model_a = fig.add_subplot(gs[0, 1], sharey=ax_swath)
    ax_model_a.imshow(model_a, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax_model_a.plot(x_trace, y_trace, "r-", lw=1, alpha=0.7)
    ax_model_a.set_title(f"Model {label_a}")
    ax_model_a.tick_params(labelbottom=False, labelleft=False)

    ax_model_b = fig.add_subplot(gs[0, 2], sharex=ax_model_a, sharey=ax_swath)
    ax_model_b.imshow(model_b, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax_model_b.plot(x_trace, y_trace, "r-", lw=1, alpha=0.7)
    ax_model_b.set_title(f"Model {label_b}")
    ax_model_b.tick_params(labelbottom=False, labelleft=False)

    # Row 1: Relative residuals
    rlim = np.nanpercentile(np.abs(resid_a), 99)
    ax_resid_a = fig.add_subplot(gs[1, 1], sharex=ax_model_a, sharey=ax_swath)
    ax_resid_a.imshow(
        resid_a, aspect="auto", origin="lower", cmap="bwr", vmin=-rlim, vmax=rlim
    )
    ax_resid_a.set_title(f"Rel. resid {label_a}")
    ax_resid_a.set_ylabel("y")
    ax_resid_a.tick_params(labelbottom=False)

    ax_resid_b = fig.add_subplot(gs[1, 2], sharex=ax_model_a, sharey=ax_swath)
    ax_resid_b.imshow(
        resid_b, aspect="auto", origin="lower", cmap="bwr", vmin=-rlim, vmax=rlim
    )
    ax_resid_b.set_title(f"Rel. resid {label_b}")
    ax_resid_b.tick_params(labelbottom=False, labelleft=False)

    # Row 2: Masks
    ax_mask_a = fig.add_subplot(gs[2, 1], sharex=ax_model_a, sharey=ax_swath)
    ax_mask_a.imshow(mask_a, aspect="auto", origin="lower", cmap="gray")
    ax_mask_a.set_title(f"Mask {label_a}")
    ax_mask_a.set_ylabel("y")
    ax_mask_a.set_xlabel("x")

    ax_mask_b = fig.add_subplot(gs[2, 2], sharex=ax_model_a, sharey=ax_swath)
    ax_mask_b.imshow(mask_b, aspect="auto", origin="lower", cmap="gray")
    ax_mask_b.set_title(f"Mask {label_b}")
    ax_mask_b.set_xlabel("x")
    ax_mask_b.tick_params(labelleft=False)

    # Mask difference
    ax_mask_diff = fig.add_subplot(gs[2, 0], sharey=ax_swath)
    mask_diff = mask_a.astype(int) - mask_b.astype(int)
    ax_mask_diff.imshow(
        mask_diff, aspect="auto", origin="lower", cmap="bwr", vmin=-1, vmax=1
    )
    ax_mask_diff.set_title(f"Mask diff ({label_a}-{label_b})")
    ax_mask_diff.set_ylabel("y")
    ax_mask_diff.set_xlabel("x")

    # Slit function panel (rightmost column, top 3 rows)
    ax_slit = fig.add_subplot(gs[0:3, 3])
    y_sl = np.linspace(0, nrows, len(sl_a))
    ax_slit.plot(sl_a, y_sl, "b-", lw=2, label=label_a)
    ax_slit.plot(sl_b, y_sl, "r--", lw=2, label=label_b)
    ax_slit.set_title("Slit function")
    ax_slit.set_xlabel("contribution")
    ax_slit.set_ylabel("y")
    ax_slit.legend(loc="upper right")
    ax_slit.yaxis.set_label_position("right")
    ax_slit.yaxis.tick_right()

    # Spectrum panel (bottom row)
    ax_spec = fig.add_subplot(gs[3, :3])
    x = np.arange(len(sp_a))
    ax_spec.plot(x, sp_a, "b-", lw=1.5, label=label_a, alpha=0.8)
    ax_spec.plot(x, sp_b, "r--", lw=1.5, label=label_b, alpha=0.8)
    ax_spec.set_xlabel("x [pixel]")
    ax_spec.set_ylabel("flux")
    ax_spec.set_xlim(0, len(sp_a) - 1)
    ax_spec.legend(loc="upper right")
    ax_spec.set_title("Spectrum")

    # Spectrum difference panel
    ax_spec_diff = fig.add_subplot(gs[3, 3])
    rel_diff = (sp_a - sp_b) / np.maximum(sp_a, 1) * 100
    ax_spec_diff.plot(x, rel_diff, "k-", lw=1)
    ax_spec_diff.axhline(0, color="gray", ls="--", lw=0.5)
    ax_spec_diff.set_xlabel("x [pixel]")
    ax_spec_diff.set_ylabel("diff [%]")
    ax_spec_diff.set_title(f"Rel. diff ({label_a}-{label_b})")
    ax_spec_diff.yaxis.set_label_position("right")
    ax_spec_diff.yaxis.tick_right()

    fig.suptitle(d["title"], fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
