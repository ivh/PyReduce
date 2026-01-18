#!/usr/bin/env python
# /// script
# dependencies = ["pyreduce-astro"]
# ///
"""Visualize swath extraction debug output from ProgressPlot."""

import sys
from pathlib import Path

import numpy as np

from pyreduce.extract import ProgressPlot


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

    nrow, ncol = swath_img.shape
    nslitf = len(slitf)

    plot = ProgressPlot(nrow, ncol, nslitf, title=f"Debug: {npz_file.name}")
    plot.plot(
        swath_img,
        spec,
        slitf,
        model,
        ycen,
        input_mask,
        output_mask,
        unc=unc,
        info=info,
        save=False,
    )

    import matplotlib.pyplot as plt

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
