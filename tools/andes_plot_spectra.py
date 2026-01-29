# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "astropy"]
# ///
"""
Plot ANDES LFC spectra for all channels.

Loads extracted science files and wavelength solutions, plots groups A and B
for each channel (R0, R1, R2) to verify wavelength alignment.

Usage:
    uv run tools/andes_plot_spectra.py

    # Custom data directory:
    uv run tools/andes_plot_spectra.py --data-dir ~/REDUCE_DATA/ANDES/reduced

    # Single channel:
    uv run tools/andes_plot_spectra.py --channels R1

    # Save without showing:
    uv run tools/andes_plot_spectra.py --no-show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def plot_channel(ax, channel, data_dir):
    """Plot spectra for a single channel."""
    sci_file = data_dir / f"psf_comp_{channel}" / "lfc_combined.science.fits"
    wave_file = (
        data_dir / f"psf_comp_{channel}" / f"andes_riz_{channel.lower()}.thar.npz"
    )

    if not sci_file.exists():
        ax.text(0.5, 0.5, f"No data for {channel}", ha="center", va="center")
        ax.set_title(f"Channel {channel}")
        return

    # Load spectrum
    with fits.open(sci_file) as hdu:
        tbl = hdu[1].data[0]
        spec = tbl["SPEC"]
        cols = tbl["COLUMNS"]

    # Load wavelength
    wave_data = np.load(wave_file)
    wave = wave_data["wave"]
    ncol_wave = wave.shape[1]

    for i, label in enumerate(["A", "B"]):
        c0, c1 = cols[i]
        c1 = min(c1, ncol_wave)
        ax.plot(wave[i, c0:c1], spec[i, c0:c1], label=f"Group {label}", alpha=0.8)

    ax.set_ylabel("Counts")
    ax.set_title(f"Channel {channel}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot ANDES LFC spectra")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "REDUCE_DATA/ANDES/reduced",
        help="Directory containing reduced data",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["R0", "R1", "R2"],
        help="Channels to plot (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: data_dir/lfc_all_channels.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't show interactive plot",
    )
    args = parser.parse_args()

    n_channels = len(args.channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3.5 * n_channels), sharex=True)

    if n_channels == 1:
        axes = [axes]

    for ax, channel in zip(axes, args.channels, strict=True):
        plot_channel(ax, channel, args.data_dir)

    axes[-1].set_xlabel("Wavelength (nm)")
    plt.suptitle("ANDES LFC Spectra", fontsize=14)
    plt.tight_layout()

    output = args.output or args.data_dir / "lfc_all_channels.png"
    plt.savefig(output, dpi=150)
    print(f"Saved: {output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
