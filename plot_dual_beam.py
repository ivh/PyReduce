#!/usr/bin/env python
"""Plot upper and lower beam spectra from a HARPSPOL science file.

Usage:
    python plot_dual_beam.py <science_file> [--order N] [--all]

Examples:
    python plot_dual_beam.py HARPS.2012-07-15T07:05:34.422.science.BLUE.fits --order 20
    python plot_dual_beam.py HARPS.2012-07-15T07:05:34.422.science.BLUE.fits --all
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def load_spectra(filename):
    """Load upper and lower beam spectra from a science FITS file."""
    hdul = fits.open(filename)
    t = hdul["SPECTRA"].data

    upper = {}
    lower = {}
    for row in t:
        m = int(row["M"])
        group = row["GROUP"].strip()
        spec = row["SPEC"].copy()
        wave = row["WAVE"].copy()
        sig = row["SIG"].copy()

        entry = {"spec": spec, "wave": wave, "sig": sig, "m": m}
        if group == "upper":
            upper[m] = entry
        else:
            lower[m] = entry

    hdul.close()
    return upper, lower


def plot_order(upper, lower, order, ax=None):
    """Plot upper and lower beam for a single order."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 5))

    has_wave = False
    for label, data, color in [
        ("upper", upper.get(order), "C0"),
        ("lower", lower.get(order), "C1"),
    ]:
        if data is None:
            continue
        wave = data["wave"]
        spec = data["spec"]
        if np.any(wave > 0):
            has_wave = True
        x = wave if has_wave else np.arange(len(spec))
        ax.plot(x, spec, color=color, alpha=0.7, lw=0.8)

    ax.set_xlabel("Wavelength [A]" if has_wave else "Pixel")
    ax.set_ylabel("Flux")
    ax.set_title(f"Order {order}")

    if standalone:
        plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Plot HARPSPOL dual-beam spectra")
    parser.add_argument("filename", help="Science FITS file")
    parser.add_argument("--order", type=int, default=None, help="Plot a single order")
    parser.add_argument("--all", action="store_true", help="Plot all orders in a grid")
    args = parser.parse_args()

    upper, lower = load_spectra(args.filename)
    all_orders = sorted(set(upper) | set(lower))

    if not all_orders:
        print("No spectra found.", file=sys.stderr)
        sys.exit(1)

    if args.order is not None:
        if args.order not in all_orders:
            print(f"Order {args.order} not found. Available: {all_orders}", file=sys.stderr)
            sys.exit(1)
        plot_order(upper, lower, args.order)
    elif args.all:
        ncols = 3
        nrows = (len(all_orders) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
        axes = axes.flatten()
        for i, order in enumerate(all_orders):
            plot_order(upper, lower, order, ax=axes[i])
        for i in range(len(all_orders), len(axes)):
            axes[i].set_visible(False)
        fig.suptitle(args.filename, fontsize=10)
        plt.tight_layout()
    else:
        # Default: pick a middle order with wavelength data
        mid = all_orders[len(all_orders) // 2]
        for m in all_orders:
            if m in upper and np.any(upper[m]["wave"] > 0):
                mid = m
                break
        plot_order(upper, lower, mid)
        print(f"Showing order {mid}. Use --order N or --all for other orders.")

    plt.show()


if __name__ == "__main__":
    main()
