"""Add wavelength polynomials from ANDES HDF optical models to a trace FITS file.

Reads the HDF file's per-fiber-per-order (translation_x, wavelength) samples,
fits a 5th-degree polynomial wl(x) for each trace (matching by order number and
fiber index or group center fiber), and writes the coefficients into the WAVE
column of the trace FITS file.

Usage:
    uv run python tools/hdf2trace.py path/to/model.hdf path/to/trace.fits
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "astropy",
#     "h5py",
# ]
# ///

import argparse
import sys

import h5py
import numpy as np
from astropy.io import fits

POLY_DEGREE = 5

# Center fiber for each group (1-indexed, for 66-fiber instruments)
GROUP_CENTER_FIBER_66 = {"A": 16, "B": 51, "cal": 33}
GROUP_CENTER_FIBER_75 = {"A": 18, "B": 56, "cal": 38}


def read_hdf_wavelengths(hdf_path):
    """Read (x, wavelength) samples for every fiber and order from HDF.

    Returns
    -------
    dict[(fiber_num, order_m)] -> (x_array, wl_nm_array)
    """
    data = {}
    with h5py.File(hdf_path, "r") as f:
        ccd = f["CCD_1"]
        for fkey in ccd:
            if not fkey.startswith("fiber_"):
                continue
            fib_num = int(fkey.replace("fiber_", ""))
            fiber_grp = ccd[fkey]
            for okey in fiber_grp:
                if not okey.startswith("order"):
                    continue
                m = int(okey.replace("order", ""))
                samples = fiber_grp[okey][:]
                x = np.asarray(samples["translation_x"], dtype=np.float64)
                wl_nm = np.asarray(samples["wavelength"], dtype=np.float64) * 1000.0
                data[(fib_num, m)] = (x, wl_nm)
    return data


def fit_wave_poly(x, wl, degree=POLY_DEGREE):
    """Fit wl(x) polynomial. Returns coefficients in np.polyval order (high to low)."""
    return np.polyfit(x, wl, degree)


def main():
    parser = argparse.ArgumentParser(description="Add HDF wavelengths to trace FITS")
    parser.add_argument("hdf", help="ANDES HDF optical model file")
    parser.add_argument("trace", help="Trace FITS file to update (modified in-place)")
    parser.add_argument(
        "--degree",
        type=int,
        default=POLY_DEGREE,
        help=f"Polynomial degree (default {POLY_DEGREE})",
    )
    args = parser.parse_args()

    hdf_data = read_hdf_wavelengths(args.hdf)
    if not hdf_data:
        print(f"No fiber/order data found in {args.hdf}", file=sys.stderr)
        sys.exit(1)

    n_fibers = max(fib for fib, _ in hdf_data)
    group_center = GROUP_CENTER_FIBER_66 if n_fibers <= 66 else GROUP_CENTER_FIBER_75
    hdf_orders = sorted({m for _, m in hdf_data})
    print(f"HDF: {n_fibers} fibers, orders {hdf_orders[0]}..{hdf_orders[-1]}")

    f = fits.open(args.trace)
    tab = f[1].data
    header = f[1].header
    ntrace = len(tab)

    wave_deg = args.degree + 1  # number of coefficients
    wave_arr = np.full((ntrace, wave_deg), np.nan, dtype=np.float64)

    matched = 0
    skipped = 0
    residuals = []

    for i, row in enumerate(tab):
        m = int(row["M"])
        grp = row["GROUP"].strip()
        fib_idx = int(row["FIBER_IDX"])

        # Determine which HDF fiber to use
        if fib_idx > 0:
            hdf_fiber = fib_idx
        elif grp in group_center:
            hdf_fiber = group_center[grp]
        else:
            skipped += 1
            continue

        key = (hdf_fiber, m)
        if key not in hdf_data:
            skipped += 1
            continue

        x, wl = hdf_data[key]
        coef = fit_wave_poly(x, wl, degree=args.degree)
        wave_arr[i, : len(coef)] = coef

        # Compute residual for diagnostics
        resid = np.max(np.abs(np.polyval(coef, x) - wl))
        residuals.append(resid)
        matched += 1

    print(f"Matched {matched}/{ntrace} traces ({skipped} skipped)")
    if residuals:
        print(
            f"Fit residuals (nm): median={np.median(residuals):.4e}, "
            f"max={np.max(residuals):.4e}"
        )

    # Remove old WAVE column if present, then add new one
    colnames = [c.name for c in tab.columns]
    if "WAVE" in colnames:
        tab = fits.BinTableHDU.from_columns(
            [c for c in tab.columns if c.name != "WAVE"], header=header
        ).data

    new_col = fits.Column(name="WAVE", format=f"{wave_deg}D", array=wave_arr)
    new_tab = fits.BinTableHDU.from_columns(
        list(tab.columns) + [new_col], header=header
    )
    # Clear any 2D wave header keys (we're writing 1D polys)
    for key in ("WAVE_X", "WAVE_M"):
        if key in new_tab.header:
            del new_tab.header[key]

    f[1] = new_tab
    f.writeto(args.trace, overwrite=True, output_verify="silentfix+ignore")
    f.close()
    print(f"Updated {args.trace}")


if __name__ == "__main__":
    main()
