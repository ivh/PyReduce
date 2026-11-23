"""
Build the SOXS static files (initial linelists, order centers) from soxspipe products.

PyReduce does not ship a SOXS line atlas for the NIR pen lamps, and the VIS arm
has four independent quasi-orders, so instead of running wavecal_init we seed
the wavelength calibration with the lines soxspipe identified on the
single-pinhole arc (soxs_disp_solution, ``*_FITTED_LINES.fits`` QC table).

soxspipe works in its trimmed raw frame (VIS: columns 50:858, NIR: rows and
columns 4:2044). PyReduce transposes VIS and flips NIR upside down, so
    VIS: x_pyreduce = y_soxspipe, y_pyreduce = 807 - x_soxspipe
    NIR: x_pyreduce = x_soxspipe, y_pyreduce = 2039 - y_soxspipe
PyReduce sorts traces by order number descending, so the linelist row index of
order m is (m_max - m).

Wavelengths stay in air, in Angstrom, like the soxspipe solution.

Usage:
    uv run python tools/soxs_from_soxspipe.py /path/to/soxspipe/workspace
"""

import argparse
import glob
from os.path import dirname, join

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table

from pyreduce.wavelength_calibration import LineList

INST_DIR = join(dirname(__file__), "..", "pyreduce", "instruments", "SOXS")

ARMS = {
    # soxspipe DSOL QC table and the order numbers of the arm
    "vis": {"pattern": "*_VIS_1X1_1_DSOL_*FITTED_LINES.fits", "orders": [1, 2, 3, 4]},
    "nir": {"pattern": "*_NIR_*DSOL_*FITTED_LINES.fits", "orders": list(range(10, 25))},
}


def to_pyreduce(arm, xs, ys):
    if arm == "vis":
        return ys, 807 - xs
    return xs, 2039 - ys


def order_centers(arm, ws):
    img = sorted(
        glob.glob(
            join(
                ws, "reduced", "*", "soxs-spat-solution", f"*{arm.upper()}*_IMAGE.fits"
            )
        )
    )[0]
    with fits.open(img) as h:
        order = h["ORDER"].data
    ny, nx = order.shape
    centers = {}
    for m in ARMS[arm]["orders"]:
        if arm == "vis":
            # PyReduce x_center is row ny//2 in soxspipe, y is soxspipe x
            cols = np.nonzero(order[ny // 2] == m)[0]
            centers[m] = round(float(807 - cols.mean()), 1)
        else:
            rows = np.nonzero(order[:, nx // 2] == m)[0]
            centers[m] = round(float(2039 - rows.mean()), 1)
    return centers


def linelist(arm, ws):
    fname = sorted(
        glob.glob(join(ws, "qc", "*", "soxs-disp-solution", ARMS[arm]["pattern"]))
    )[0]
    t = Table.read(fname)
    t = t[~t["sigma_clipped"].astype(bool)]
    mmax = max(ARMS[arm]["orders"])

    xp, _ = to_pyreduce(arm, np.asarray(t["fit_x"]), np.asarray(t["fit_y"]))
    width = np.asarray(t["fwhm_pin_px"], dtype=float)
    width = np.where(np.isfinite(width) & (width > 0), width, 2.0)
    height = np.asarray(t["peak"], dtype=float)
    height = np.where(np.isfinite(height) & (height > 0), height, 1.0)

    lines = np.zeros(len(t), dtype=LineList.dtype)
    wave = np.asarray(t["wavelength"], dtype=float) * 10  # nm -> Angstrom
    lines["wlc"] = lines["wll"] = wave
    lines["posc"] = lines["posm"] = xp
    lines["xfirst"] = np.floor(xp - 2 * width).astype(int)
    lines["xlast"] = np.ceil(xp + 2 * width).astype(int)
    lines["approx"] = b"G"
    lines["width"] = width
    lines["height"] = height / np.nanmax(height)
    lines["order"] = mmax - np.asarray(t["order"]).astype(int)
    lines["bundle"] = -1
    lines["flag"] = True
    return LineList(lines), fname


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "workspace", help="soxspipe workspace after `soxspipe reduce all`"
    )
    args = parser.parse_args()

    for arm in ARMS:
        centers = order_centers(arm, args.workspace)
        out = join(INST_DIR, f"order_centers_{arm}.yaml")
        with open(out, "w") as f:
            f.write(
                f"# SOXS {arm.upper()}: y of the slit centre at the detector centre column\n"
                "# (PyReduce frame), from the soxspipe spatial-solution ORDER image\n"
            )
            yaml.safe_dump(centers, f, sort_keys=True)
        print(f"{out}: {centers}")

        ll, src = linelist(arm, args.workspace)
        out = join(INST_DIR, f"wavecal_{arm}.npz")
        ll.save(out)
        print(f"{out}: {len(ll)} lines from {src}")


if __name__ == "__main__":
    main()
