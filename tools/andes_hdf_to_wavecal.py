# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py", "numpy"]
# ///
"""
Generate PyReduce wavecal files from ANDES E2E HDF optical models.

Two output modes:
1. reference: Generate wavecal_*_HDF.npz reference files for wavecal_init step
2. thar: Generate .thar.npz output files that can be loaded directly by wavecal step

Usage:
    # Generate reference files for all channels
    uv run tools/andes_hdf_to_wavecal.py reference

    # Generate .thar.npz for a specific reduction (requires trace file)
    uv run tools/andes_hdf_to_wavecal.py thar --channel R1 \\
        --trace-file ~/REDUCE_DATA/ANDES/reduced/psf_comp_R1/andes_riz_r1.traces.npz \\
        --out-file ~/REDUCE_DATA/ANDES/reduced/psf_comp_R1/andes_riz_r1.thar.npz
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

# Channel to HDF model file mapping
MODELS = {
    "R0": "ANDES_123_R3.hdf",
    "R1": "ANDES_full_F18A33_win_jmr_MC_T0019_Rband_p0.hdf",
    "R2": "Andes_full_F18A33_win_jmr_MC_T0108_Rband_P0_cfg1.hdf",
}

# Fiber groups for wavecal (group name -> middle fiber number)
# A: fibers 1-32, middle = 16
# B: fibers 36-67, middle = 51
WAVECAL_GROUPS = {"A": 16, "B": 51}

# Groups used for wavecal (order matters for trace assignment)
USE_GROUPS = ["A", "B"]

# Detector size
NCOL = 9216

# Polynomial degrees for 2D fit
DEG_ORDER = 4
DEG_X = 4


def fit_2d_polynomial(orders, x_positions, wavelengths, deg_o, deg_x, ncol):
    """Fit 2D polynomial: wavelength = f(order, x)."""
    order_min, order_max = orders.min(), orders.max()
    x_min, x_max = 0, ncol

    # Normalize to [-1, 1]
    o_norm = (orders - order_min) / (order_max - order_min) * 2 - 1
    x_norm = (x_positions - x_min) / (x_max - x_min) * 2 - 1

    # Build Vandermonde matrix
    terms = []
    for i in range(deg_o + 1):
        for j in range(deg_x + 1):
            terms.append(o_norm**i * x_norm**j)
    V = np.column_stack(terms)

    # Solve least squares
    coef, _, _, _ = np.linalg.lstsq(V, wavelengths, rcond=None)
    rms = np.sqrt(np.mean((V @ coef - wavelengths) ** 2))

    return coef, rms, order_min, order_max, x_min, x_max


def eval_2d_polynomial(coef, order_num, ncol, deg_o, deg_x, order_min, order_max):
    """Evaluate 2D polynomial for a single order across all columns."""
    x = np.arange(ncol)
    x_norm = x / ncol * 2 - 1  # Normalize to [-1, 1]
    o_norm = (order_num - order_min) / (order_max - order_min) * 2 - 1

    wl = np.zeros(ncol)
    idx = 0
    for io in range(deg_o + 1):
        for ix in range(deg_x + 1):
            wl += coef[idx] * (o_norm**io) * (x_norm**ix)
            idx += 1
    return wl


def extract_group_coefs(hdf_path, channel):
    """Extract per-group wavelength coefficients from HDF file."""
    print(f"Processing {channel} from {hdf_path.name}...")

    with h5py.File(hdf_path, "r") as f:
        # Get all orders from center fiber
        fiber_group = f["CCD_1/fiber_33"]
        orders = sorted(
            int(k.replace("order", ""))
            for k in fiber_group.keys()
            if k.startswith("order")
        )
        print(f"  Orders: {orders[0]}-{orders[-1]}")

        # Build line list from center fiber
        cs_lines = []
        center_fiber = 33
        for order in orders:
            key = f"CCD_1/fiber_{center_fiber}/order{order}"
            data = f[key][:]
            for x, wl in zip(data["translation_x"], data["wavelength"], strict=False):
                if 0 <= x <= NCOL:
                    wl_nm = wl * 1000  # um to nm
                    cs_lines.append(
                        (wl_nm, wl_nm, x, x, 0, NCOL, None, 1.0, 1.0, order, True)
                    )

        dtype = [
            ("WLC", ">f8"),
            ("WLL", ">f8"),
            ("POSC", ">f8"),
            ("POSM", ">f8"),
            ("XFIRST", ">i2"),
            ("XLAST", ">i2"),
            ("APPROX", "O"),
            ("WIDTH", ">f8"),
            ("HEIGHT", ">f8"),
            ("ORDER", ">i2"),
            ("flag", "?"),
        ]
        cs_lines_arr = np.array(cs_lines, dtype=dtype)

        # Build per-group wavelength coefficients
        group_coefs = {}
        order_min, order_max = None, None
        for group_name, fiber in WAVECAL_GROUPS.items():
            all_orders = []
            all_x = []
            all_wl = []

            for order in orders:
                key = f"CCD_1/fiber_{fiber}/order{order}"
                if key not in f:
                    continue
                data = f[key][:]
                for x, wl in zip(
                    data["translation_x"], data["wavelength"], strict=False
                ):
                    if 0 <= x <= NCOL:
                        all_orders.append(order)
                        all_x.append(x)
                        all_wl.append(wl * 1000)

            all_orders = np.array(all_orders)
            all_x = np.array(all_x)
            all_wl = np.array(all_wl)

            coef, rms, omin, omax, _, _ = fit_2d_polynomial(
                all_orders, all_x, all_wl, DEG_ORDER, DEG_X, NCOL
            )
            group_coefs[group_name] = coef
            order_min, order_max = omin, omax
            print(f"  Group {group_name} (fiber {fiber}): RMS = {rms:.4f} nm")

    return {
        "cs_lines": cs_lines_arr,
        "orders": orders,
        "order_min": order_min,
        "order_max": order_max,
        "group_coefs": group_coefs,
    }


def cmd_reference(args):
    """Generate wavecal_*_HDF.npz reference files."""
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for channel in args.channels:
        if channel not in MODELS:
            print(f"Unknown channel: {channel}, skipping")
            continue

        hdf_path = args.hdf_dir / MODELS[channel]
        if not hdf_path.exists():
            print(f"HDF file not found: {hdf_path}, skipping {channel}")
            continue

        data = extract_group_coefs(hdf_path, channel)

        out_data = {
            "cs_lines": data["cs_lines"],
            "obase": np.int16(data["orders"][0]),
            "oincr": np.int16(1),
            "bad_order": np.zeros(len(data["orders"]), dtype=np.uint8),
            "deg_o": DEG_ORDER,
            "deg_x": DEG_X,
            "order_min": data["order_min"],
            "order_max": data["order_max"],
            "x_min": 0.0,
            "x_max": float(NCOL),
            "group_names": list(WAVECAL_GROUPS.keys()),
            "group_fibers": list(WAVECAL_GROUPS.values()),
        }
        for name, coef in data["group_coefs"].items():
            out_data[f"coef_{name}"] = coef

        out_path = args.out_dir / f"wavecal_{channel.lower()}_HDF.npz"
        np.savez(out_path, **out_data)
        print(f"  Saved: {out_path}")


def cmd_thar(args):
    """Generate .thar.npz file from HDF and trace file."""
    if args.channel not in MODELS:
        raise ValueError(f"Unknown channel: {args.channel}")

    hdf_path = args.hdf_dir / MODELS[args.channel]
    if not hdf_path.exists():
        raise FileNotFoundError(f"HDF file not found: {hdf_path}")

    if not args.trace_file.exists():
        raise FileNotFoundError(f"Trace file not found: {args.trace_file}")

    # Load HDF data
    data = extract_group_coefs(hdf_path, args.channel)
    group_coefs = data["group_coefs"]
    order_min = data["order_min"]
    order_max = data["order_max"]

    # Load trace file to get order/group assignments
    trace_data = np.load(args.trace_file, allow_pickle=True)
    group_names = trace_data["group_names"].tolist()

    # Build trace info: which order and group each trace belongs to
    trace_info = []  # List of (order_num, group_name)
    for group_name in USE_GROUPS:
        if group_name not in group_names:
            continue
        group_traces = trace_data[f"group_{group_name}_traces"].item()
        for order_num in sorted(group_traces.keys()):
            traces_in_order = group_traces[order_num]
            ntrace = traces_in_order.shape[0] if traces_in_order.ndim == 2 else 1
            for _ in range(ntrace):
                trace_info.append((int(order_num), group_name))

    if not trace_info:
        raise ValueError("No traces found for wavecal groups")

    print(f"  Generating wavelengths for {len(trace_info)} traces...")

    # Generate wave array
    wave = np.zeros((len(trace_info), NCOL))
    for i, (order_num, group_name) in enumerate(trace_info):
        coef = group_coefs[group_name]
        wave[i] = eval_2d_polynomial(
            coef, order_num, NCOL, DEG_ORDER, DEG_X, order_min, order_max
        )

    # Use first group's coef as the saved coef (for compatibility)
    coef = next(iter(group_coefs.values()))

    # Save in .thar.npz format
    np.savez(
        args.out_file,
        wave=wave,
        coef=coef,
        linelist=data["cs_lines"],
    )
    print(f"  Saved: {args.out_file}")
    print(f"  Wavelength range: {wave[0, 0]:.2f} - {wave[0, -1]:.2f} nm")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyReduce wavecal files from ANDES HDF models"
    )
    parser.add_argument(
        "--hdf-dir",
        type=Path,
        default=Path.home() / "ANDES/E2E/src/HDF",
        help="Directory containing HDF model files",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # reference subcommand
    ref_parser = subparsers.add_parser(
        "reference", help="Generate wavecal_*_HDF.npz reference files"
    )
    ref_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: pyreduce/instruments/ANDES_RIZ/)",
    )
    ref_parser.add_argument(
        "--channels",
        nargs="+",
        default=list(MODELS.keys()),
        help="Channels to process (default: all)",
    )

    # thar subcommand
    thar_parser = subparsers.add_parser(
        "thar", help="Generate .thar.npz output file from trace file"
    )
    thar_parser.add_argument(
        "--channel", required=True, choices=list(MODELS.keys()), help="Channel to use"
    )
    thar_parser.add_argument(
        "--trace-file", type=Path, required=True, help="Path to trace .npz file"
    )
    thar_parser.add_argument(
        "--out-file", type=Path, required=True, help="Output .thar.npz file path"
    )

    args = parser.parse_args()

    if args.command == "reference":
        if args.out_dir is None:
            script_dir = Path(__file__).parent.parent
            args.out_dir = script_dir / "pyreduce/instruments/ANDES_RIZ"
        cmd_reference(args)
    elif args.command == "thar":
        cmd_thar(args)


if __name__ == "__main__":
    main()
