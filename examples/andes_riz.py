# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7"]
# ///
"""
ANDES_RIZ instrument example: Step-by-step pipeline execution.

This example demonstrates running individual pipeline steps for the ANDES R-band
instrument with simulated PSF comparison data. Files are distinguished by:
- HDFMODEL header: selects channel (R0, R1, R2 = different optical models)
- SIMTYPE header: file type (flat_field, lfc, spectrum)
- FIBMODE header: even/odd fiber illumination pattern

Usage:
    PYREDUCE_PLOT=1 uv run python examples/andes_riz.py

Or import and run steps interactively:
    from examples.andes_riz import pipe, flat_files, lfc_files
    pipe.trace(flat_files)
    pipe.run()
"""

import os
from os.path import join

from pyreduce import util
from pyreduce.configuration import load_config
from pyreduce.instruments import instrument_info
from pyreduce.pipeline import Pipeline

# --- Configuration ---
instrument_name = "ANDES_RIZ"
target = "psf_comp"
night = ""
channel = "R0"  # R0, R1, or R2 (different optical models)

# Plot settings
plot = int(os.environ.get("PYREDUCE_PLOT", "1"))
plot_dir = os.environ.get("PYREDUCE_PLOT_DIR")
if plot_dir:
    util.set_plot_dir(plot_dir)

# Data location
data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
raw_dir = join(data_dir, "ANDES", "psf_comp_R")
output_dir = join(data_dir, "ANDES", "reduced", f"psf_comp_{channel}")

# --- Discover files ---
print(f"Discovering files in {raw_dir}...")
sorted_files = instrument_info.sort_files(
    input_dir=raw_dir,
    target=target,
    night=night,
    instrument=instrument_name,
    channel=channel,
)

if not sorted_files:
    raise RuntimeError(f"No files found for channel {channel}")

# Extract file lists from sorted results
setting, files = sorted_files[0]
flat_files = list(files.get("flat", []))
trace_files = list(files.get("trace", []))
wavecal_files = list(files.get("wavecal_master", []))

print(f"Channel: {channel} ({setting['channel']})")
print(f"Flat files: {[os.path.basename(f) for f in flat_files]}")
print(f"Trace files: {[os.path.basename(f) for f in trace_files]}")
print(f"Wavecal files: {[os.path.basename(f) for f in wavecal_files]}")

# --- Create Pipeline ---
config = load_config(None, instrument_name, channel=channel)
pipe = Pipeline(
    instrument=instrument_name,
    output_dir=output_dir,
    target=target,
    channel=channel,
    night=night,
    config=config,
    plot=plot,
)

print(f"\nInstrument: {pipe.instrument.name}")
print(f"Output: {output_dir}")

# Show fiber config if present
if pipe.instrument.config.fibers:
    fc = pipe.instrument.config.fibers
    print(f"Fibers per order: {fc.fibers_per_order}")
    print(f"Groups: {list(fc.groups.keys())}")


def run_trace(keep_orders=None):
    """Step 1: Trace orders from flat field.

    Parameters
    ----------
    keep_orders : list of int, optional
        If specified, filter the saved npz to only keep these orders.
        E.g., keep_orders=[87] to keep only order 87.
    """
    pipe.trace(trace_files)
    results = pipe.run()

    if keep_orders is not None:
        filter_trace_orders(keep_orders)

    return results


def filter_trace_orders(keep_orders):
    """Filter the trace npz file to only keep specified orders.

    Parameters
    ----------
    keep_orders : list of int
        Order numbers to keep (e.g., [87])
    """
    import numpy as np

    npz_path = join(output_dir, f"andes_riz_{channel.lower()}.traces.npz")
    data = dict(np.load(npz_path, allow_pickle=True))

    # Group data is stored as group_{name}_traces, group_{name}_cr
    group_names = data.get("group_names", [])
    if hasattr(group_names, "tolist"):
        group_names = group_names.tolist()

    modified = False
    for name in group_names:
        for suffix in ["traces", "cr"]:
            key = f"group_{name}_{suffix}"
            if key in data:
                group_data = data[key].item()  # 0-d array containing dict
                if isinstance(group_data, dict):
                    filtered = {
                        o: v for o, v in group_data.items() if int(o) in keep_orders
                    }
                    data[key] = np.array(filtered, dtype=object)
                    modified = True

    if modified:
        np.savez(npz_path, **data)
        print(f"Filtered traces to orders {keep_orders}")


def run_normalize_flat():
    """Step 2: Normalize flat field (requires trace)."""
    pipe.normalize_flat()
    return pipe.run()


def run_wavecal():
    """Steps 3-5: Full wavelength calibration."""
    pipe.wavecal_master(wavecal_files)
    pipe.wavecal_init()
    pipe.wavecal()
    return pipe.run()


def run_all():
    """Run all steps in sequence."""
    pipe.trace(trace_files)
    pipe.normalize_flat()
    pipe.wavecal_master(wavecal_files)
    pipe.wavecal_init()
    pipe.wavecal()
    return pipe.run()


if __name__ == "__main__":
    # Only order 87 spans the full detector in the test data
    # Orders 86 and 88 are partial (edge only)
    KEEP_ORDERS = [87]

    print("\n=== Running trace step ===")
    results = run_trace(keep_orders=KEEP_ORDERS)

    if "trace" in results:
        traces, column_range = results["trace"]
        print(f"Traces found: {len(traces)}")

    if "trace_groups" in results and results["trace_groups"]:
        group_traces, group_cr, group_heights = results["trace_groups"]
        print(f"Fiber groups: {list(group_traces.keys())}")
        for name, traces_dict in group_traces.items():
            if isinstance(traces_dict, dict):
                print(f"  {name}: orders {list(traces_dict.keys())}")
