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

from astropy.io import fits

from pyreduce import util
from pyreduce.combine_frames import combine_calibrate
from pyreduce.configuration import load_config
from pyreduce.instruments import instrument_info
from pyreduce.pipeline import Pipeline

# --- Configuration ---
instrument_name = "ANDES_RIZ"
target = "psf_comp"
night = ""
channel = "R1"  # R0, R1, or R2 (different optical models)

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


def combine_lfc_files(lfc_files, output_path):
    """Combine odd+even LFC files into a single frame."""
    print(f"Combining {len(lfc_files)} LFC files...")
    combined, head = combine_calibrate(
        lfc_files,
        pipe.instrument,
        channel,
        mask=None,
    )
    # Convert masked array to regular array for FITS writing
    import numpy as np

    data = np.asarray(combined.filled(0) if hasattr(combined, "filled") else combined)
    fits.writeto(output_path, data, head, overwrite=True)
    print(f"Saved combined LFC: {output_path}")
    return output_path


if __name__ == "__main__":
    # Combine odd+even LFC files before extraction
    lfc_combined_path = join(output_dir, "lfc_combined.fits")
    if wavecal_files:
        combine_lfc_files(wavecal_files, lfc_combined_path)

    print("\n=== Running pipeline ===")
    # pipe.trace(trace_files)
    # pipe.curvature(wavecal_files)
    pipe.extract([lfc_combined_path])

    results = pipe.run()

    if "trace" in results:
        traces, column_range, *_ = results["trace"]
        print(f"Traces found: {len(traces)}")

    if "trace_groups" in results and results["trace_groups"]:
        group_traces, group_cr, group_heights = results["trace_groups"]
        print(f"Fiber groups: {list(group_traces.keys())}")
        for name, traces_dict in group_traces.items():
            if isinstance(traces_dict, dict):
                print(f"  {name}: orders {list(traces_dict.keys())}")
