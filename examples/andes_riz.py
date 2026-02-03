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


def combine_lfc_with_flats(lfc_files, flat_files, output_path):
    """Combine LFC files with flat-field frames to add continuum."""
    import numpy as np

    print(f"Combining {len(lfc_files)} LFC + {len(flat_files)} flat files...")

    # Combine LFC files
    lfc_combined, head = combine_calibrate(
        lfc_files,
        pipe.instrument,
        channel,
        mask=None,
    )
    lfc_data = np.asarray(
        lfc_combined.filled(0) if hasattr(lfc_combined, "filled") else lfc_combined
    )

    if flat_files:
        # Combine flat files
        flat_combined, _ = combine_calibrate(
            flat_files,
            pipe.instrument,
            channel,
            mask=None,
        )
        flat_data = np.asarray(
            flat_combined.filled(0)
            if hasattr(flat_combined, "filled")
            else flat_combined
        )

        # Add flat continuum to LFC (scale flat to ~10% of LFC peak to not overwhelm lines)
        lfc_peak = (
            np.percentile(lfc_data[lfc_data > 0], 95) if np.any(lfc_data > 0) else 1
        )
        flat_peak = (
            np.percentile(flat_data[flat_data > 0], 95) if np.any(flat_data > 0) else 1
        )
        scale = 0.1 * lfc_peak / flat_peak
        combined = lfc_data + flat_data * scale
        print(
            f"  LFC peak: {lfc_peak:.0f}, flat peak: {flat_peak:.0f}, scale: {scale:.4f}"
        )
    else:
        combined = lfc_data
        print("  No flats available, using LFC only")

    fits.writeto(output_path, combined.astype(np.float32), head, overwrite=True)
    print(f"Saved combined LFC+flat: {output_path}")
    return output_path


if __name__ == "__main__":
    # Combine LFC + flat files before extraction
    lfc_combined_path = join(output_dir, "lfc_combined.fits")
    if wavecal_files and flat_files:
        combine_lfc_with_flats(wavecal_files, flat_files, lfc_combined_path)
    elif wavecal_files:
        # Fallback to LFC only if no flats
        combine_lfc_with_flats(wavecal_files, [], lfc_combined_path)

    print("\n=== Running pipeline ===")
    # pipe.trace(trace_files)
    # pipe.curvature(wavecal_files)
    # Full wavecal: master -> init (MCMC line matching) -> finalize
    pipe.wavelength_calibration([lfc_combined_path])
    pipe.extract([lfc_combined_path])

    results = pipe.run()

    if "trace" in results:
        traces = results["trace"]  # list[Trace]
        print(f"Traces found: {len(traces)}")
        for t in traces[:3]:
            print(f"  m={t.m}, group={t.group}, columns={t.column_range}")
