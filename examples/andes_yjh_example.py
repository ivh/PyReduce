# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7b4"]
# ///
"""
ANDES_YJH instrument example: Multi-fiber tracing with Pipeline API.

Demonstrates tracing fibers illuminated in separate flat field images
(even/odd pattern) using the Pipeline's trace_raw() and organize() methods.

The fiber config in ANDES_YJH/config.yaml handles:
- order_centers_file: assigns traces to spectral orders by y-position (channel-specific)
- groups: organizes fibers into logical groups (A, cal, B) within each order
- merge: average - averages fiber traces within each group

ANDES_YJH has three channels: Y, J, H (selected by BAND header in files).
"""

import os

import numpy as np

from pyreduce.configuration import load_config
from pyreduce.pipeline import Pipeline

# --- Configuration ---
instrument_name = "ANDES_YJH"
channel = "J"  # Y, J, or H
data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
raw_dir = os.path.join(data_dir, "ANDES", channel)
output_dir = os.path.join(data_dir, "ANDES", "reduced", channel)

# Input files (even and odd illuminated flats)
# File selection is header-based:
#   BAND header determines channel (Y, J, H)
#   SIMTYPE='flat_field' → flat, SIMTYPE='spectrum' → science
#   FIBMODE='even' → even flat, FIBMODE='odd' → odd flat
file_even = os.path.join(raw_dir, f"{channel}_FF_even_1s.fits")
file_odd = os.path.join(raw_dir, f"{channel}_FF_odd_1s.fits")

# Plot settings
plot = int(os.environ.get("PYREDUCE_PLOT", "1"))

# --- Create Pipeline ---
config = load_config(None, instrument_name)
pipe = Pipeline(
    instrument=instrument_name,
    channel=channel,
    output_dir=output_dir,
    target="ANDES_fiber_test",
    config=config,
    plot=plot,
)

print(f"Instrument: {pipe.instrument.name}")
fibers_config = pipe.instrument.config.fibers
print(f"Per-order grouping: {fibers_config.per_order}")
print(f"Groups: {list(fibers_config.groups.keys())}")

# --- Trace or load from previous run ---
LOAD_TRACE = True  # Set True to load traces from previous run

if LOAD_TRACE:
    print("\nLoading traces from previous run...")
    traces, column_range, heights = pipe._run_step("trace", None, load_only=True)
    print(f"  Loaded {len(traces)} traces")

    # Re-run organize with current config (picks up any config changes)
    print("\nRe-organizing traces with current config...")
    pipe.organize(traces, column_range)
else:
    # Trace each flat independently
    print(f"\nTracing even fibers from {os.path.basename(file_even)}...")
    traces_even, cr_even, _ = pipe.trace_raw([file_even])
    print(f"  Found {len(traces_even)} traces")

    print(f"\nTracing odd fibers from {os.path.basename(file_odd)}...")
    traces_odd, cr_odd, _ = pipe.trace_raw([file_odd])
    print(f"  Found {len(traces_odd)} traces")

    # Organize into fiber groups
    print("\nOrganizing traces into fiber groups...")
    pipe.organize(traces_even, cr_even, traces_odd, cr_odd)

# Access organized groups
if "trace_groups" in pipe._data and pipe._data["trace_groups"][0]:
    group_traces, group_cr, group_heights = pipe._data["trace_groups"]
    print("Fiber groups:")
    for name, traces_dict in group_traces.items():
        n_traces = len(traces_dict)
        print(f"  {name}: {n_traces} traces")

# --- Create combined flat for extraction ---
print("\nCombining even/odd flats...")
img_even, head = pipe.instrument.load_fits(file_even, channel=channel, extension=0)
img_odd, _ = pipe.instrument.load_fits(file_odd, channel=channel, extension=0)
img_combined = np.asarray(img_even, dtype=np.float64) + np.asarray(
    img_odd, dtype=np.float64
)

# Save combined flat to file for the science step
from astropy.io import fits

combined_file = os.path.join(output_dir, "combined_flat.fits")
os.makedirs(output_dir, exist_ok=True)
fits.writeto(combined_file, img_combined.astype(np.float32), head, overwrite=True)
print(f"  Saved combined flat: {combined_file}")

# --- Extract using the science step ---
print("\nExtracting spectra (group A from fiber config)...")
pipe.instrument.config.fibers.use["science"] = ["ring1"]
pipe.config["science"]["extraction_height"] = 60
pipe.extract([combined_file]).run()
