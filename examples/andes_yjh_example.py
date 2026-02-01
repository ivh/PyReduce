# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7"]
# ///
"""
ANDES_YJH instrument example: Multi-fiber tracing with Pipeline API.

Demonstrates tracing fibers illuminated in separate flat field images
(even/odd pattern). The Pipeline trace() step handles merging traces
from multiple files.

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
    trace_objects = pipe._run_step("trace", None, load_only=True)  # list[Trace]
    print(f"  Loaded {len(trace_objects)} traces")
else:
    # Trace both files together - the pipeline will organize by fiber config
    print(
        f"\nTracing fibers from {os.path.basename(file_even)} and {os.path.basename(file_odd)}..."
    )
    pipe.trace([file_even, file_odd])
    results = pipe.run()
    trace_objects = results["trace"]  # list[Trace]
    print(f"  Found {len(trace_objects)} traces")

# Show trace info
print("\nTraces:")
fibers = {t.fiber for t in trace_objects}
print(f"  Fibers: {sorted(fibers)}")
for fiber in sorted(fibers)[:3]:
    count = sum(1 for t in trace_objects if t.fiber == fiber)
    print(f"  {fiber}: {count} traces")

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
pipe.instrument.config.fibers.use["science"] = ["ring2"]
pipe.extract([combined_file]).run()
