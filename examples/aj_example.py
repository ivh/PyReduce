# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7b3"]
# ///
"""
AJ instrument example: Multi-fiber tracing with Pipeline API.

Demonstrates tracing fibers illuminated in separate flat field images
(even/odd pattern) using the Pipeline's trace_raw() and organize() methods.

The fiber config in AJ/config.yaml handles:
- order_centers_file: assigns traces to spectral orders by y-position
- groups: organizes fibers into logical groups (A, cal, B) within each order
- merge: average - averages fiber traces within each group
"""

import os

import numpy as np

from pyreduce.configuration import load_config
from pyreduce.extract import extract
from pyreduce.pipeline import Pipeline

# --- Configuration ---
instrument_name = "AJ"
data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
raw_dir = os.path.join(data_dir, "AJ", "raw")
output_dir = os.path.join(data_dir, "AJ", "reduced")

# Input files (even and odd illuminated flats)
file_even = os.path.join(raw_dir, "J_FF_even_1s.fits")
file_odd = os.path.join(raw_dir, "J_FF_odd_1s.fits")

# Plot settings
plot = int(os.environ.get("PYREDUCE_PLOT", "1"))

# --- Create Pipeline ---
config = load_config(None, instrument_name)
pipe = Pipeline(
    instrument=instrument_name,
    output_dir=output_dir,
    target="AJ_fiber_test",
    config=config,
    plot=plot,
)

print(f"Instrument: {pipe.instrument.name}")
fibers_config = pipe.instrument.config.fibers
print(f"Per-order grouping: {fibers_config.per_order}")
print(f"Groups: {list(fibers_config.groups.keys())}")

# --- Trace or load from previous run ---
LOAD_TRACE = True  # Set False to re-run tracing

if LOAD_TRACE:
    print("\nLoading traces from previous run...")
    orders, column_range = pipe._run_step("trace", None, load_only=True)
    print(f"  Loaded {len(orders)} traces")
else:
    # Trace each flat independently
    print(f"\nTracing even fibers from {os.path.basename(file_even)}...")
    traces_even, cr_even = pipe.trace_raw([file_even])
    print(f"  Found {len(traces_even)} traces")

    print(f"\nTracing odd fibers from {os.path.basename(file_odd)}...")
    traces_odd, cr_odd = pipe.trace_raw([file_odd])
    print(f"  Found {len(traces_odd)} traces")

    # Organize into fiber groups
    print("\nOrganizing traces into fiber groups...")
    pipe.organize(traces_even, cr_even, traces_odd, cr_odd)

# Access organized groups
if "trace_groups" in pipe._data and pipe._data["trace_groups"][0]:
    group_traces, group_cr = pipe._data["trace_groups"]
    print("Fiber groups:")
    for name, traces_dict in group_traces.items():
        n_orders = len(traces_dict)
        print(f"  {name}: {n_orders} orders")

# --- Extract group A as example ---
print("\nExtracting group A spectra...")

# Load combined flat for extraction
img_even, _ = pipe.instrument.load_fits(file_even, channel="ALL", extension=0)
img_odd, _ = pipe.instrument.load_fits(file_odd, channel="ALL", extension=0)
img_combined = img_even.astype(np.float64) + img_odd.astype(np.float64)

# Get group A traces (stacked across orders)
if "trace_groups" in pipe._data and pipe._data["trace_groups"][0]:
    group_traces, group_cr = pipe._data["trace_groups"]
    a_orders = sorted(group_traces["A"].keys())
    a_traces = np.vstack([group_traces["A"][m] for m in a_orders])
    a_cr = np.vstack([group_cr["A"][m] for m in a_orders])
else:
    # Fallback to raw traces
    a_traces, a_cr = pipe._data["trace"]

print(f"  Extracting {len(a_traces)} traces...")

norm, spec, blaze, unc = extract(
    img_combined,
    a_traces,
    column_range=a_cr,
    extraction_type="normalize",
    extraction_height=config["science"]["extraction_height"],
    osample=config["science"]["oversampling"],
    gain=1.0,
    readnoise=0.0,
    dark=0.0,
    plot=plot,
)

print(f"  Blaze shape: {blaze.shape}")

# Save results
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "aj_blaze.npz")
np.savez(output_file, blaze=blaze, norm=norm, spec=spec, unc=unc)
print(f"\nSaved to: {output_file}")
