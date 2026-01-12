# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7b3"]
# ///
"""
AJ instrument example using config-based per-order fiber grouping.

This script demonstrates:
1. Using OrderTracing step class with settings from settings.json
2. Tracing even and odd flat fields separately
3. Combining traces and using config-based per-order fiber grouping
4. Extracting spectra from grouped fibers

The fiber config in AJ/config.yaml handles:
- per_order: true - enables per-spectral-order grouping
- order_centers_file - y-positions to assign traces to orders
- groups - logical fiber groups (A, cal, B) within each order
- use - which groups to use for each reduction step
"""

import os

import numpy as np

from pyreduce.configuration import load_config
from pyreduce.extract import extract
from pyreduce.instruments.instrument_info import load_instrument
from pyreduce.reduce import OrderTracing
from pyreduce.trace import organize_fibers

# --- Configuration ---
instrument_name = "AJ"
data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
raw_dir = os.path.join(data_dir, "AJ", "raw")
output_dir = os.path.join(data_dir, "AJ", "reduced")

# Input files
file_even = os.path.join(raw_dir, "J_FF_even_1s.fits")
file_odd = os.path.join(raw_dir, "J_FF_odd_1s.fits")

# Output
output_file = os.path.join(output_dir, "fiber_traces_v2.npz")

# --- Load instrument and config ---
instrument = load_instrument(instrument_name)
channel = instrument.info["channels"][0]
config = load_config(None, instrument_name)
plot = int(os.environ.get("PYREDUCE_PLOT", "1"))

fibers_config = instrument.config.fibers
print(f"Instrument: {instrument.name}")
print(f"Per-order grouping: {fibers_config.per_order}")
print(f"Fibers per order: {fibers_config.fibers_per_order}")
print(f"Groups: {list(fibers_config.groups.keys())}")

# --- Create OrderTracing step (once, reuse for both flats) ---
trace_config = {**config["trace"], "plot": plot}
trace_step = OrderTracing(
    instrument,
    channel=channel,
    target="AJ_test",
    night="",
    output_dir=output_dir,
    order_range=None,
    **trace_config,
)

# --- Step 1: Load images ---
print(f"\nLoading {file_even}...")
img_even, head_even = instrument.load_fits(file_even, channel=channel, extension=0)
print(f"  Shape: {img_even.shape}")

print(f"Loading {file_odd}...")
img_odd, head_odd = instrument.load_fits(file_odd, channel=channel, extension=0)

# Combined flat for extraction
img_combined = img_even.astype(np.float64) + img_odd.astype(np.float64)

# --- Step 2: Trace each flat using OrderTracing step ---
print("\nTracing even-illuminated fibers...")
trace_step.plot_title = "Even fibers"
traces_even, cr_even = trace_step.run([file_even])
print(f"  Found {len(traces_even)} traces")

print("\nTracing odd-illuminated fibers...")
trace_step.plot_title = "Odd fibers"
traces_odd, cr_odd = trace_step.run([file_odd])
print(f"  Found {len(traces_odd)} traces")

# --- Step 3: Combine all traces ---
all_traces = np.vstack([traces_even, traces_odd])
all_cr = np.vstack([cr_even, cr_odd])
print(f"\nTotal traces: {len(all_traces)}")

# --- Step 4: Organize into fiber groups using config ---
print("\nOrganizing traces into fiber groups...")
channels = instrument.channels or []
channel_index = channels.index(channel.upper()) if channel.upper() in channels else 0
group_traces, group_cr, group_counts = organize_fibers(
    all_traces,
    all_cr,
    fibers_config,
    degree=config["trace"]["degree"],
    instrument_dir=instrument._inst_dir,
    channel_index=channel_index,
)

print("Fiber groups created:")
for name, traces_dict in group_traces.items():
    n_orders = len(traces_dict)
    print(f"  {name}: {n_orders} orders, {group_counts[name]} fibers per order")

# --- Step 5: Save results ---
os.makedirs(output_dir, exist_ok=True)

save_dict = {"raw_traces": all_traces, "raw_cr": all_cr}
for group_name, order_dict in group_traces.items():
    for order_m, tr in order_dict.items():
        save_dict[f"{group_name}_order_{order_m}"] = tr
        save_dict[f"{group_name}_cr_{order_m}"] = group_cr[group_name][order_m]

np.savez(output_file, **save_dict)
print(f"\nSaved traces to: {output_file}")

# --- Step 6: Extract from group A as example ---
print("\nExtracting group A spectra...")

# Stack A traces across orders (sorted by order number)
a_traces = np.vstack([group_traces["A"][m] for m in sorted(group_traces["A"].keys())])
a_cr = np.vstack([group_cr["A"][m] for m in sorted(group_cr["A"].keys())])
print(f"  Group A: {len(a_traces)} traces (one per order)")

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

print(f"  Extracted blaze shape: {blaze.shape}")
print("Done.")
