# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7a5"]
# ///
"""
AJ instrument example: Fiber bundle tracing with direct function calls.

This script demonstrates tracing a multi-fiber instrument where even and
odd fibers are illuminated in separate flat field images. Instead of
embedding this logic in the Pipeline, we call the functions directly,
giving full control over the workflow and access to intermediate results.

The instrument class still provides:
- Image loading with proper orientation
- Header parsing
- Detector properties (gain, readnoise, etc.)

The script controls:
- Which files to use for what
- Order of operations
- Parameters for each step
- What to save and where
"""

import os

import numpy as np

from pyreduce.instruments.instrument_info import load_instrument
from pyreduce.trace import group_and_refit, merge_traces, trace

# --- Configuration ---
instrument_name = "AJ"
raw_dir = os.path.expanduser("~/REDUCE_DATA/AJ/raw")

# Input files
file_even = os.path.join(raw_dir, "J_FF_even_1s.fits")
file_odd = os.path.join(raw_dir, "J_FF_odd_1s.fits")
orders_file = os.path.join(raw_dir, "ANDES_75fibre_J_orders.npz")

# Output
output_dir = os.path.expanduser("~/REDUCE_DATA/AJ/reduced")
output_file = os.path.join(output_dir, "fiber_traces.npz")

# Load order centers from npz file
orders_data = np.load(orders_file)
order_numbers = orders_data["order"]
order_centers = orders_data["y_mid"]
print(
    f"Loaded {len(order_centers)} orders ({order_numbers[0]}-{order_numbers[-1]}) from {orders_file}"
)

# Order tracing parameters (tune these for your data)
trace_params = {
    "min_cluster": 500,
    "min_width": 0.1,  # Fraction of detector height
    "filter_x": 0,  # Smooth along dispersion to reduce noise
    "filter_y": 4,  # Small value to preserve thin fiber separation
    "noise": 0,
    "degree": 4,
    "degree_before_merge": 4,
    "regularization": 0,
    "closing_shape": (1, 1),
    "opening_shape": (1, 1),
    "border_width": 0,
    "manual": False,
    "auto_merge_threshold": 1.0,
    "merge_min_threshold": 0.1,
    "sigma": 0,
    "plot": 1,
}

# Logical fiber grouping (fiber number ranges, 1-based)
logical_fibers = {
    "A": (1, 36),
    "cal": (37, 39),
    "B": (40, 76),
}

# --- Load instrument ---
instrument = load_instrument(instrument_name)
channel = instrument.info["channels"][0]  # Use first channel
print(f"Instrument: {instrument.name}, channel: {channel}")

# --- Step 1: Load images using instrument class ---
print(f"\nLoading {file_even}...")
img_even, head_even = instrument.load_fits(file_even, channel=channel, extension=0)
print(f"  Shape: {img_even.shape}, dtype: {img_even.dtype}")

print(f"Loading {file_odd}...")
img_odd, head_odd = instrument.load_fits(file_odd, channel=channel, extension=0)

# --- Step 2: Trace each flat independently ---
print("\nTracing even-illuminated fibers...")
traces_even, cr_even = trace(
    img_even, plot_title="Even fibers", debug_dir="./debug/even", **trace_params
)
print(f"  Found {len(traces_even)} traces")

print("\nTracing odd-illuminated fibers...")
traces_odd, cr_odd = trace(
    img_odd, plot_title="Odd fibers", debug_dir="./debug/odd", **trace_params
)
print(f"  Found {len(traces_odd)} traces")

# --- Step 3: Merge traces and assign to spectral orders ---
print("\nMerging traces and assigning to orders...")
traces_by_order, cr_by_order, fiber_ids = merge_traces(
    traces_even,
    cr_even,
    traces_odd,
    cr_odd,
    order_centers=order_centers,
    order_numbers=order_numbers,
    ncols=img_even.shape[1],
)

print(f"  Assigned to {len(traces_by_order)} spectral orders:")
for oid in sorted(traces_by_order.keys()):
    print(f"    Order {oid}: {len(traces_by_order[oid])} traces")

# --- Step 4: Group into logical fibers (optional) ---
# This step averages physical fiber traces into logical fiber traces.
# It may fail if too few traces are detected - trace parameters need tuning.
print("\nGrouping into logical fibers...")
try:
    logical_traces, logical_cr, fiber_counts = group_and_refit(
        traces_by_order,
        cr_by_order,
        fiber_ids,
        groups=logical_fibers,
        degree=trace_params["degree"],
    )

    for name, (start, end) in logical_fibers.items():
        total = sum(fiber_counts[name].values())
        print(f"  {name}: {total} physical fibers (range {start}-{end})")
except Exception as e:
    print(f"  Grouping failed: {e}")
    print("  (This is expected if trace detection found too few traces)")
    print("  Skipping grouping step - saving raw traces only")
    logical_traces = None

# --- Step 5: Save results ---
os.makedirs(output_dir, exist_ok=True)

save_dict = {
    # Raw traces per order (always saved)
    **{f"traces_order_{k}": v for k, v in traces_by_order.items()},
    **{f"cr_order_{k}": v for k, v in cr_by_order.items()},
    **{f"fiber_ids_order_{k}": v for k, v in fiber_ids.items()},
}

if logical_traces is not None:
    # Add logical fiber traces
    save_dict["orders"] = np.array(logical_traces["A"])
    save_dict["column_range"] = np.array(logical_cr)
    save_dict["traces_A"] = np.array(logical_traces["A"])
    save_dict["traces_B"] = np.array(logical_traces["B"])
    save_dict["traces_cal"] = np.array(logical_traces["cal"])
else:
    # Fallback: flatten all traces as orders
    all_traces = np.vstack([traces_by_order[k] for k in sorted(traces_by_order.keys())])
    all_cr = np.vstack([cr_by_order[k] for k in sorted(cr_by_order.keys())])
    save_dict["orders"] = all_traces
    save_dict["column_range"] = all_cr

np.savez(output_file, **save_dict)
print(f"\nSaved to: {output_file}")
