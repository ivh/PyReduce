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
from pyreduce.trace_orders import group_and_refit, mark_orders, merge_traces

# --- Configuration ---
instrument_name = "AJ"
base_dir = os.path.expanduser("~/REDUCE_DATA/AJ")

# Input files - explicitly specify which is which
file_even = os.path.join(base_dir, "raw/J_FF_even_1s.fits")
file_odd = os.path.join(base_dir, "raw/J_FF_odd_1s.fits")

# Output
output_dir = os.path.join(base_dir, "reduced")
output_file = os.path.join(output_dir, "fiber_traces.npz")

# Order tracing parameters (tune these for your data)
trace_params = {
    "min_cluster": 100,
    "min_width": 0.2,
    "filter_size": 5,
    "noise": 0,
    "opower": 4,
    "degree_before_merge": 4,
    "regularization": 0,
    "closing_shape": (1, 1),
    "border_width": 0,
    "manual": False,
    "auto_merge_threshold": 1.0,
    "merge_min_threshold": 0.1,
    "plot": 1,
}

# Order centers: y-position of each spectral order at x=ncols/2
order_centers = [
    377,
    1217,
    1442,
    1639,
    1879,
    2087,
    2314,
    2539,
    2781,
    3017,
    3265,
    3519,
    3749,
]

# Logical fiber grouping (fiber index ranges)
logical_fibers = {
    "A": (0, 35),
    "cal": (36, 38),
    "B": (39, 75),
}

# --- Load instrument ---
instrument = load_instrument(instrument_name)
arm = instrument.info["arms"][0]  # Use first arm
print(f"Instrument: {instrument.name}, arm: {arm}")

# --- Step 1: Load images using instrument class ---
print(f"\nLoading {file_even}...")
img_even, head_even = instrument.load_fits(file_even, arm=arm, extension=0)
print(f"  Shape: {img_even.shape}, dtype: {img_even.dtype}")

print(f"Loading {file_odd}...")
img_odd, head_odd = instrument.load_fits(file_odd, arm=arm, extension=0)

# --- Step 2: Trace each flat independently ---
print("\nTracing even-illuminated fibers...")
traces_even, cr_even = mark_orders(img_even, plot_title="Even fibers", **trace_params)
print(f"  Found {len(traces_even)} traces")

print("\nTracing odd-illuminated fibers...")
traces_odd, cr_odd = mark_orders(img_odd, plot_title="Odd fibers", **trace_params)
print(f"  Found {len(traces_odd)} traces")

# --- Step 3: Merge traces and assign to spectral orders ---
print("\nMerging traces and assigning to orders...")
traces_by_order, cr_by_order, fiber_ids = merge_traces(
    traces_even,
    cr_even,
    traces_odd,
    cr_odd,
    order_centers=order_centers,
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
        degree=trace_params["opower"],
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
