# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7b3"]
# ///
"""
MOSAIC NIR spectrograph example.

Demonstrates fiber bundle tracing and extraction on simulated E2E data.
After tracing all fibers, identifies the central fiber in each group of 7
and extracts only those 90 traces.
"""

import os
from os.path import join

import numpy as np

from pyreduce import util
from pyreduce.configuration import load_config
from pyreduce.instruments.instrument_info import load_instrument
from pyreduce.reduce import (
    OrderTracing,
    ScienceExtraction,
    SlitCurvatureDetermination,
)

# Parameters
instrument_name = "MOSAIC"
target = "MOSAIC_NIR"
night = ""
channel = "NIR"
order_range = None
plot = 1

# Handle plot environment variables
if "PYREDUCE_PLOT" in os.environ:
    plot = int(os.environ["PYREDUCE_PLOT"])
plot_dir = os.environ.get("PYREDUCE_PLOT_DIR")
util.set_plot_dir(plot_dir)

# Data location
data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
base_dir = join(data_dir, "MOSAIC", "REF_E2E", "NIR")
output_dir = join(data_dir, "MOSAIC", "reduced", "NIR")

os.makedirs(output_dir, exist_ok=True)

# File paths (simulated data)
flat_file = join(
    base_dir,
    "E2E_FLAT_DIT_20s_MOSAIC_2Cam_c01",
    "E2E_FLAT_DIT_20s_MOSAIC_2Cam_c01_STATIC_FOCAL_PLANE.fits",
)
thar_file = join(
    base_dir,
    "E2E_ThAr_DIT_20s_MOSAIC_2Cam_c01",
    "E2E_ThAr_DIT_20s_MOSAIC_2Cam_c01_STATIC_FOCAL_PLANE.fits",
)

# Verify files exist
for fpath in [flat_file, thar_file]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")

print(f"FLAT: {flat_file}")
print(f"ThAr: {thar_file}")

# Load instrument and configuration
instrument = load_instrument(instrument_name)
config = load_config(None, instrument_name, 0)

# Common step arguments
step_args = (instrument, channel, target, night, output_dir, order_range)


def step_config(name):
    """Get step config with plot level override."""
    cfg = config.get(name, {}).copy()
    cfg["plot"] = plot
    return cfg


print("\n=== TRACE ===")
trace_step = OrderTracing(*step_args, **step_config("trace"))
# traces, column_range = trace_step.run([flat_file])
traces, column_range = trace_step.load()
print(f"Found {len(traces)} traces (expected ~630)")

# --- STEP 3: Identify group centers from gap pattern ---
print("\n=== IDENTIFY GROUP CENTERS ===")

# Evaluate all traces at detector center (x=2048)
x_center = 2048
traced_y = np.array([np.polyval(o, x_center) for o in traces])
print(f"Traced y-positions at x={x_center}: {traced_y[0]:.1f} to {traced_y[-1]:.1f}")

# Sort traces by y position
sort_idx = np.argsort(traced_y)
sorted_y = traced_y[sort_idx]

# Compute gaps between consecutive traces
gaps = np.diff(sorted_y)

# Inter-group gaps are ~2x larger than intra-group gaps.
# Use median gap as the intra-group spacing estimate (robust to outliers).
median_gap = np.median(gaps)
threshold = 1.5 * median_gap  # Between 1x (intra) and 2x (inter)
print(f"Median gap: {median_gap:.2f} px, threshold: {threshold:.2f} px")

# Find group boundaries (where gap exceeds threshold)
is_group_boundary = gaps > threshold

# Assign group IDs: increment at each boundary
group_ids = np.zeros(len(sorted_y), dtype=int)
group_ids[1:] = np.cumsum(is_group_boundary)

n_groups = group_ids.max() + 1
print(f"Identified {n_groups} groups")

# For each group, find the center trace (middle element)
center_trace_indices = []
group_sizes = []

for g in range(n_groups):
    group_mask = group_ids == g
    group_orig_indices = sort_idx[group_mask]
    group_y_values = traced_y[group_orig_indices]

    # Sort by y within group to find center
    within_sort = np.argsort(group_y_values)
    center_idx = len(within_sort) // 2  # Middle element
    center_trace_indices.append(group_orig_indices[within_sort[center_idx]])
    group_sizes.append(len(group_orig_indices))

center_trace_indices = np.array(center_trace_indices)
group_sizes = np.array(group_sizes)

# Report group size distribution
size_counts = {sz: (group_sizes == sz).sum() for sz in sorted(set(group_sizes))}
print(f"Group sizes: {size_counts}")

# Extract only the center traces
center_traces = traces[center_trace_indices]
center_column_range = column_range[center_trace_indices]
print(f"\nUsing {len(center_traces)} group center traces for extraction")

center_trace = (center_traces, center_column_range)

print("\n=== CURVATURE ===")
curve_step = SlitCurvatureDetermination(*step_args, **step_config("curvature"))
# curvature = curve_step.run([thar_file], center_trace)
curvature = curve_step.load()
print("Curvature determination complete")

science_step = ScienceExtraction(*step_args, **step_config("science"))

print("\n=== EXTRACT ThAr ===")
try:
    thar_spec = science_step.run([thar_file], center_trace, curvature=curvature)
    print("ThAr extraction complete")
except Exception as e:
    print(f"ThAr extraction failed: {e}")
    thar_spec = None

print("\n=== EXTRACT FLAT ===")
try:
    flat_spec = science_step.run([flat_file], center_trace, curvature=curvature)
    print("FLAT extraction complete")
except Exception as e:
    print(f"FLAT extraction failed: {e}")
    flat_spec = None
