# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7b2"]
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
    Mask,
    NormalizeFlatField,
    OrderTracing,
    ScienceExtraction,
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
fiber_pos_file = join(data_dir, "MOSAIC", "mosaic_fiber_positions.npz")

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
for fpath in [flat_file, thar_file, fiber_pos_file]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")

print(f"FLAT: {flat_file}")
print(f"ThAr: {thar_file}")
print(f"Fiber positions: {fiber_pos_file}")

# Load fiber positions (expected positions from instrument design)
fiber_data = np.load(fiber_pos_file)
group_center_pix_y = fiber_data["group_center_pix_y"]  # 90 group centers
print("\nExpected: 630 fibers in 90 groups of 7")
print(
    f"Group center positions: {len(group_center_pix_y)} (y={group_center_pix_y[0]:.1f} to {group_center_pix_y[-1]:.1f})"
)

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


# --- STEP 1: Mask ---
print("\n=== MASK ===")
mask_step = Mask(*step_args, **step_config("mask"))
mask = mask_step.run()

bias = None

# --- STEP 2: Trace all fibers ---
print("\n=== TRACE ===")
trace_step = OrderTracing(*step_args, **step_config("trace"))
orders, column_range = trace_step.run([flat_file], mask=mask)
print(f"Found {len(orders)} traces (expected ~630)")

# --- STEP 3: Match traces to group centers ---
print("\n=== IDENTIFY GROUP CENTERS ===")

# Evaluate all traces at detector center (x=2048)
x_center = 2048
traced_y = np.array([np.polyval(o, x_center) for o in orders])
print(f"Traced y-positions at x={x_center}: {traced_y[0]:.1f} to {traced_y[-1]:.1f}")

# For each expected group center, find the closest traced fiber
center_trace_indices = []
center_distances = []

for expected_y in group_center_pix_y:
    distances = np.abs(traced_y - expected_y)
    closest_idx = np.argmin(distances)
    closest_dist = distances[closest_idx]
    center_trace_indices.append(closest_idx)
    center_distances.append(closest_dist)

center_trace_indices = np.array(center_trace_indices)
center_distances = np.array(center_distances)

print(f"Matched {len(center_trace_indices)} group centers")
print(
    f"Match distances: mean={center_distances.mean():.2f}, max={center_distances.max():.2f} pixels"
)

# Check for any bad matches (distance > half fiber spacing ~3 pixels)
bad_matches = center_distances > 3.0
if np.any(bad_matches):
    print(f"WARNING: {bad_matches.sum()} matches have distance > 3 pixels")
    for i in np.where(bad_matches)[0]:
        print(
            f"  Group {i}: expected y={group_center_pix_y[i]:.1f}, closest trace y={traced_y[center_trace_indices[i]]:.1f}, dist={center_distances[i]:.1f}"
        )

# Extract only the center traces
center_orders = orders[center_trace_indices]
center_column_range = column_range[center_trace_indices]
print(f"\nUsing {len(center_orders)} group center traces for extraction")

center_trace = (center_orders, center_column_range)

# --- STEP 4: Prepare flat data ---
print("\n=== FLAT ===")
from astropy.io import fits

flat_data = fits.getdata(flat_file).astype(float)
with fits.open(flat_file) as hdul:
    flat_header = hdul[0].header.copy()
flat = (flat_data, flat_header)

# --- STEP 5: Normalize flat (using center traces) ---
print("\n=== NORM_FLAT ===")
norm_flat_step = NormalizeFlatField(*step_args, **step_config("norm_flat"))
try:
    norm_flat = norm_flat_step.run(flat, center_trace)
    print("Normalized flat complete")
except Exception as e:
    print(f"Norm flat failed: {e}")
    norm_flat = None

# --- STEP 6: Extract from FLAT ---
print("\n=== EXTRACT FLAT ===")
science_step = ScienceExtraction(*step_args, **step_config("science"))
try:
    flat_spec = science_step.run([flat_file], center_trace)
    print("FLAT extraction complete")
except Exception as e:
    print(f"FLAT extraction failed: {e}")
    flat_spec = None

# --- STEP 7: Extract from ThAr ---
print("\n=== EXTRACT ThAr ===")
try:
    thar_spec = science_step.run([thar_file], center_trace)
    print("ThAr extraction complete")
except Exception as e:
    print(f"ThAr extraction failed: {e}")
    thar_spec = None

print("\nDone!")
print(f"Output saved to: {output_dir}")
