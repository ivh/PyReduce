# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7b2"]
# ///
"""
MOSAIC NIR spectrograph example.

Demonstrates fiber bundle tracing and extraction on simulated E2E data.
Uses peak-based tracing for dense fiber spectrographs (630 fibers).
"""

import os
from os.path import join

import numpy as np
from astropy.io import fits
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import find_peaks

from pyreduce import util
from pyreduce.configuration import load_config
from pyreduce.instruments.instrument_info import load_instrument
from pyreduce.reduce import (
    NormalizeFlatField,
    ScienceExtraction,
)

# Parameters
instrument_name = "MOSAIC"
target = "MOSAIC_NIR"
night = ""
channel = "NIR"
order_range = None  # Process all orders
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


# --- STEP 1: Trace fibers using peak detection ---
# Standard cluster-based tracing is too slow for 630 fibers.
# Use peak detection across columns instead.

print("\n=== TRACE (peak-based) ===")
trace_file = join(output_dir, "mosaic_nir.ord_default.npz")

if os.path.exists(trace_file):
    print(f"Loading existing traces from {trace_file}")
    trace_data = np.load(trace_file)
    orders = trace_data["orders"]
    column_range = trace_data["column_range"]
else:
    print("Running peak-based fiber tracing...")
    flat_data = fits.getdata(flat_file)
    ncol = flat_data.shape[1]

    # Find peaks at several column positions
    sample_cols = [100, 500, 1000, 1500, 2048, 2500, 3000, 3500, 3900]
    all_peak_positions = []
    for col_idx in sample_cols:
        col = flat_data[:, col_idx]
        height_thresh = np.percentile(col, 90) * 0.1
        peaks, _ = find_peaks(col, height=height_thresh, distance=4)
        all_peak_positions.append((col_idx, peaks))
        print(f"  Column {col_idx}: {len(peaks)} peaks")

    # Use center column as reference
    ref_col_idx = 2048
    ref_peaks = [p[1] for p in all_peak_positions if p[0] == ref_col_idx][0]

    # Match peaks across columns to form traces
    traces = []
    for peak_y in ref_peaks:
        trace_points = [(ref_col_idx, peak_y)]
        for col_idx, peaks in all_peak_positions:
            if col_idx == ref_col_idx:
                continue
            if len(peaks) > 0:
                distances = np.abs(peaks - peak_y)
                closest_idx = np.argmin(distances)
                if distances[closest_idx] < 20:
                    trace_points.append((col_idx, peaks[closest_idx]))
        if len(trace_points) >= 5:
            traces.append(trace_points)

    # Fit polynomials
    orders = []
    column_ranges = []
    for trace_points in traces:
        x = np.array([p[0] for p in trace_points])
        y = np.array([p[1] for p in trace_points])
        fit = Polynomial.fit(x, y, deg=2, domain=[])
        coeffs = fit.coef[::-1]
        orders.append(coeffs)
        column_ranges.append([50, ncol - 50])

    orders = np.array(orders)
    column_range = np.array(column_ranges)

    # Sort by y-position at center
    center_y = np.array([np.polyval(o, 2048) for o in orders])
    sort_idx = np.argsort(center_y)
    orders = orders[sort_idx]
    column_range = column_range[sort_idx]

    # Save traces
    np.savez(trace_file, orders=orders, column_range=column_range)
    print(f"Saved {len(orders)} traces to {trace_file}")

trace = (orders, column_range)
print(f"Using {len(orders)} fiber traces")

# --- STEP 2: Prepare flat data ---
print("\n=== FLAT ===")
flat_data = fits.getdata(flat_file).astype(float)
with fits.open(flat_file) as hdul:
    flat_header = hdul[0].header.copy()
flat = (flat_data, flat_header)

# No bias, scatter, curvature, or mask for simulated data
bias = None
scatter = None
curvature = None
mask = None

# --- STEP 3: Normalize flat (optional, for blaze function) ---
print("\n=== NORM_FLAT ===")
norm_flat_step = NormalizeFlatField(*step_args, **step_config("norm_flat"))

# Limit to subset for testing
test_order_range = (0, min(20, len(orders)))
print(f"Processing orders {test_order_range[0]}-{test_order_range[1]} of {len(orders)}")

# Temporarily limit orders for norm_flat
test_trace = (
    orders[test_order_range[0] : test_order_range[1]],
    column_range[test_order_range[0] : test_order_range[1]],
)
try:
    norm_flat = norm_flat_step.run(flat, test_trace, scatter, curvature)
    print("Normalized flat complete")
except Exception as e:
    print(f"Norm flat failed: {e}")
    norm_flat = None

# --- STEP 4: Extract from FLAT ---
print("\n=== EXTRACT FLAT ===")
science_step = ScienceExtraction(*step_args, **step_config("science"))

try:
    flat_spec = science_step.run(
        [flat_file], bias, test_trace, norm_flat, curvature, scatter, mask
    )
    print("FLAT extraction complete")
except Exception as e:
    print(f"FLAT extraction failed: {e}")
    flat_spec = None

# --- STEP 5: Extract from ThAr ---
print("\n=== EXTRACT ThAr ===")
try:
    thar_spec = science_step.run(
        [thar_file], bias, test_trace, norm_flat, curvature, scatter, mask
    )
    print("ThAr extraction complete")
except Exception as e:
    print(f"ThAr extraction failed: {e}")
    thar_spec = None

print("\nDone!")
print(f"Output saved to: {output_dir}")
