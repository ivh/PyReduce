# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7b2"]
# ///
"""
MOSAIC NIR spectrograph example.

Demonstrates fiber bundle tracing and extraction on simulated E2E data.
"""

import os
from os.path import join

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


# --- STEP 1: Mask ---
print("\n=== MASK ===")
mask_step = Mask(*step_args, **step_config("mask"))
mask = mask_step.run()

# No bias for simulated data
bias = None

# --- STEP 2: Trace fibers ---
print("\n=== TRACE ===")
trace_step = OrderTracing(*step_args, **step_config("trace"))
orders, column_range = trace_step.run([flat_file], mask, bias)
print(f"Found {len(orders)} traces")

trace = (orders, column_range)

# --- STEP 3: Prepare flat data ---
print("\n=== FLAT ===")
from astropy.io import fits

flat_data = fits.getdata(flat_file).astype(float)
with fits.open(flat_file) as hdul:
    flat_header = hdul[0].header.copy()
flat = (flat_data, flat_header)

scatter = None
curvature = None

# --- STEP 4: Normalize flat ---
print("\n=== NORM_FLAT ===")
norm_flat_step = NormalizeFlatField(*step_args, **step_config("norm_flat"))
try:
    norm_flat = norm_flat_step.run(flat, trace, scatter, curvature)
    print("Normalized flat complete")
except Exception as e:
    print(f"Norm flat failed: {e}")
    norm_flat = None

# --- STEP 5: Extract from FLAT ---
print("\n=== EXTRACT FLAT ===")
science_step = ScienceExtraction(*step_args, **step_config("science"))
try:
    flat_spec = science_step.run(
        [flat_file], bias, trace, norm_flat, curvature, scatter, mask
    )
    print("FLAT extraction complete")
except Exception as e:
    print(f"FLAT extraction failed: {e}")
    flat_spec = None

# --- STEP 6: Extract from ThAr ---
print("\n=== EXTRACT ThAr ===")
try:
    thar_spec = science_step.run(
        [thar_file], bias, trace, norm_flat, curvature, scatter, mask
    )
    print("ThAr extraction complete")
except Exception as e:
    print(f"ThAr extraction failed: {e}")
    thar_spec = None

print("\nDone!")
print(f"Output saved to: {output_dir}")
