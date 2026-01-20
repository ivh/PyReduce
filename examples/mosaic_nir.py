# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7b3"]
# ///
"""
MOSAIC NIR spectrograph example using preset slit function from flat.

Demonstrates single-pass ThAr extraction using the slit function
from norm_flat instead of iteratively solving for it.
"""

import os
from os.path import join

import numpy as np

from pyreduce import util
from pyreduce.configuration import load_config
from pyreduce.extract import extract

# Parameters
instrument_name = "MOSAIC"
target = "MOSAIC_NIR"
night = ""
channel = "NIR"
plot = 2  # show swath progress

# Handle plot environment variables
if "PYREDUCE_PLOT" in os.environ:
    plot = int(os.environ["PYREDUCE_PLOT"])
plot_dir = os.environ.get("PYREDUCE_PLOT_DIR")
util.set_plot_dir(plot_dir)

# Data location
data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
base_dir = join(data_dir, "MOSAIC", "REF_E2E", "NIR")
output_dir = join(data_dir, "MOSAIC", "reduced", "NIR")

# File paths (simulated data)
flat_file = join(
    base_dir,
    "E2E_FLAT_DIT_20s_MOSAIC_2Cam_c01",
    "E2E_FLAT_DIT_20s_MOSAIC_2Cam_c01_FOCAL_PLANE.fits",
)
thar_file = join(
    base_dir,
    "E2E_ThAr_DIT_20s_MOSAIC_2Cam_c01",
    "E2E_ThAr_DIT_20s_MOSAIC_2Cam_c01_FOCAL_PLANE.fits",
)

# Verify files exist
for fpath in [flat_file, thar_file]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")

print(f"FLAT: {flat_file}")
print(f"ThAr: {thar_file}")

# Load configuration and instrument
config = load_config(None, instrument_name, plot)
from pyreduce.instruments import instrument_info

instrument = instrument_info.load_instrument(instrument_name)

# File prefix for saved results (match existing files)
prefix = "mosaic_nir"

# Load saved results from previous pipeline run
print("\n=== Loading saved results ===")

trace_file = join(output_dir, f"{prefix}.ord_default.npz")
norm_file = join(output_dir, f"{prefix}.flat_norm.npz")
curve_file = join(output_dir, f"{prefix}.curve.npz")

# Load trace (use grouped traces that norm_flat used)
trace_data = np.load(trace_file)
group_names = trace_data["group_names"]
all_traces = [trace_data[f"group_{name}_traces"] for name in group_names]
all_cr = [trace_data[f"group_{name}_cr"] for name in group_names]
orders = np.vstack(all_traces)
column_range = np.vstack(all_cr)
print(f"Loaded {len(orders)} grouped traces from {trace_file}")

# Load norm_flat (with slitfunc)
norm_data = np.load(norm_file, allow_pickle=True)
slitfunc_list = list(norm_data["slitfunc"])
slitfunc_meta = norm_data["slitfunc_meta"].item()
print(f"Loaded {len(slitfunc_list)} slit functions from {norm_file}")
print(f"Slitfunc meta: {slitfunc_meta}")

# Load curvature if available
if os.path.exists(curve_file):
    curve_data = np.load(curve_file)
    p1, p2 = curve_data["p1"], curve_data["p2"]
    print(f"Loaded curvature from {curve_file}")
else:
    p1, p2 = None, None
    print("No curvature file found, using p1=p2=None")

# Now extract ThAr using preset slit function
print("\n=== Extracting ThAr with preset slit function ===")

# Load ThAr image
thar_img, thar_head = instrument.load_fits(thar_file, channel)

# Use the grouped traces (same as norm_flat)
selected_orders = orders
selected_cr = column_range

# Extraction parameters from slitfunc_meta
osample = slitfunc_meta["osample"]
extraction_height = slitfunc_meta["extraction_height"]

print(f"Extracting {len(selected_orders)} traces with preset slitfunc")
print(f"  osample={osample}, extraction_height={extraction_height}")

# Extract using the extract() function with preset_slitfunc
spectrum, uncertainties, slitfunc_out, column_range_out = extract(
    thar_img,
    selected_orders,
    column_range=selected_cr,
    extraction_type="optimal",
    extraction_height=extraction_height,
    p1=p1,
    p2=p2,
    plot=plot,
    plot_title="ThAr with preset slitfunc",
    # Extraction kwargs
    osample=osample,
    lambda_sf=0.1,
    lambda_sp=0,
    maxiter=1,  # single pass
    reject_threshold=0,  # no rejection when using preset
    gain=thar_head.get("e_gain", 1),
    readnoise=thar_head.get("e_readn", 0),
    # The preset slit function from norm_flat
    preset_slitfunc=slitfunc_list,
)

print(f"\nExtracted spectra shape: {spectrum.shape}")
print(f"Non-zero values: {np.count_nonzero(spectrum)}")
