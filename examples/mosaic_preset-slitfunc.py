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
from pyreduce.trace_model import load_traces

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

trace_file = join(output_dir, f"{prefix}.traces.fits")
norm_file = join(output_dir, f"{prefix}.flat_norm.npz")

# Load traces from FITS file (includes curvature if available)
trace_list, _ = load_traces(trace_file)
print(f"Loaded {len(trace_list)} traces from {trace_file}")

# Convert to arrays for extract()
traces = np.array([t.pos for t in trace_list])
column_range = np.array([t.column_range for t in trace_list])

# Get curvature from traces (if available)
# The slit curvature is now stored in each Trace object
has_curvature = trace_list[0].slit is not None
if has_curvature:
    # For the extract() function, we need p1, p2 as arrays (ntrace, 2)
    # Extract the linear and quadratic curvature terms
    p1 = np.zeros(len(trace_list))
    p2 = np.zeros(len(trace_list))
    for i, t in enumerate(trace_list):
        if t.slit is not None and t.slit.shape[0] >= 2:
            # slit[1, :] contains coefficients for the y^1 term (linear tilt)
            # Take the constant term (value at x=0) as p1
            p1[i] = t.slit[1, -1] if len(t.slit[1]) > 0 else 0
            if t.slit.shape[0] >= 3:
                p2[i] = t.slit[2, -1] if len(t.slit[2]) > 0 else 0
    print(f"Loaded curvature from traces (p1 range: [{p1.min():.4f}, {p1.max():.4f}])")
else:
    p1, p2 = None, None
    print("No curvature data in traces")

# Load norm_flat (with slitfunc)
norm_data = np.load(norm_file, allow_pickle=True)
slitfunc_list = list(norm_data["slitfunc"])
slitfunc_meta = norm_data["slitfunc_meta"].item()
print(f"Loaded {len(slitfunc_list)} slit functions from {norm_file}")
print(f"Slitfunc meta: {slitfunc_meta}")

# Now extract ThAr using preset slit function
print("\n=== Extracting ThAr with preset slit function ===")

# Load ThAr image
thar_img, thar_head = instrument.load_fits(thar_file, channel)

# Extraction parameters from slitfunc_meta
osample = slitfunc_meta["osample"]
extraction_height = slitfunc_meta["extraction_height"]

print(f"Extracting {len(traces)} traces with preset slitfunc")
print(f"  osample={osample}, extraction_height={extraction_height}")

# Extract using the extract() function with preset_slitfunc
spectrum, uncertainties, slitfunc_out, column_range_out = extract(
    thar_img,
    traces,
    column_range=column_range,
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
    maxiter=2,  # need 2 for rejection to take effect
    reject_threshold=6,  # outlier rejection in sigma
    gain=thar_head.get("e_gain", 1),
    readnoise=thar_head.get("e_readn", 0),
    # The preset slit function from norm_flat
    preset_slitfunc=slitfunc_list,
)

print(f"\nExtracted spectra shape: {spectrum.shape}")
print(f"Non-zero values: {np.count_nonzero(spectrum)}")
