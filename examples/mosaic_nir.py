"""
MOSAIC NIR spectrograph example using config-based fiber grouping.

The fibers.bundles config in MOSAIC/config.yaml handles:
- Organizing 630 fibers into 90 bundles of 7
- Selecting center fiber from each bundle for extraction

This replaces the manual gap detection in the original mosaic_example.py.
"""

import os
from os.path import join

from pyreduce import util
from pyreduce.configuration import load_config
from pyreduce.pipeline import Pipeline

# Parameters
instrument_name = "MOSAIC"
target = "MOSAIC_NIR"
night = ""
# NIR modes (one 4096x4096 detector each): J_LR (c01), H_LR (c02), H_HR (c03)
channel = "J_LR"
plot = 1

# Handle plot environment variables
if "PYREDUCE_PLOT" in os.environ:
    plot = int(os.environ["PYREDUCE_PLOT"])
plot_dir = os.environ.get("PYREDUCE_PLOT_DIR")
util.set_plot_dir(plot_dir)

# Data location
data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
base_dir = join(data_dir, "MOSAIC", "E2E_june26", "NIR")
output_dir = join(data_dir, "MOSAIC", "reduced", channel)

# File paths (simulated data). The "cNN" cube index encodes the mode:
# c01 -> J_LR, c02 -> H_LR, c03 -> H_HR (verify against ESO INS MODE header).
cube = {"J_LR": "c01", "H_LR": "c02", "H_HR": "c03"}[channel]
flat_file = join(
    base_dir, f"E2E_FLAT_DIT_20s_MOSAIC_2Cam_{cube}_FOCAL_PLANE_000_REF.fits"
)
thar_file = join(
    base_dir, f"E2E_ThAr_DIT_20s_MOSAIC_2Cam_{cube}_FOCAL_PLANE_000_REF.fits"
)

# Verify files exist
for fpath in [flat_file, thar_file]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")

print(f"FLAT: {flat_file}")
print(f"ThAr: {thar_file}")

# Load configuration (with channel for channel-specific settings)
config = load_config(None, instrument_name, plot, channel=channel)

# Create pipeline - fiber grouping is handled by config.yaml
pipe = Pipeline(
    instrument=instrument_name,
    output_dir=output_dir,
    target=target,
    channel=channel,
    night=night,
    config=config,
    plot=plot,
)

# Trace + curvature are slow; compute them once, then reload to iterate fast
# on wavecal/extract (cf. examples/andes_yjh.py).
#   First run:  LOAD_TRACE=LOAD_CURVE=False, STOP_AFTER_CURVATURE=True
#   Then flip:  LOAD_TRACE=LOAD_CURVE=True,  STOP_AFTER_CURVATURE=False
LOAD_TRACE = True
LOAD_CURVE = True
STOP_AFTER_CURVATURE = False

if LOAD_TRACE:
    trace_objects = pipe._run_step("trace", None, load_only=True)
    print(f"Loaded {len(trace_objects)} traces")
else:
    pipe.trace([flat_file])

if not LOAD_CURVE:
    pipe.curvature([thar_file])

if STOP_AFTER_CURVATURE:
    pipe.run()
    print("Cached trace + curvature; exiting.")
    raise SystemExit(0)

# Per-bundle wavelength guess (wavelength_range_j_lr.yaml) is used automatically
# by wavecal_init via the instrument's get_wavelength_range_per_bundle.
pipe.wavecal_master([thar_file])
pipe.wavecal_init()
pipe.wavecal()
pipe.extract([thar_file])

print("\n=== Running Pipeline ===")
results = pipe.run()
