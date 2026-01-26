"""
MOSAIC VIS spectrograph example.

The VIS detector is a 4-quadrant mosaic (12788x12394 pixels total).
Each quadrant is processed as a separate channel:
- VIS1: lower-left quadrant
- VIS2: lower-right quadrant
- VIS3: upper-left quadrant
- VIS4: upper-right quadrant

This example processes VIS1. Run with different channel values for other quadrants.
"""

import os
from os.path import join

from pyreduce import util
from pyreduce.configuration import load_config
from pyreduce.pipeline import Pipeline

# Parameters
# Change channel to VIS2, VIS3, or VIS4 for other quadrants
instrument_name = "MOSAIC"
target = "MOSAIC_VIS"
night = ""
channel = "VIS3"
plot = 2

# Data location
data_dir = "/disk/miri-b1/jeand/mosaic/virtualmosaic/simdata"
base_dir = join(data_dir, "VIS")
output_dir = join(data_dir, "reduced", channel)

# Handle plot environment variables
if "PYREDUCE_PLOT" in os.environ:
    plot = int(os.environ["PYREDUCE_PLOT"])
plot_dir = join(data_dir, "pyreduce_plots", channel)
util.set_plot_dir(plot_dir)
# to not show plots during processing:
# PYREDUCE_PLOT_SHOW=off uv run...

# File paths (simulated data)
flat_file = join(
    base_dir,
    "E2E_as_built_FLAT_DIT_20s_MOSAIC_VIS_c01_FOCAL_PLANE_000.fits",
)
thar_file = join(
    base_dir,
    "E2E_as_built_ThAr_DIT_20s_MOSAIC_VIS_c01_FOCAL_PLANE.fits",
)

# Verify files exist
for fpath in [flat_file, thar_file]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")

print(f"FLAT: {flat_file}")
print(f"ThAr: {thar_file}")

# Load configuration
config = load_config(None, instrument_name, channel=channel)

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

# Run pipeline steps
pipe.trace([flat_file])
pipe.curvature([thar_file])
pipe.extract([thar_file])

print("\n=== Running Pipeline ===")
results = pipe.run()

print("\n=== Results ===")
traces = results["trace"]  # list[Trace]
print(f"Traces: {len(traces)}")
for t in traces[:3]:
    print(f"  m={t.m}, fiber={t.fiber}, columns={t.column_range}")
