# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7b3"]
# ///
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
channel = "NIR"
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

# Run pipeline steps
# The fibers.bundles config automatically:
# - Groups 630 traces into 90 bundles of 7
# - Selects center fiber from each bundle
# - Uses grouped traces for curvature and science steps
pipe.trace_orders([flat_file])
# pipe.curvature([thar_file])
# pipe.flat([flat_file])
# pipe.normalize_flat()
# pipe.extract([flat_file])

print("\n=== Running Pipeline ===")
results = pipe.run()

print("\n=== Results ===")
orders, column_range = results["trace"]
print(f"Raw traces: {len(orders)}")

if "trace_groups" in results and results["trace_groups"]:
    group_traces, group_cr = results["trace_groups"]
    print(
        f"Fiber groups: {list(group_traces.keys())[:5]}... ({len(group_traces)} total)"
    )
