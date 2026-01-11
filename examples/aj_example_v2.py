# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7b3"]
# ///
"""
AJ instrument example using config-based fiber grouping.

The fibers.groups config in AJ/config.yaml handles:
- Organizing 75 fibers into groups A (1-35), cal (37-39), B (40-75)
- Merging each group's traces by averaging
- Selecting which groups to use for each step

Note: This example uses a combined flat (even+odd). The even/odd
flat separation for tracing is a separate concern not yet in config.
"""

import os
from os.path import join

import numpy as np

from pyreduce import util
from pyreduce.configuration import load_config
from pyreduce.instruments.instrument_info import load_instrument
from pyreduce.pipeline import Pipeline

# Parameters
instrument_name = "AJ"
target = "AJ_test"
night = ""
channel = "ALL"
plot = 1

# Handle plot environment variables
if "PYREDUCE_PLOT" in os.environ:
    plot = int(os.environ["PYREDUCE_PLOT"])
plot_dir = os.environ.get("PYREDUCE_PLOT_DIR")
util.set_plot_dir(plot_dir)

# Data location
data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
raw_dir = join(data_dir, "AJ", "raw")
output_dir = join(data_dir, "AJ", "reduced")

# Input files - combine even and odd flats for tracing
file_even = join(raw_dir, "J_FF_even_1s.fits")
file_odd = join(raw_dir, "J_FF_odd_1s.fits")
combined_flat = join(output_dir, "combined_flat.fits")

# Verify input files exist
for fpath in [file_even, file_odd]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")

print(f"Even flat: {file_even}")
print(f"Odd flat: {file_odd}")

# Create combined flat if needed
os.makedirs(output_dir, exist_ok=True)
instrument = load_instrument(instrument_name)

if not os.path.exists(combined_flat):
    print("\nCombining even and odd flats...")
    img_even, head = instrument.load_fits(file_even, channel=channel, extension=0)
    img_odd, _ = instrument.load_fits(file_odd, channel=channel, extension=0)
    # Convert masked arrays to regular arrays
    img_even = np.asarray(img_even)
    img_odd = np.asarray(img_odd)
    img_combined = img_even.astype(np.float64) + img_odd.astype(np.float64)

    from astropy.io import fits

    fits.writeto(combined_flat, img_combined, head, overwrite=True)
    print(f"Saved combined flat to: {combined_flat}")
else:
    print(f"\nUsing existing combined flat: {combined_flat}")

# Load configuration
config = load_config(None, instrument_name, plot)

# Create pipeline - fiber grouping is handled by config.yaml:
#   groups:
#     A: {range: [1, 36], merge: average}
#     cal: {range: [37, 40], merge: average}
#     B: {range: [40, 76], merge: average}
#   use:
#     science: [A, B]
#     wavecal: [cal]
#     norm_flat: all
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
pipe.flat([combined_flat])  # Register flat for norm_flat step
pipe.trace_orders([combined_flat])
pipe.normalize_flat()  # Uses flat registered above

print("\n=== Running Pipeline ===")
results = pipe.run()

print("\n=== Results ===")
orders, column_range = results["trace"]
print(f"Raw traces: {len(orders)}")

if "trace_groups" in results and results["trace_groups"]:
    group_traces, group_cr = results["trace_groups"]
    print(f"Fiber groups: {list(group_traces.keys())}")
    for name, traces in group_traces.items():
        print(f"  {name}: {len(traces)} trace(s)")
