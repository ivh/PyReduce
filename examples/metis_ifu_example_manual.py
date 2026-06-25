# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7"]
# ///
"""
Simple usage example for PyReduce
Loads a simulated METIS dataset, and runs the full extraction
"""

import os
from os.path import join

from pyreduce import datasets
from pyreduce.pipeline import Pipeline
from pyreduce.configuration import load_config

# define parameters
instrument_name = "METIS_IFU"
instrument = "METIS_IFU"
target = ""
night = ""
channel = "3.555_det1"
plot = 1

# Data location: uses $REDUCE_DATA or ~/REDUCE_DATA
base_dir = os.path.join(datasets.get_data_dir(), "METIS")
input_dir = join(base_dir, "raw")
output_dir = join(base_dir, "reduced")

# File paths (simulated data)
flat_file = join(
    input_dir,
    "METIS.IFU_RSRF_RAW.2027-01-25_00_19_34.fits",
)
print(f"FLAT: {flat_file}")

wav_file = join(
    input_dir,
    "METIS.IFU_WAVE_RAW.2027-01-25_00_01_09.fits",
)
print(f"WAVE: {wav_file}")

# Load configuration (with channel for channel-specific settings)
config = load_config(None, instrument_name, plot, channel=channel)

pipe = Pipeline(
    instrument=instrument_name,
    output_dir=output_dir,
    target=target,
    night=night,
    channel=channel,
    config=config,
    # trace_range=(16, 17),
    plot=plot,
)

# Run pipeline steps
pipe.trace([flat_file])
# pipe.curvature([wav_file])

print("\n=== Running Pipeline ===")
results = pipe.run()

# print("\n=== Results ===")
# traces = results["trace"]  # list[Trace]
# print(f"Traces: {len(traces)}")
# for t in traces:
#     print(f"  m={t.m}, height={t.height}, pos={t.pos}, columns={t.column_range}")
