# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7a5"]
# ///
"""
Simple usage example for PyReduce
Loads a HARPS-N dataset, and runs the extraction
"""

import os

from pyreduce import datasets
from pyreduce.pipeline import Pipeline

# define parameters
instrument = "HARPN"
target = ""
night = ""
channel = "HARPN"
steps = (
    # "bias",
    # "flat",
    # "orders",
    # "norm_flat",
    # "wavecal_master",
    "wavecal",
    # "science",
    # "continuum",
    # "finalize",
)

# Data location: uses $REDUCE_DATA or ~/REDUCE_DATA
base_dir = os.path.join(datasets.get_data_dir(), "HARPN")
input_dir = "raw"
output_dir = "reduced/{channel}"

Pipeline.from_instrument(
    instrument,
    target,
    night=night,
    channel=channel,
    steps=steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    # order_range=(0, 25),
    plot=1,
).run()
