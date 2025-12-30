# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7a5"]
# ///
"""
Simple usage example for PyReduce
Loads a CRIRES+ dataset, and runs the extraction
"""

import os

from pyreduce import datasets
from pyreduce.pipeline import Pipeline

# define parameters
instrument = "CRIRES_PLUS"
target = ""
night = ""
channel = "J1228_det1"
steps = (
    # "bias",
    # "flat",
    # "orders",
    # "curvature",
    # "scatter",
    # "norm_flat",
    # "wavecal_master",
    # "wavecal_init",
    # "wavecal",
    # "freq_comb_master",
    "freq_comb",
    # "science",
    # "continuum",
    # "finalize",
)

# Data location: uses $REDUCE_DATA or ~/REDUCE_DATA
base_dir = os.path.join(datasets.get_data_dir(), "CRIRES")
input_dir = ""
output_dir = "reduced"

Pipeline.from_instrument(
    instrument,
    target,
    night=night,
    channel=channel,
    steps=steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    allow_calibration_only=True,
    # order_range=(0, 4),
).run()
