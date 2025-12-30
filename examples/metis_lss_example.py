# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7a5"]
# ///
"""
Simple usage example for PyReduce
Loads a simulated METIS dataset, and runs the full extraction
"""

import os

from pyreduce import datasets
from pyreduce.pipeline import Pipeline

# define parameters
instrument = "METIS_LSS"
target = ""
night = ""
channel = "LSS_M"
steps = (
    # "bias",
    "flat",
    "orders",
    "curvature",
    # "scatter",
    # "norm_flat",
    "wavecal_master",
    # "wavecal_init",
    "wavecal",
    # "rectify",
    # "science",
    # "continuum",
    # "finalize",
)

# Data location: uses $REDUCE_DATA or ~/REDUCE_DATA
base_dir = os.path.join(datasets.get_data_dir(), "METIS")
input_dir = "raw"
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
    # order_range=(16, 17),
    plot=1,
).run()
