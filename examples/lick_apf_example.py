# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7a5"]
# ///
"""
Simple usage example for PyReduce
Loads a Lick APF dataset, and runs the extraction
"""

from pyreduce import datasets
from pyreduce.pipeline import Pipeline

# define parameters
instrument = "LICK_APF"
target = "KIC05005618"
night = None
channel = ""
steps = (
    "bias",
    # "flat",
    # "orders",
    # "norm_flat",
    # "wavecal",
    # "curvature",
    "science",
    # "continuum",
    # "finalize",
)

# Data location: uses $REDUCE_DATA or ~/REDUCE_DATA
base_dir = datasets.LICK_APF()
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
    # order_range=(0, 25),
    plot=0,
).run()
