# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7a5"]
# ///
"""
Simple usage example for PyReduce
Loads a sample NIRSPEC dataset, and runs the extraction
"""

from pyreduce import datasets
from pyreduce.pipeline import Pipeline

# define parameters
instrument = "NIRSPEC"
target = "GJ1214"
night = ""
channel = "NIRSPEC"
steps = (
    "bias",
    "flat",
    "orders",
    "norm_flat",
    "wavecal",
    "freq_comb",
    # "curvature",
    "science",
    # "continuum",
    "finalize",
)

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
base_dir = datasets.KECK_NIRSPEC()
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
).run()
