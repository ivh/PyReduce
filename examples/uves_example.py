# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7a5"]
# ///
"""
Simple usage example for PyReduce
Loads a sample UVES dataset, and runs the full extraction
"""

from pyreduce import datasets
from pyreduce.pipeline import Pipeline

# define parameters
instrument = "UVES"
target = "HD[- ]?132205"
night = "2010-04-01"
channel = "middle"
steps = (
    "bias",
    "flat",
    "orders",
    "norm_flat",
    "curvature",
    "wavecal",
    "science",
    "continuum",
    "finalize",
)

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
base_dir = datasets.UVES()  # Uses $REDUCE_DATA or ~/REDUCE_DATA
input_dir = "raw/"
output_dir = "reduced/{night}/{channel}"

Pipeline.from_instrument(
    instrument,
    target,
    night=night,
    channel=channel,
    steps=steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    order_range=(1, 21),
    plot=1,
).run()
