# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7a5"]
# ///
"""
Simple usage example for PyReduce
Loads a sample XSHOOTER dataset, and runs the extraction
"""

from pyreduce import datasets
from pyreduce.configuration import get_configuration_for_instrument
from pyreduce.pipeline import Pipeline

# define parameters
instrument = "XShooter"
target = "UX-Ori"
night = None
channel = "NIR"
steps = (
    # "bias",
    # "flat",
    # "orders",
    # "scatter",
    # "norm_flat",
    # "curvature",
    "wavecal",
    "science",
    # "continuum",
    # "finalize",
)

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
# change the location (as set in datasets() to some folder of you choice)
# or dont pass a path to use the local directory
base_dir = datasets.XSHOOTER()  # Uses $REDUCE_DATA or ~/REDUCE_DATA
input_dir = "raw"
output_dir = "reduced"

config = get_configuration_for_instrument(instrument)
# config["science"]["extraction_method"] = "arc"
# config["science"]["extraction_cutoff"] = 0

Pipeline.from_instrument(
    instrument,
    target,
    night=night,
    channel=channel,
    steps=steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    configuration=config,
    order_range=(0, 15),
    plot=0,
).run()
