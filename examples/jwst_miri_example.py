# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7"]
# ///
"""
Simple usage example for PyReduce
Loads a JWST MIRI dataset, and runs the extraction
"""

from pyreduce.configuration import get_configuration_for_instrument
from pyreduce.pipeline import Pipeline

# define parameters
instrument = "JWST_MIRI"
target = None
night = None
arm = "LRS_SLITLESS"
steps = (
    "bias",
    "flat",
    "orders",
    "norm_flat",
    # "wavecal",
    # "curvature",
    "science",
    # "continuum",
    # "finalize",
)

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
base_dir = "/DATA/JWST/MIRI/MIRIsim/"
input_dir = "raw"
output_dir = "reduced"

# Path to the configuration parameters, that are to be used for this reduction
config = get_configuration_for_instrument(instrument)

Pipeline.from_instrument(
    instrument,
    target,
    night=night,
    arm=arm,
    steps=steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    configuration=config,
    # order_range=(0, 25),
    plot=1,
).run()
