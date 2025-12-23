# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7"]
# ///
"""
Simple usage example for PyReduce
Loads a CRIRES+ dataset, and runs the extraction
"""

from pyreduce.configuration import get_configuration_for_instrument
from pyreduce.pipeline import Pipeline

# define parameters
instrument = "Crires_plus"
target = ""
night = ""
arm = "J1228_Open_det1"
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

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
base_dir = "/DATA/datasets/CRIRES"
input_dir = base_dir
output_dir = base_dir + "/reduced/"

# Path to the configuration parameters, that are to be used for this reduction

config = get_configuration_for_instrument(
    instrument,
    plot=1,
)

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
    allow_calibration_only=True,
    # order_range=(0, 4),
).run()
