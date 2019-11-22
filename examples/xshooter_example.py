"""
Simple usage example for PyReduce
Loads a sample UVES dataset, and runs the full extraction
"""

import os.path
import pyreduce


# define parameters
instrument = "XShooter"
target = "UX-Ori"
night = ""
mode = "NIR"
steps = (
    # "bias",
    # "flat",
    # "orders",
    # "scatter",
    # "norm_flat",
    "curvature",
    # "wavecal",
    "science",
    # "continuum",
    # "finalize",
)

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
base_dir = "/DATA/ESO/XShooter"
input_dir = "raw"
output_dir = "reduced"

config = pyreduce.configuration.get_configuration_for_instrument(instrument, plot=1)

pyreduce.reduce.main(
    instrument,
    target,
    night,
    mode,
    steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    configuration=config,
    order_range=(0, 15),
)
