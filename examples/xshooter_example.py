"""
Simple usage example for PyReduce
Loads a sample UVES dataset, and runs the full extraction
"""

import os.path
import pyreduce
from pyreduce import datasets

# define parameters
instrument = "XShooter"
target = "UX-Ori"
night = None
mode = "NIR"
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
base_dir = datasets.XSHOOTER("/DATA/PyReduce")
input_dir = "raw"
output_dir = "reduced"

config = pyreduce.configuration.get_configuration_for_instrument(instrument, plot=0)
# config["science"]["extraction_method"] = "arc"
# config["science"]["extraction_cutoff"] = 0

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
