"""
Simple usage example for PyReduce
Loads a sample UVES dataset, and runs the full extraction
"""

import os.path
import pyreduce
from pyreduce import datasets
from pyreduce.configuration import get_configuration_for_instrument


# define parameters
instrument = "Crires_plus"
target = None
night = "2019-07-21"
mode = "J/2/3_Open"
steps = (
    # "bias",
    # "flat",
    # "orders",
    "curvature",
    # "scatter",
    "norm_flat",
    "wavecal",
    "freq_comb",
    # "science",
    # "continuum",
    # "finalize",
)

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
base_dir = "/DATA/ESO/CRIRES+"
input_dir = "tdata/CD13/UNCLASSIFIED"
output_dir = "reduced/{mode}/"

# Path to the configuration parameters, that are to be used for this reduction

config = get_configuration_for_instrument(instrument, plot=1)

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
    order_range=(0, 4),
)
