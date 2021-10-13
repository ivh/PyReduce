# -*- coding: utf-8 -*-
"""
Simple usage example for PyReduce
Loads a sample UVES dataset, and runs the full extraction
"""

import os.path

import pyreduce
from pyreduce import datasets

# define parameters
instrument = "McDonald"
target = "Vega"
night = None
mode = "CS23"
steps = (
    # "bias",
    # "flat",
    # "orders",
    # "curvature",
    "scatter",
    # "norm_flat",
    # "wavecal",
    # "freq_comb",
    # "science",
    # "continuum",
    # "finalize",
)

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
base_dir = datasets.MCDONALD("/DATA/PyReduce")
input_dir = "raw"
output_dir = "reduced"

# Path to the configuration parameters, that are to be used for this reduction
config = pyreduce.configuration.get_configuration_for_instrument(instrument, plot=2)

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
    # order_range=(0, 25),
)
