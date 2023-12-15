# -*- coding: utf-8 -*-
"""
Simple usage example for PyReduce
Loads a sample UVES dataset, and runs the full extraction
"""

import os.path

import pyreduce
from pyreduce import datasets

# define parameters
instrument = "NIRSPEC"
target = "GJ1214"
night = ""
mode = "NIRSPEC"
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

# Path to the configuration parameters, that are to be used for this reduction

pyreduce.reduce.main(
    instrument,
    target,
    night,
    mode,
    steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    # order_range=(0, 25),
)
