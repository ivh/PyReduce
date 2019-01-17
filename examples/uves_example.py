"""
Simple usage example for PyReduce
Loads a sample UVES dataset, and runs the full extraction
"""

import pyreduce
import datasets


# define parameters
instrument = "UVES"
target = "HD132205"
night = "2010-04-02"
mode = "middle"
steps = ("bias", "flat", "orders", "norm_flat", "wavecal", "science", "continuum")

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
base_dir = datasets.UVES_HD132205()
input_dir = "datasets/{target}/"
output_dir = "reduced/{target}/{night}/{mode}"

pyreduce.reduce.main(
    instrument,
    target,
    night,
    mode,
    steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
)
