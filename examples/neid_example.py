"""
Simple usage example for PyReduce
Loads a sample UVES dataset, and runs the full extraction
"""

import pyreduce

# define parameters
instrument = "NEID"
# target = "HD 152843"
target = ""
night = ""
arm = "NEID"
steps = (
    #  "bias",
    # "flat",
    # "orders",
    # "norm_flat",
    # "wavecal_master",
    "wavecal",
    #    "science",
    #    "continuum",
    #    "finalize",
)

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
# base_dir = datasets.HARPS("/DATA/PyReduce")
base_dir = "/home/tom/pipes/neid_data"
input_dir = "raw"
output_dir = "reduced_{arm}"

# instrument = HARPS()
# files = instrument.find_files(base_dir + "/" + input_dir)
# ev = instrument.get_expected_values(None, None, "red", None, True)
# files = instrument.apply_filters(files, ev)

# Path to the configuration parameters, that are to be used for this reduction
config = pyreduce.configuration.get_configuration_for_instrument(instrument)

pyreduce.reduce.main(
    instrument,
    target,
    night,
    arm,
    steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    configuration=config,
    # order_range=(0, 25),
    plot=1,
)
