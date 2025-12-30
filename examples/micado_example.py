# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7a5"]
# ///
"""
Simple usage example for PyReduce
Loads a ScopeSim simulated MICADO dataset (with updated spectral layout and updated line lists), and runs the full extraction.
"""

import pyreduce

# define parameters
instrument = "MICADO"
target = ""
night = ""
channel = ""
steps = (
    # "bias",
    "flat",
    "orders",
    "curvature",
    # # "scatter",
    # "norm_flat",
    "wavecal",
    # "science",
    # "continuum",
    # "finalize",
)

# Some basic settings
# Expected Folder Structure: base_dir/datasets/MICADO/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# Define the path for the base, input and output directories
# The data (with fixed header keywords) can be fetched from https://www.dropbox.com/sh/e3lnvtkmyjveajk/AABPHxeUdDO5AnkWCAjbM0e1a?dl=0 and stored in input_dir

# PC
base_dir = "/media/data/Dropbox/Dropbox/WORKING/iMICADO/Working/WORKING_PyReduce/DATA/datasets/MICADO/"  # an example path which you should change to your prefered one

input_dir = "raw_new/HK/"
output_dir = "reduced_new/"

config = pyreduce.configuration.get_configuration_for_instrument(instrument)


# Configuring parameters of individual steps here overwrites those defined  in the settings_MICADO.json file.
# Once you are satisfied with a certain parameter, you can update it in settings_MICADO.json.


# config["orders"]["noise"] = 100
# config["curvature"]["extraction_width"] = 350 # curvature can still be improved with this and the following parameters
# config["curvature"]["peak_threshold"] =10
# config["curvature"]["peak_width"] =2 #CHECK 6 also works and detects one less line
# config["curvature"]["window_width"] = 5
# config["wavecal"]["extraction_width"] = 350

# NOTE: micado.thar_master.fits (created and controlled by wavecal_master) is NOT overwritten if any parameter in the steps in or before it are changed. Thus it has to be deleted before running PyReduce again.

pyreduce.reduce.main(
    instrument,
    target,
    night,
    channel,
    steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    configuration=config,
    order_range=(
        3,
        4,
    ),  # for MICADO, when one order is on the detector (currently detector 5 of the HK band)
    plot=1,
)
