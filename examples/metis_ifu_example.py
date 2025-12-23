# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7"]
# ///
"""
Simple usage example for PyReduce
Loads a simulated METIS dataset, and runs the full extraction
"""

import pyreduce

# define parameters
instrument = "METIS_IFU"
target = ""
night = ""
arm = "NOMINAL"  # LSS_M (settings_metis.json is now optimized for LSS_M arm)
steps = (
    # "bias",
    # "flat",
    "orders",
    "curvature",
    # "scatter",
    # "norm_flat",
    # "wavecal_master",
    # # "wavecal_init",
    # "wavecal",
    # "rectify",
    # "science",
    # "continuum",
    # "finalize",
)

# some basic settings
# Expected Folder Structure: base_dir/datasets/METIS/*.fits.gz or *.fits
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# Define the path for the base, input and output directories
# The da://neon.physics.uu.se/metis/lms_pinholes.fitsta can be fetched from https://www.dropbox.com/sh/h1dz80vsw4lwoel/AAAqJD_FGDGC-t12wgnPXVR8a?dl=0 and stored in /raw/

# laptop
base_dir = "/Users/Nadeen/Dropbox/WORKING/iMETIS/Working/WORKING_PyReduce/DATA/datasets/METIS/"  # an example path which you should change to your prefereed one
# PC
# base_dir ="/media/data/Dropbox/Dropbox/WORKING/iMETIS/Working/WORKING_PyReduce/DATA/datasets/METIS/"


input_dir = "raw/"
output_dir = "reduced/"

config = pyreduce.configuration.get_configuration_for_instrument(instrument)


# Configuring parameters of individual steps here overwrites those defined in the the settings_METIS.json file.
# Once you are satisfied with the chosen parameter, you can update it in settings_METIS.json.

# config["orders"]["noise"] = 120

# config["curvature"]['dimensionality']= '1D'
# config["curvature"]['curv_degree']= 2
# config["curvature"]["extraction_width"] = 0.7700
# config["curvature"]["peak_threshold"] = 0.9725
# config["curvature"]["peak_width"] = 1# 1 worked for lband
# config["curvature"]["window_width"] = 1 #  2 worked for lband
# config["curvature"]["degree"] = 2 #


# # config["wavecal"]["extraction_width"] = 0.7825 # 0.7325


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
    # order_range=(16, 17),  #(16, 17) # I had to change it inside reduce.py becuase it does the fix_column_range for all detected orders > outside of image
    plot=1,
)
