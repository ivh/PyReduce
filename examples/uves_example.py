"""
Main entry point for REDUCE scripts,
default values can be changed as required if reduce is used as a script

Finds input directories, and loops over observation nights and instrument modes

instrument : str
    instrument used for the observation (e.g. UVES, HARPS)
target : str
    the observed star, as named in the folder structure/fits headers
steps : {tuple(str), "all"}, optional
    which steps of the reduction process to perform
    the possible steps are: "bias", "flat", "orders", "norm_flat", "wavecal", "science"
    alternatively set steps to "all", which is equivalent to setting all steps
    Note that the later steps require the previous intermediary products to exist and raise an exception otherwise
"""
import os
import json
import glob
import logging

import numpy as np

import PyReduce

instrument = "UVES"
target = "HD132205"
steps = ("bias", "flat", "orders", "norm_flat", "wavecal", "science", "continuum")

#TODO provide the test data from a server and load dynamically when needed

# some basic settings
# Expected Folder Structure: base_dir/instrument/target/raw/night/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}
# base_dir = "/DATA/ESO_Archive/"
base_dir = "../Test/"
input_dir = base_dir + "{instrument}/{target}/raw/{night}"
output_dir = base_dir + "{instrument}/{target}/reduced/{night}/Reduced_{mode}"

# Setup logging, otherwise default logging will be used
log_file = "logs/%s.log" % target
PyReduce.util.start_logging(log_file)

# load settings for the run & instrument
# config: paramters for the current reduction
# info: constant, instrument specific parameters
with open("settings_%s.json" % instrument.upper()) as f:
    config = json.load(f)
info = PyReduce.instruments.instrument_info.get_instrument_info(instrument)

# TODO: Test settings
config["plot"] = True
config["manual"] = True
modes = [info["modes"][1]]

# start one run per night and mode

# Search the available days
dates = input_dir.format(instrument=instrument, target=target, night="????-??-??")
dates = glob.glob(dates)
dates = [r for r in dates if os.path.isdir(r)]

logging.info("Instrument: %s", instrument)
logging.info("Target: %s", target)
for night in dates:
    night = os.path.basename(night)
    logging.info("Observation Date: %s", night)
    for mode in modes:
        logging.info("Instrument Mode: %s", mode)

        input_dir_night = input_dir.format(
            instrument=instrument, target=target, night=night, mode=mode
        )

        # find input files and sort them by type
        files = glob.glob(os.path.join(input_dir_night, "%s.*.fits" % instrument))
        files += glob.glob(os.path.join(input_dir_night, "%s.*.fits.gz" % instrument))
        files = np.array(files)

        f_bias, f_flat, f_wave, f_order, f_spec = PyReduce.instruments.instrument_info.sort_files(
            files, target, night, instrument, mode, **config
        )
        logging.debug("Bias files:\n%s", str(f_bias))
        logging.debug("Flat files:\n%s", str(f_flat))
        logging.debug("Wavecal files:\n%s", str(f_wave))
        logging.debug("Orderdef files:\n%s", str(f_order))
        logging.debug("Science files:\n%s", str(f_spec))

        if isinstance(f_spec, dict):
            for key, _ in f_spec.items():
                fb, ff, fw, fo, fs = (
                    f_bias[key],
                    f_flat[key],
                    f_wave[key],
                    f_order[key],
                    f_spec[key],
                )
        else:
            fb, ff, fw, fo, fs = f_bias, f_flat, f_wave, f_order, f_spec

        PyReduce.reduce.run_steps(
            fb,
            ff,
            fw,
            fo,
            fs,
            output_dir,
            target,
            instrument,
            mode,
            night,
            config,
            info,
            steps=steps,
        )
