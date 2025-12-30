# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7a5"]
# ///
"""
Simple usage example for PyReduce
Loads a sample UVES dataset, and runs the full extraction
"""

from pyreduce.configuration import get_configuration_for_instrument
from pyreduce.instruments.common import create_custom_instrument
from pyreduce.pipeline import Pipeline
from pyreduce.util import start_logging

# Define the path to support files if possible
# otherwise set them to None
# Obviously they are necessary for their respective steps
bpm_mask = "path/to/bpm_mask.fits"
wavecal_file = "path/to/wavecal_file"


# create our custom instrument
instrument = create_custom_instrument(
    "custom", extension=1, mask_file=bpm_mask, wavecal_file=wavecal_file
)

# Override default values
# those can either be fixed values or refer to FITS header keywords
instrument.info["readnoise"] = 1
instrument.info["prescan_x"] = "PRESCAN X"

# For loading the config we specify pyreduce as the source, since this is the default
config = get_configuration_for_instrument("pyreduce")
# Define your own configuration
config["orders"]["degree"] = 5

# Since we can't find the files ourselves (at least not without defining the criteria we are looking for)
# We need to manually define which files go where
files = {"bias": ["file1", "file2"], "flat": ["file3"]}

# We define the path to the output directory
output_dir = "path/to/output"

# (optional) We need to define the log file
log_file = "path/to/log_file.txt"
start_logging(log_file)


# Define other parameter for PyReduce
target = ""
night = "2019-07-21"
channel = ""
steps = (
    "bias",
    "flat",
    "orders",
    "curvature",
    "scatter",
    "norm_flat",
    # "wavecal",
    # "freq_comb",
    "science",
    # "continuum",
    # "finalize",
)

# Call the PyReduce algorithm
pipe = Pipeline.from_files(
    files=files,
    output_dir=output_dir,
    target=target,
    instrument=instrument,
    channel=channel,
    night=night,
    config=config,
    # order_range=order_range,
    steps=steps,
    plot=1,
)
data = pipe.run()
