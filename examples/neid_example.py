# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7b3"]
# ///
"""
NEID reduction example

NEID is a fiber-fed, high-resolution (R~110,000) spectrograph on the
WIYN 3.5m telescope at Kitt Peak. L0 data has 16 amplifiers that are
automatically assembled during loading.

This example reduces HD 4628 observations from night 2024-09-19.
Note: Observations before 12:00 UTC belong to the previous night.
"""

import os

from pyreduce.configuration import get_configuration_for_instrument
from pyreduce.pipeline import Pipeline

# Define parameters
instrument = "NEID"
target = "HD 4628"
night = "2024-09-19"  # Observations before 12:00 UTC belong to previous night
channel = "HR"

# Reduction steps to run
# Start with basic steps; add wavecal/science once calibration files are set up
steps = (
    "flat",
    "trace",
    # "curvature",
    # "norm_flat",
    # "wavecal_master",
    # "wavecal_init",
    # "wavecal",
    # "science",
    # "continuum",
    # "finalize",
)

# Data paths
# Set REDUCE_DATA environment variable or modify base_dir
base_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
base_dir = os.path.join(base_dir, "NEID")
input_dir = ""  # Files directly in base_dir
output_dir = "reduced"

# Load default configuration
config = get_configuration_for_instrument(instrument)

# Run the pipeline
if __name__ == "__main__":
    Pipeline.from_instrument(
        instrument,
        target,
        night=night,
        channel=channel,
        steps=steps,
        base_dir=base_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        configuration=config,
        plot=1,
    ).run()
