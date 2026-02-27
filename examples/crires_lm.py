"""
CRIRES+ L-band example
Runs trace, flat, and norm_flat on L3340 flat-field frames.
"""

import os

from pyreduce import datasets
from pyreduce.pipeline import Pipeline

instrument = "CRIRES_PLUS"
target = ""
night = ""
channel = "L3340_det1"
steps = (
    "flat",
    "trace",
    "norm_flat",
)

base_dir = os.path.join(datasets.get_data_dir(), "CRIRES_LM")
input_dir = ""
output_dir = "reduced"

pipe = Pipeline.from_instrument(
    instrument,
    target,
    night=night,
    channel=channel,
    steps=steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    plot=1,
)
pipe._data["bias"] = None
pipe._data["mask"] = None
pipe._data["scatter"] = None
pipe.run()
