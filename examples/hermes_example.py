"""Reducing HERMES data with PyReduce.

HERMES (High Efficiency and Resolution Mercator Echelle Spectrograph)
at the Mercator telescope, La Palma.

Fiber modes:
  HRF - High Resolution Fibre (default, R~85000)

File types:
  BIAS     - bias frames
  HRF_FF   - flatfields (Tungsten lamp)
  HRF_TH   - wavelength calibration (ThAr + Ne lamps)
  HRF_OBJ  - science/object spectra
"""

import os

from pyreduce.pipeline import Pipeline

base_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))

steps = (
    "bias",
    "flat",
    "trace",
    # "curvature",
    "norm_flat",
    "wavecal_master",
    # "wavecal_init",
    # "wavecal",
    "science",
    # "continuum",
    # "finalize",
)

data = Pipeline.from_instrument(
    instrument="HERMES",
    target="HD 82106",
    night="2026-04-12",
    channel="",
    steps=steps,
    base_dir=base_dir,
    input_dir=os.path.join(base_dir, "HERMES"),
    plot=1,
).run()
