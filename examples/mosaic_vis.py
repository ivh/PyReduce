"""
MOSAIC VIS spectrograph example.

Each VIS mode (B_LR, B1_HR, B2_HR, R_LR, R1_HR, R2_HR) is a single
12788x12394 image stitching four slightly-misaligned detectors in a 2x2
mosaic. Each detector is reduced independently as its own channel:

    <mode>_LL  lower-left      <mode>_LR  lower-right
    <mode>_UL  upper-left      <mode>_UR  upper-right

The mode half of the channel selects which files belong to it (via the
ESO INS MODE header); the quadrant half selects the detector crop.

Note: as of the June-2026 release only a single VIS frame is available
locally. Point base_dir at the VIS data and confirm the cube->mode mapping
from the ESO INS MODE header before running.
"""

import os
from os.path import join

from pyreduce import util
from pyreduce.configuration import load_config
from pyreduce.pipeline import Pipeline

instrument_name = "MOSAIC"
target = "MOSAIC_VIS"
night = ""
plot = 2

# Pick one mode; its four detector quadrants are processed in turn.
mode = "R1_HR"
quadrants = [f"{mode}_{q}" for q in ("LL", "LR", "UL", "UR")]

# Raw E2E focal-plane frames use a "cNN" cube index that maps to a mode via
# the ESO INS MODE header (e.g. c04 -> R1_HR). Confirm for each mode.
cube = "c04"
data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
base_dir = join(data_dir, "MOSAIC", "E2E_june26")
flat_file = join(
    base_dir, f"E2E_FLAT_DIT_20s_MOSAIC_VIS_{cube}_FOCAL_PLANE_000_REF.fits"
)
thar_file = join(
    base_dir, f"E2E_ThAr_DIT_20s_MOSAIC_VIS_{cube}_FOCAL_PLANE_000_REF.fits"
)
sky_file = join(
    base_dir, f"E2E_SKY_DIT_150s_MOSAIC_VIS_{cube}_FOCAL_PLANE_001_REF.fits"
)

for fpath in [flat_file, thar_file]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")

print(f"FLAT: {flat_file}")
print(f"ThAr: {thar_file}")
print(f"Sky:  {sky_file}")

if "PYREDUCE_PLOT" in os.environ:
    plot = int(os.environ["PYREDUCE_PLOT"])

for channel in quadrants:
    output_dir = join(data_dir, "MOSAIC", "reduced", channel)
    plot_dir = join(data_dir, "MOSAIC", "pyreduce_plots", channel)
    util.set_plot_dir(plot_dir)

    config = load_config(None, instrument_name, channel=channel)

    pipe = Pipeline(
        instrument=instrument_name,
        output_dir=output_dir,
        target=target,
        channel=channel,
        night=night,
        config=config,
        plot=plot,
    )

    pipe.trace([flat_file])
    pipe.curvature([thar_file])
    pipe.wavecal_master([thar_file])
    pipe.wavecal_init()
    pipe.wavecal()
    science_files = [thar_file] + ([sky_file] if os.path.exists(sky_file) else [])
    pipe.extract(science_files)

    print(f"\n=== Running Pipeline ({channel}) ===")
    results = pipe.run()
