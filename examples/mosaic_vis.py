"""
MOSAIC VIS spectrograph example.

The VIS detector is a 4-quadrant mosaic (12788x12394 pixels total).
Each quadrant is processed as a separate channel:
- VIS1: lower-left quadrant
- VIS2: lower-right quadrant
- VIS3: upper-left quadrant
- VIS4: upper-right quadrant

This example processes VIS1. Run with different channel values for other quadrants.
"""

import os
from os.path import join

from pyreduce import util
from pyreduce.configuration import load_config
from pyreduce.pipeline import Pipeline

# instrument = instrument_info.load_instrument("MOSAIC")
# wave_range = instrument.get_wavelength_range(None, "B")

# Parameters
# Change channel to VIS2, VIS3, or VIS4 for other quadrants
instrument_name = "MOSAIC"
target = "MOSAIC_VIS"
night = ""
channels = ["VIS1", "VIS2", "VIS3", "VIS4"]
plot = 2

# File paths (simulated data)
# Set MOSAIC_USE_LOCAL=1 to use Tom's local E2E_may26 copies; otherwise Jens' paths.
if os.environ.get("MOSAIC_USE_LOCAL", "0") == "1":
    base_dir = "/Users/tom/REDUCE_DATA/MOSAIC/E2E_may26"
else:
    base_dir = (
        "/disk/miri-b1/jeand/mosaic/virtualmosaic/simdata_260429/VIS/moons_reduce"
    )

flat_file = join(base_dir, "E2E_FLAT_DIT_20s_MOSAIC_VIS_c01_FOCAL_PLANE_REF.fits")
thar_file = join(base_dir, "E2E_ThAr_DIT_20s_MOSAIC_VIS_c01_FOCAL_PLANE_REF.fits")
sky_file = join(base_dir, "E2E_SKY_DIT_150s_MOSAIC_VIS_c01_FOCAL_PLANE_000_REF.fits")

for fpath in [flat_file, thar_file] + ([sky_file] if sky_file else []):
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")

print(f"FLAT: {flat_file}")
print(f"ThAr: {thar_file}")
print(f"Sky:  {sky_file}")

for channel in [channels[2]]:
    output_dir = join(base_dir, "reduced", channel)

    # Handle plot environment variables
    if "PYREDUCE_PLOT" in os.environ:
        plot = int(os.environ["PYREDUCE_PLOT"])
    plot_dir = join(base_dir, "pyreduce_plots", channel)
    util.set_plot_dir(plot_dir)

    # Load configuration
    config = load_config(None, instrument_name, channel=channel)

    # Create pipeline - fiber grouping is handled by config.yaml
    pipe = Pipeline(
        instrument=instrument_name,
        output_dir=output_dir,
        target=target,
        channel=channel,
        night=night,
        config=config,
        plot=plot,
    )

    # Run pipeline steps
    pipe.trace([flat_file])
    pipe.curvature([thar_file])
    # pipe.flat([flat_file])
    # pipe.normalize_flat()
    pipe.wavecal_master([thar_file])
    pipe.wavecal_init()
    pipe.wavecal()
    science_files = [thar_file] + ([sky_file] if sky_file else [])
    pipe.extract(science_files)

    print("\n=== Running Pipeline ===")
    results = pipe.run()

# print("\n=== Results ===")
# traces = results["trace"]  # list[Trace]
# print(f"Traces: {len(traces)}")
# for t in traces[:3]:
#    print(f"  m={t.m}, fiber={t.fiber}, columns={t.column_range}")
