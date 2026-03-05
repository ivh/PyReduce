"""Reducing data from an unsupported spectrograph with PyReduce.

PyReduce includes built-in support for many instruments (UVES, HARPS, XSHOOTER, ...),
but you can also reduce data from any echelle spectrograph by creating a custom
instrument. This example shows how.

You need to provide:
  1. Basic detector properties (gain, readnoise, orientation, FITS extension)
  2. Your FITS files, sorted by type (bias, flat, science, etc.)
  3. Reduction parameters tuned to your data

The key idea: detector properties can be either literal numbers (e.g., gain=1.2)
or FITS header keyword strings (e.g., gain="GAIN"). When a string is given,
PyReduce looks up the value from each file's FITS header at runtime.
"""

from pyreduce.configuration import get_configuration_for_instrument
from pyreduce.instruments.common import create_custom_instrument
from pyreduce.pipeline import Pipeline

# ============================================================================
# Step 1: Create your instrument
# ============================================================================

# At minimum you need gain, readnoise, and the FITS extension where data lives.
# Use literal values if you know them, or header keywords if they vary per file.

instrument = create_custom_instrument(
    "MySpectrograph",  # name (used in output filenames)
    extension=0,  # FITS extension containing the image data
    # Optional paths to pre-existing calibration files:
    # mask_file="path/to/bad_pixel_mask.fits",
    # wavecal_file="path/to/wavecal_solution",
    #
    # Detector properties - literal values or FITS header keywords:
    gain=1.2,  # e-/ADU (or "GAIN" to read from header)
    readnoise=4.5,  # e-   (or "RDNOISE" to read from header)
    orientation=0,  # image rotation code, see below
)

# Orientation codes (applied so that dispersion runs horizontally):
#   0: no change            4: transpose
#   1: 90 deg CCW           5: transpose + 90 deg CCW
#   2: 180 deg              6: transpose + 180 deg
#   3: 270 deg CCW          7: transpose + 270 deg CCW
# Tip: try different values until traces appear as roughly horizontal lines
# in the trace detection plot.

# ============================================================================
# Step 2: Load default reduction settings and tune for your data
# ============================================================================

config = get_configuration_for_instrument("pyreduce")

# Trace detection - these depend heavily on your detector and data:
config["trace"]["degree"] = 4  # polynomial degree for trace shape
config["trace"]["noise"] = (
    100  # detection threshold (try lowering if orders are missed)
)
config["trace"]["min_cluster"] = 500  # minimum pixels per detected order
config["trace"]["filter_y"] = 120  # smoothing along cross-dispersion direction

# Extraction:
config["science"]["extraction_height"] = 0.5  # fraction of order separation
config["science"]["oversampling"] = 10  # subpixel oversampling factor

# ============================================================================
# Step 3: Provide your FITS files
# ============================================================================

# Since PyReduce can't auto-classify files for an unknown instrument,
# you must specify which files belong to which reduction step.

files = {
    "bias": [
        "data/bias_001.fits",
        "data/bias_002.fits",
        "data/bias_003.fits",
    ],
    "flat": [
        "data/flat_001.fits",
        "data/flat_002.fits",
    ],
    "science": [
        "data/science_001.fits",
    ],
    # Uncomment if you have wavelength calibration (arc lamp) frames:
    # "wavecal_master": [
    #     "data/thar_001.fits",
    # ],
}

# ============================================================================
# Step 4: Run the pipeline
# ============================================================================

output_dir = "reduced"
target = "MyTarget"
night = "2024-01-15"

steps = (
    "bias",
    "flat",
    "trace",
    "norm_flat",
    "curvature",
    # "wavecal_master",  # uncomment for wavelength calibration
    # "wavecal_init",
    # "wavecal",
    "science",
    # "continuum",
    # "finalize",
)

pipe = Pipeline.from_files(
    files=files,
    output_dir=output_dir,
    target=target,
    instrument=instrument,
    channel="",
    night=night,
    config=config,
    steps=steps,
    plot=1,  # 0=off, 1=basic plots, 2=detailed
)
data = pipe.run()

# Results are saved in output_dir as FITS files.
# The 'data' dict contains in-memory results keyed by step name.
