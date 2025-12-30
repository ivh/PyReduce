# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7a5"]
# ///
import numpy as np

from pyreduce.configuration import get_configuration_for_instrument
from pyreduce.instruments.common import create_custom_instrument
from pyreduce.pipeline import Pipeline
from pyreduce.util import start_logging

# Define the path to support files if possible
# otherwise set them to None
# Obviously they are necessary for their respective steps
bpm_mask = None
bias_file = "TOES-reduced/toes.bias.fits"
flat_file = "TOES-reduced/toes.flat.fits"
wavecal_file = "TOES-reduced/toes.thar_master.fits"

# create our custom instrument
instrument = create_custom_instrument("TOES", extension=0)
# Detector
# Override default values
# those can either be fixed values or refer to FITS header keywords
instrument.info["instrument"] = "TOES"
instrument.info["longitude"] = "GEO_LONG"
instrument.info["latitude"] = "GEO_LAT"
instrument.info["altitude"] = "GEO_ELEV"
instrument.info["gain"] = 1.1
instrument.info["readnoise"] = 5
instrument.info["prescan_x"] = 0
instrument.info["prescan_y"] = 0
instrument.info["overscan_x"] = 0
instrument.info["overscan_y"] = 0
instrument.info["orientation"] = 0
instrument.info["wavelength_range"] = [
    [
        [7359, 7519],  # Order 46
        [7205, 7360],  # Order 47
        [7054, 7207],  # Order 48
        [6912, 7058],  # Order 49
        [6775, 6919],  # Order 50
        [6627, 6783],  # Order 51
        [6499, 6649],  # Order 52
        [6377, 6526],  # Order 53
        [6258, 6406],  # Order 54
        [6145, 6288],  # Order 55
        [6035, 6177],  # Order 56
        [5929, 6068],  # Order 57
        [5827, 5963],  # Order 58
        [5737, 5854],  # Order 59
        [5633, 5765],  # Order 60
        [5540, 5671],  # Order 61
        [5451, 5578],  # Order 62
        [5364, 5490],  # Order 63
        [5281, 5404],  # Order 64
        [5199, 5322],  # Order 65
        [5121, 5240],  # Order 66
        [5044, 5163],  # Order 67
        [4970, 5086],  # Order 68
        [4898, 5012],  # Order 69
        [4828, 4941],  # Order 70
        [4760, 4871],  # Order 71
        [4694, 4804],  # Order 72
        [4630, 4738],  # Order 73
        [4567, 4675],  # Order 74
        [4506, 4612],  # Order 75
        [4447, 4551],  # Order 76
        [4388, 4493],  # Order 77
        [4333, 4435],  # Order 78
        [4278, 4379],  # Order 79
        [4224, 4324],  # Order 80
        [4173, 4271],  # Order 81
        [4121, 4219],  # Order 82
        [4072, 4168],  # Order 83
        [4024, 4118],  # Order 84
        #    [3977, 4069],  # Order 85
    ][::-1]
]  # ATTN, flipping the order to match the order of the traces

# For loading the config we specify pyreduce as the source, since this is the default
config = get_configuration_for_instrument("pyreduce")
# Define your own configuration
config["orders"]["filter_y"] = 20  # smoothing along cross-dispersion
config["orders"]["degree"] = 4
config["orders"]["degree_before_merge"] = 2
config["orders"]["noise"] = 5.5
config["orders"]["min_cluster"] = 3000
config["orders"]["min_width"] = 200
config["orders"]["manual"] = True
config["norm_flat"]["oversampling"] = 8  # Subpixel scale for slit function modelling
config["norm_flat"]["swath_width"] = 400  # Extraction swath width (columns)
config["wavecal_master"]["extraction_width"] = 2
config["wavecal_master"]["collapse_function"] = "sum"
config["wavecal_master"]["bias_scaling"] = "number_of_files"
config["wavecal"]["medium"] = "vac"
config["wavecal"]["threshold"] = 900
config["wavecal"]["shift_window"] = 0.01
config["wavecal"]["correlate_cols"] = True
config["wavecal"]["degree"] = [7, 7]
config["science"]["oversampling"] = 8  # Subpixel scale for slit function modelling
config["science"]["swath_width"] = 400  # Extraction swath width (columns)
config["science"]["smooth_slitfunction"] = 1.0  # Smoothing of the slit function
config["science"]["smooth_spectrum"] = 1.0e-6  # Smoothing in spectral direction
config["science"]["extraction_width"] = [5, 5]  # Extraction slit height (rows)
config["science"]["bias_scaling"] = "number_of_files"

# Since we can't find the files ourselves (at least not without defining the criteria we are looking for)
# We need to manually define which files go where
files = {
    "bias": ["Bias_0s_20240621_221716-%d.fit" % i for i in np.arange(1, 11)],
    "flat": ["Flat_5s_20240621_222912-%d.fit" % i for i in np.arange(1, 5)],
    "orders": [flat_file],
    "science": ["Vega_Object_25s_20240621_224908-%d.fit" % i for i in np.arange(1, 2)],
    "wavecal_master": [
        "Sun_Calibration_35s_20240621_184136-%d.fit" % i for i in np.arange(1, 2)
    ],
    # "Vega_Calibration_30s_20240621_225633-1.fit",
    # "Vega_Calibration_30s_20240621_225633-2.fit",
    # "Vega_Calibration_30s_20240621_225633-3.fit",
    # "Vega_Calibration_30s_20240621_225633-4.fit",
    # "Vega_Calibration_30s_20240621_225633-5.fit"],
}


# We define the path to the output directory
output_dir = "TOES-reduced"

# (optional) We need to define the log file
log_file = "TOES-reduced/log_file.txt"
start_logging(log_file)


# Define other parameter for PyReduce
target = ""
night = "2024-06-21"
channel = ""
steps = (
    #    "bias",
    #    "flat",
    #    "orders",
    #    "norm_flat",
    "wavecal_master",
    "wavecal",
    #     "science",
    #    "continuum",
    #    "finalize",
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
    # order_range=[6,8],
    steps=steps,
    plot=1,
)
data = pipe.run()
