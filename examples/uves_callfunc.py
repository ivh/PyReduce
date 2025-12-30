# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7a5"]
# ///
"""
Example showing direct function calls for each step.

Instead of using Pipeline.run(), this script calls each step's run()
method directly. This allows access to file lists and intermediate
results between steps, enabling custom processing.
"""

import os
from os.path import join

from pyreduce import datasets, util
from pyreduce.configuration import load_config
from pyreduce.instruments.instrument_info import load_instrument
from pyreduce.reduce import (
    Bias,
    ContinuumNormalization,
    Finalize,
    Flat,
    Mask,
    NormalizeFlatField,
    OrderTracing,
    ScienceExtraction,
    SlitCurvatureDetermination,
    WavelengthCalibrationFinalize,
    WavelengthCalibrationInitialize,
    WavelengthCalibrationMaster,
)

# Parameters
instrument_name = "UVES"
target = "HD[- ]?132205"
night = "2010-04-01"
channel = "middle"
order_range = (1, 21)
plot = 1

# Handle plot environment variables (same as Pipeline does)
if "PYREDUCE_PLOT" in os.environ:
    plot = int(os.environ["PYREDUCE_PLOT"])
plot_dir = os.environ.get("PYREDUCE_PLOT_DIR")
util.set_plot_dir(plot_dir)

# Load dataset
base_dir = datasets.UVES()
input_dir = join(base_dir, "raw/")
output_dir = join(base_dir, f"reduced/{night}/{channel}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load instrument and configuration
instrument = load_instrument(instrument_name)
config = load_config(None, instrument_name, 0)

# Common step arguments
step_args = (instrument, channel, target, night, output_dir, order_range)

# Find and classify files automatically
file_groups = instrument.sort_files(
    input_dir,
    target,
    night,
    channel=channel,
    **config["instrument"],
)

if len(file_groups) == 0:
    raise FileNotFoundError(f"No files found for {target} in {input_dir}")

# Get the first file group (single channel/night)
settings, files = file_groups[0]
print("Settings:", settings)

# Files are now accessible as a dict with keys:
#   bias, flat, orders, curvature, scatter, wavecal_master, freq_comb_master, science
# Each is a list/array of file paths

bias_files = files.get("bias", [])
flat_files = files.get("flat", [])
order_files = files.get("orders", flat_files)  # Use flat if no dedicated order files
curvature_files = files.get("curvature", files.get("wavecal_master", []))
wavecal_files = files.get("wavecal_master", [])
science_files = files.get("science", [])

print(f"Bias files: {len(bias_files)}")
print(f"Flat files: {len(flat_files)}")
print(f"Curvature files: {len(curvature_files)}")
print(f"Wavecal files: {len(wavecal_files)}")
print(f"Science files: {len(science_files)}")

# --- Run each step manually ---


def step_config(name):
    """Get step config with plot level override."""
    cfg = config.get(name, {}).copy()
    cfg["plot"] = plot
    return cfg


# Step 1: Mask
print("\n=== MASK ===")
mask_step = Mask(*step_args, **step_config("mask"))
mask = mask_step.run()

# Step 2: Bias
print("\n=== BIAS ===")
bias_step = Bias(*step_args, **step_config("bias"))
bias = bias_step.run(bias_files, mask)

# Step 3: Flat
print("\n=== FLAT ===")
flat_step = Flat(*step_args, **step_config("flat"))
flat = flat_step.run(flat_files, bias, mask)

# Step 4: Order tracing
print("\n=== ORDERS ===")
orders_step = OrderTracing(*step_args, **step_config("orders"))
orders = orders_step.run(order_files, mask, bias)

# Step 5: Curvature
print("\n=== CURVATURE ===")
curvature_step = SlitCurvatureDetermination(*step_args, **step_config("curvature"))
curvature = curvature_step.run(curvature_files, orders, mask, bias)

# Step 6: Normalize flat
print("\n=== NORM_FLAT ===")
norm_flat_step = NormalizeFlatField(*step_args, **step_config("norm_flat"))
scatter = None  # Optional background scatter
norm_flat = norm_flat_step.run(flat, orders, scatter, curvature)

# Step 7: Wavelength calibration (three sub-steps)
print("\n=== WAVECAL ===")
wavecal_master_step = WavelengthCalibrationMaster(
    *step_args, **step_config("wavecal_master")
)
wavecal_master = wavecal_master_step.run(
    wavecal_files, orders, mask, curvature, bias, norm_flat
)

# wavecal_init: load existing linelist (from instrument defaults or previous run)
# Use run() only if you want to create a new linelist from scratch
wavecal_init_step = WavelengthCalibrationInitialize(
    *step_args, **step_config("wavecal_init")
)
wavecal_init = wavecal_init_step.load(config, wavecal_master)

wavecal_step = WavelengthCalibrationFinalize(*step_args, **step_config("wavecal"))
wavecal = wavecal_step.run(wavecal_master, wavecal_init)
wave, coef, linelist = wavecal

# Step 8: Science extraction
print("\n=== SCIENCE ===")
science_step = ScienceExtraction(*step_args, **step_config("science"))
science = science_step.run(
    science_files, bias, orders, norm_flat, curvature, scatter, mask
)

# Step 9: Continuum normalization
print("\n=== CONTINUUM ===")
continuum_step = ContinuumNormalization(*step_args, **step_config("continuum"))
continuum = continuum_step.run(science, wave, norm_flat)

# Step 10: Finalize
print("\n=== FINALIZE ===")
finalize_step = Finalize(*step_args, **step_config("finalize"))
finalize_step.run(continuum, wave, config)

print("\nDone!")
