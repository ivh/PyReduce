# /// script
# requires-python = ">=3.13"
# dependencies = ["pyreduce-astro>=0.7"]
# ///
"""
Debug script for extracting a single swath from a specific trace.

Manually loads traces from FITS, cuts out one swath, runs extraction,
and saves all intermediate outputs for inspection.
"""

import os
from os.path import join

import numpy as np
from astropy.io import fits

from pyreduce.cwrappers import slitfunc_curved
from pyreduce.extract import fix_parameters, make_bins
from pyreduce.trace_model import load_traces
from pyreduce.util import make_index

# === Configuration ===
# Which trace and swath to extract
TRACE_INDEX = 4  # 5th trace (0-indexed), i.e. 5th fiber bundle center
SWATH_INDEX = 2  # 3rd swath (0-indexed)

# Extraction parameters
EXTRACTION_HEIGHT = 0.5  # fraction of order spacing
OSAMPLE = 1
LAMBDA_SF = 0.1
LAMBDA_SP = 0
MAXITER = 20
GAIN = 1.0

# === Data paths ===
data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
base_dir = join(data_dir, "MOSAIC", "REF_E2E", "NIR")
output_dir = join(data_dir, "MOSAIC", "reduced", "NIR")

flat_file = join(
    base_dir,
    "E2E_FLAT_DIT_20s_MOSAIC_2Cam_c01",
    "E2E_FLAT_DIT_20s_MOSAIC_2Cam_c01_STATIC_FOCAL_PLANE.fits",
)
trace_file = join(output_dir, "MOSAIC_NIR.traces.fits")


def load_image(flat_file):
    """Load the flat field image."""
    with fits.open(flat_file) as hdu:
        img = hdu[0].data.astype(float)
    print(f"Loaded image: {img.shape}")
    return img


def main():
    # Load data
    img = load_image(flat_file)
    nrow, ncol = img.shape

    # Load traces from FITS file
    trace_list, _ = load_traces(trace_file)
    print(f"Loaded {len(trace_list)} traces from {trace_file}")

    # Convert to arrays for fix_parameters
    traces = np.array([t.pos for t in trace_list])
    column_range = np.array([t.column_range for t in trace_list])
    ntrace = len(traces)

    # Get extraction parameters
    xwd, cr = fix_parameters(
        EXTRACTION_HEIGHT, column_range, traces, nrow, ncol, ntrace
    )
    print(f"Extraction heights: {xwd[TRACE_INDEX]}")
    print(f"Column range: {cr[TRACE_INDEX]}")

    # Select the trace
    trace = traces[TRACE_INDEX]
    xlow, xhigh = xwd[TRACE_INDEX]
    ibeg, iend = cr[TRACE_INDEX]

    # Calculate trace center positions
    ix = np.arange(ncol)
    ycen = np.polyval(trace, ix)
    ycen_int = ycen.astype(int)

    # Make swath bins
    bins = make_bins(None, ibeg, iend, ycen)  # swath_width=None for default
    print(f"Number of swaths: {len(bins) - 1}")
    print(f"Swath boundaries: {bins}")

    # Select one swath
    swath_start = bins[SWATH_INDEX]
    swath_end = bins[SWATH_INDEX + 1]
    print(f"Processing swath {SWATH_INDEX}: columns {swath_start} to {swath_end}")

    # Extract swath region
    index = make_index(ycen_int - xlow, ycen_int + xhigh + 1, swath_start, swath_end)
    swath_img = img[index]
    swath_ycen = ycen[swath_start:swath_end] - ycen_int[swath_start:swath_end]

    print(f"Swath image shape: {swath_img.shape}")
    print(f"Swath ycen range: [{swath_ycen.min():.3f}, {swath_ycen.max():.3f}]")

    # Save input for debugging
    np.savetxt("debug_swath_img.txt", swath_img)
    np.savetxt("debug_swath_ycen.txt", swath_ycen)
    print("Saved debug_swath_img.txt and debug_swath_ycen.txt")

    # Run extraction
    print("\nRunning slitfunc_curved...")
    result = slitfunc_curved(
        swath_img,
        swath_ycen,
        0,  # p0 (tilt)
        0,  # p1 (curvature)
        LAMBDA_SF,
        LAMBDA_SP,
        osample=OSAMPLE,
        maxiter=MAXITER,
        gain=GAIN,
        yrange=(int(xlow), int(xhigh)),
    )

    spec, slitf, model, unc, mask = result

    # Save outputs
    np.savetxt("debug_spectrum.txt", spec)
    np.savetxt("debug_slitfunc.txt", slitf)
    np.savetxt("debug_model.txt", model)
    np.savetxt("debug_unc.txt", unc)
    np.savetxt("debug_mask.txt", mask.astype(int))

    print("\nResults saved:")
    print(f"  Spectrum: {len(spec)} points, sum={spec.sum():.1f}")
    print(f"  Slit function: {len(slitf)} points")
    print(f"  Model: {model.shape}")
    print(f"  Masked pixels: {mask.sum()} / {mask.size}")


if __name__ == "__main__":
    main()
