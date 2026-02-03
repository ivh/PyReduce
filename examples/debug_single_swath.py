"""
Debug script for extracting a single swath from a specific trace.

Manually loads traces from NPZ, cuts out one swath, runs extraction,
and saves all intermediate outputs for inspection.
"""

import os
from os.path import join

import numpy as np
from astropy.io import fits

from pyreduce.cwrappers import slitfunc_curved
from pyreduce.extract import fix_parameters, make_bins
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
trace_file = join(output_dir, "MOSAIC_NIR.traces.npz")


def load_traces(trace_file):
    """Load traces and column ranges from NPZ."""
    import warnings

    data = np.load(trace_file, allow_pickle=True)
    if "traces" in data:
        traces = data["traces"]
    elif "orders" in data:
        warnings.warn(
            f"Trace file {trace_file} uses old key 'orders'. "
            "Re-run the trace step to update the file format.",
            DeprecationWarning,
            stacklevel=2,
        )
        traces = data["orders"]
    else:
        raise KeyError("Trace file missing 'traces' key")
    column_range = data["column_range"]
    print(f"Loaded {len(traces)} traces from {trace_file}")
    return traces, column_range


def load_image(flat_file):
    """Load and prepare the flat field image."""
    with fits.open(flat_file) as hdul:
        img = hdul[0].data.astype(float)
    print(f"Loaded image {img.shape} from {flat_file}")
    return img


def extract_single_swath(
    img,
    trace_coef,
    column_range,
    swath_index,
    extraction_height,
    p1=None,
    p2=None,
):
    """
    Extract a single swath from one trace.

    Returns the swath image, extraction results, and metadata.
    """
    nrow, ncol = img.shape
    xlow, xhigh = column_range

    # Evaluate trace polynomial
    x = np.arange(ncol)
    ycen = np.polyval(trace_coef, x)

    # Get extraction height in pixels
    ylow, yhigh = extraction_height

    # Create swath bins
    nbin, bins_start, bins_end = make_bins(None, xlow, xhigh, ycen)
    nswath = 2 * nbin - 1
    print(f"Order has {nswath} swaths, extracting swath {swath_index}")

    if swath_index >= nswath:
        raise ValueError(f"Swath index {swath_index} >= number of swaths {nswath}")

    # Get this swath's column range
    ibeg = bins_start[swath_index]
    iend = bins_end[swath_index]
    print(f"Swath columns: {ibeg} - {iend} (width={iend - ibeg})")

    # Cut out swath from image
    ycen_int = np.floor(ycen).astype(int)
    index = make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
    swath_img = img[index].copy()
    swath_ycen = ycen[ibeg:iend]

    # Fractional y offset within each pixel
    swath_ycen_frac = swath_ycen - np.floor(swath_ycen)

    print(f"Swath image shape: {swath_img.shape}")
    print(f"Swath ycen range: {swath_ycen.min():.1f} - {swath_ycen.max():.1f}")

    # Get curvature for this swath
    swath_p1 = p1[ibeg:iend] if p1 is not None else 0
    swath_p2 = p2[ibeg:iend] if p2 is not None else 0

    # Run extraction
    yrange = (ylow, yhigh)
    spec, slitf, model, unc, mask, info = slitfunc_curved(
        swath_img,
        swath_ycen_frac,
        swath_p1,
        swath_p2,
        lambda_sp=LAMBDA_SP,
        lambda_sf=LAMBDA_SF,
        osample=OSAMPLE,
        yrange=yrange,
        maxiter=MAXITER,
        gain=GAIN,
    )

    return {
        "swath_img": swath_img,
        "spec": spec,
        "slitf": slitf,
        "model": model,
        "unc": unc,
        "mask": mask,
        "info": info,
        "ycen": swath_ycen_frac,
        "ibeg": ibeg,
        "iend": iend,
        "yrange": yrange,
    }


def main():
    # Check files exist
    for fpath in [flat_file, trace_file]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")

    # Load data
    img = load_image(flat_file)
    traces, column_range = load_traces(trace_file)
    nrow, ncol = img.shape
    ntrace = len(traces)

    print(f"\nImage: {nrow} x {ncol}")
    print(f"Traces: {ntrace}")
    print(f"Extracting trace {TRACE_INDEX}, swath {SWATH_INDEX}\n")

    # Fix extraction parameters
    xwd, cr, traces = fix_parameters(
        EXTRACTION_HEIGHT, column_range, traces, nrow, ncol, ntrace
    )

    # Get the specific trace
    trace_coef = traces[TRACE_INDEX]
    trace_cr = cr[TRACE_INDEX]
    trace_xwd = xwd[TRACE_INDEX]

    print(f"Trace {TRACE_INDEX}:")
    print(f"  Column range: {trace_cr}")
    print(f"  Extraction height: {trace_xwd}")

    # Extract single swath
    result = extract_single_swath(
        img,
        trace_coef,
        trace_cr,
        SWATH_INDEX,
        trace_xwd,
        p1=None,  # No curvature correction
        p2=None,
    )

    # Save outputs
    out_file = join(output_dir, f"debug_swath_t{TRACE_INDEX}_s{SWATH_INDEX}.npz")
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        out_file,
        swath_img=result["swath_img"],
        spec=result["spec"],
        slitf=result["slitf"],
        model=result["model"],
        unc=result["unc"],
        mask=result["mask"],
        ycen=result["ycen"],
        ibeg=result["ibeg"],
        iend=result["iend"],
        yrange=result["yrange"],
        trace_index=TRACE_INDEX,
        swath_index=SWATH_INDEX,
    )
    print(f"\nSaved results to: {out_file}")

    # Print summary
    print("\n=== Extraction Results ===")
    print(f"Spectrum shape: {result['spec'].shape}")
    print(f"Slitfunction shape: {result['slitf'].shape}")
    print(f"Model shape: {result['model'].shape}")
    print(f"Chi-squared: {result['info'][1]:.3f}")
    print(f"Iterations: {result['info'][0]}")
    print(f"Masked pixels: {result['mask'].sum()}")


if __name__ == "__main__":
    main()
