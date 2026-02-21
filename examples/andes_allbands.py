"""
ANDES all-bands example: trace, curvature, and extraction for U/B/V/R/IZ/Y/J/H.

Runs the trace step on even/odd flat fields, then curvature on LFC data,
then extracts a combined frame (even + odd + LFC) for each ANDES band.

Data layout:
    ~/REDUCE_DATA/ANDES/trace_allbands/{BAND}_FLAT_even.fits
    ~/REDUCE_DATA/ANDES/trace_allbands/{BAND}_FLAT_odd.fits
    ~/REDUCE_DATA/ANDES/lfc_allfib_allbands/{BAND}_LFC_combined_all.fits

Usage:
    PYREDUCE_PLOT=1 uv run python examples/andes_allbands.py
    PYREDUCE_PLOT=1 uv run python examples/andes_allbands.py R Y
"""

import os
import sys

import numpy as np
from astropy.io import fits

from pyreduce.configuration import load_config
from pyreduce.pipeline import Pipeline

RERUN_TRACE = False

# Band -> (instrument, channel)
BANDS = {
    "U": ("ANDES_UBV", "U"),
    "B": ("ANDES_UBV", "B"),
    "V": ("ANDES_UBV", "V"),
    "R": ("ANDES_RIZ", "R"),
    "IZ": ("ANDES_RIZ", "IZ"),
    "Y": ("ANDES_YJH", "Y"),
    "J": ("ANDES_YJH", "J"),
    "H": ("ANDES_YJH", "H"),
}

data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
base_output = os.path.join(data_dir, "ANDES", "reduced_allbands")
trace_dir = os.path.join(data_dir, "ANDES", "trace_allbands")
lfc_dir = os.path.join(data_dir, "ANDES", "lfc_allfib_allbands")
plot = int(os.environ.get("PYREDUCE_PLOT", "1"))

# Select bands from command line, or all
requested = sys.argv[1:] if len(sys.argv) > 1 else list(BANDS)

for band in requested:
    band = band.upper()
    if band not in BANDS:
        print(f"Unknown band: {band}, skipping")
        continue

    instrument_name, channel = BANDS[band]
    output_dir = os.path.join(base_output, band)

    flat_even = os.path.join(trace_dir, f"{band}_FLAT_even.fits")
    flat_odd = os.path.join(trace_dir, f"{band}_FLAT_odd.fits")
    lfc_file = os.path.join(lfc_dir, f"{band}_LFC_combined_all.fits")

    trace_files = [f for f in [flat_even, flat_odd] if os.path.exists(f)]
    lfc_files = [lfc_file] if os.path.exists(lfc_file) else []

    if not trace_files:
        print(f"[{band}] No flat files found, skipping")
        continue

    print(f"\n{'=' * 60}")
    print(f"  {band}  ({instrument_name} / {channel})")
    print(f"  flats: {len(trace_files)}, lfc: {len(lfc_files)}")
    print(f"{'=' * 60}")

    config = load_config(None, instrument_name, channel=channel)
    pipe = Pipeline(
        instrument=instrument_name,
        output_dir=output_dir,
        target="ANDES_allbands",
        channel=channel,
        config=config,
        plot=plot,
        plot_dir=output_dir,
    )

    # --- Trace & curvature ---
    if RERUN_TRACE:
        pipe.trace(trace_files)
        if lfc_files:
            pipe.curvature(lfc_files)

        try:
            results = pipe.run()
        except Exception as e:
            print(f"[{band}] TRACE FAILED: {e}")
            continue

        traces = results.get("trace", [])
        groups = sorted({t.group for t in traces}, key=str)
        print(f"[{band}] {len(traces)} traces, groups: {groups}")
    else:
        print(f"[{band}] Skipping trace (RERUN_TRACE=False)")

    # --- Build combined frame for extraction ---
    input_files = trace_files + lfc_files
    img_combined = None
    for f in input_files:
        img, head = pipe.instrument.load_fits(f, channel=channel, extension=0)
        if img_combined is None:
            img_combined = np.asarray(img, dtype=np.float64)
        else:
            img_combined += np.asarray(img, dtype=np.float64)

    combined_file = os.path.join(output_dir, f"{band}_combined.fits")
    os.makedirs(output_dir, exist_ok=True)
    fits.writeto(combined_file, img_combined.astype(np.float32), head, overwrite=True)
    print(f"[{band}] Combined {len(input_files)} frames -> {combined_file}")

    # --- Extract ---
    # No bias/flat/scatter calibration for these simulated frames
    pipe._data.setdefault("mask", None)
    pipe._data.setdefault("bias", None)
    pipe._data.setdefault("norm_flat", None)
    pipe._data.setdefault("scatter", None)

    pipe.extract([combined_file])
    try:
        results = pipe.run()
    except Exception as e:
        print(f"[{band}] EXTRACT FAILED: {e}")
        continue

    spectra = results.get("science", (None, None))[1]
    if spectra:
        print(f"[{band}] Extracted {len(spectra[0])} spectra")
