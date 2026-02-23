"""
ANDES LSF comparison: extract the same combined frame 5 ways.

Loads pre-computed traces (from andes_tracecurve.py) and extracts
a combined (even + odd + LFC) frame with different extraction modes
to compare line shapes.

Modes:
  1. standard     - merged slitA/B traces, CFFI backend
  2. charslit     - merged slitA/B, charslit backend, no deltas
  3. charslit+d   - merged slitA/B, charslit backend, with deltas
  4. fibers       - individual fiber traces (not merged), CFFI backend
  5. no_curve     - merged slitA/B, curvature nulled (vertical slit)

Data layout (same as andes_tracecurve.py):
    ~/REDUCE_DATA/ANDES/trace_allbands/{BAND}_FLAT_even.fits
    ~/REDUCE_DATA/ANDES/trace_allbands/{BAND}_FLAT_odd.fits
    ~/REDUCE_DATA/ANDES/lfc_allfib_allbands/{BAND}_LFC_combined_all.fits
    ~/REDUCE_DATA/ANDES/reduced_allbands/{BAND}/  (trace results)

Usage:
    PYREDUCE_PLOT=1 uv run python examples/andes_compLSF.py R
    PYREDUCE_PLOT=0 uv run python examples/andes_compLSF.py U B V
"""

import copy
import os
import sys

import numpy as np
from astropy.io import fits

from pyreduce.configuration import load_config
from pyreduce.pipeline import Pipeline
from pyreduce.trace_model import load_traces

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

MODES = [
    ("standard", {}),
    ("charslit", {"PYREDUCE_USE_CHARSLIT": "1", "PYREDUCE_USE_DELTAS": "0"}),
    ("charslit+d", {"PYREDUCE_USE_CHARSLIT": "1", "PYREDUCE_USE_DELTAS": "1"}),
    ("fibers", {}),
    ("no_curve", {}),
]

data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
trace_dir = os.path.join(data_dir, "ANDES", "trace_allbands")
lfc_dir = os.path.join(data_dir, "ANDES", "lfc_allfib_allbands")
base_output = os.path.join(data_dir, "ANDES", "reduced_allbands")
plot = int(os.environ.get("PYREDUCE_PLOT", "1"))

requested = sys.argv[1:] if len(sys.argv) > 1 else list(BANDS)


def make_pipeline(instrument_name, channel, output_dir, config):
    pipe = Pipeline(
        instrument=instrument_name,
        output_dir=output_dir,
        target="ANDES_compLSF",
        channel=channel,
        config=config,
        plot=plot,
        plot_dir=output_dir,
    )
    pipe._data["mask"] = None
    pipe._data["bias"] = None
    pipe._data["norm_flat"] = None
    pipe._data["scatter"] = None
    return pipe


for band in requested:
    band = band.upper()
    if band not in BANDS:
        print(f"Unknown band: {band}, skipping")
        continue

    instrument_name, channel = BANDS[band]
    band_dir = os.path.join(base_output, band)

    # Load pre-computed traces
    prefix = f"{instrument_name.lower()}_{channel.lower()}"
    trace_file = os.path.join(band_dir, f"{prefix}.traces.fits")
    if not os.path.exists(trace_file):
        print(f"[{band}] No trace file {trace_file}, skipping")
        continue

    traces, trace_header = load_traces(trace_file)
    grouped = [t for t in traces if t.group is not None]
    individual = [t for t in traces if t.fiber_idx is not None]
    groups = sorted({t.group for t in grouped}, key=str)
    print(f"\n{'=' * 60}")
    print(f"  {band}  ({instrument_name} / {channel})")
    print(f"  {len(grouped)} grouped traces ({groups}), {len(individual)} fiber traces")
    print(f"{'=' * 60}")

    config = load_config(None, instrument_name, channel=channel)

    # Build combined frame
    flat_even = os.path.join(trace_dir, f"{band}_FLAT_even.fits")
    flat_odd = os.path.join(trace_dir, f"{band}_FLAT_odd.fits")
    lfc_file = os.path.join(lfc_dir, f"{band}_LFC_combined_all.fits")
    trace_files = [f for f in [flat_even, flat_odd] if os.path.exists(f)]
    lfc_files = [lfc_file] if os.path.exists(lfc_file) else []
    input_files = trace_files + lfc_files

    if not input_files:
        print(f"[{band}] No input files, skipping")
        continue

    pipe_tmp = make_pipeline(instrument_name, channel, band_dir, config)
    img_combined = None
    for f in input_files:
        img, head = pipe_tmp.instrument.load_fits(f, channel=channel, extension=0)
        if img_combined is None:
            img_combined = np.asarray(img, dtype=np.float64)
        else:
            img_combined += np.asarray(img, dtype=np.float64)

    combined_file = os.path.join(band_dir, f"{band}_combined.fits")
    os.makedirs(band_dir, exist_ok=True)
    fits.writeto(combined_file, img_combined.astype(np.float32), head, overwrite=True)
    print(f"[{band}] Combined {len(input_files)} frames -> {combined_file}")

    for mode_name, env_vars in MODES:
        output_dir = os.path.join(band_dir, mode_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n--- {band} / {mode_name} ---")

        # Select traces for this mode
        if mode_name == "fibers":
            if not individual:
                print(
                    f"[{band}/{mode_name}] No individual fiber traces in {trace_file}, skipping"
                )
                continue
            mode_traces = individual
        elif mode_name == "no_curve":
            mode_traces = copy.deepcopy(grouped)
            for t in mode_traces:
                t.slit = None
                t.slitdelta = None
        else:
            mode_traces = grouped

        # Set env vars for this mode
        old_env = {}
        for k, v in env_vars.items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            pipe = make_pipeline(instrument_name, channel, output_dir, config)
            pipe._data["trace"] = mode_traces
            pipe.extract([combined_file])
            results = pipe.run()

            spectra = results.get("science", (None, None))[1]
            if spectra:
                print(f"[{band}/{mode_name}] Extracted {len(spectra[0])} spectra")

        except Exception as e:
            print(f"[{band}/{mode_name}] FAILED: {e}")
            import traceback

            traceback.print_exc()
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
