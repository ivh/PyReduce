"""ANDES_YJH: extract the Fabry-Perot calibration fiber (H-band) and
characterize the FP line comb.

Extracts only the FP-fed calibration fiber from an H-band IFU frame, plots the
extracted comb, and measures the line spacing and median line width (FWHM) in
one spectral order.

Note on fiber choice: in the IFU (75-fiber) layout the FP calibration light is
carried by the two end fibers (fiber 1 and fiber 75); fiber 2 is a gap and the
central `cal` group (37-40) sits inside the IFU bundle and picks up sky/object
light instead. We therefore extract fiber 1 (fiber 75 is its identical twin).

Run:
    PYREDUCE_PLOT=0 uv run python examples/andes_yjh_fp.py
"""

import os

import numpy as np
from scipy.signal import find_peaks, peak_widths

from pyreduce.configuration import load_config
from pyreduce.pipeline import Pipeline

# --- Configuration ---
instrument_name = "ANDES_YJH"
channel = "H"
fp_fiber = os.environ.get("ANDES_FP_FIBER", "1")  # FP cal fiber (1 or 75)
data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
raw_dir = os.path.join(data_dir, "ANDES", channel)
output_dir = os.path.join(data_dir, "ANDES", "reduced", channel)

# FP-containing frame of interest
fp_file = os.path.join(raw_dir, "H_ifu_HR1544_skyabs_skyemi_fp_20260314.fits")

plot = int(os.environ.get("PYREDUCE_PLOT", "1"))

config = load_config(None, instrument_name, channel=channel)

pipe = Pipeline(
    instrument=instrument_name,
    channel=channel,
    output_dir=output_dir,
    target="ANDES_fp_cal",
    config=config,
    plot=plot,
    plot_dir=output_dir,
)

# Extract only the FP calibration fiber.
pipe.use_fibers([fp_fiber], step="science")

# --- Load traces (with stored curvature) from a previous run ---
print("Loading traces from previous run...")
trace_objects = pipe._run_step("trace", None, load_only=True)
pipe._data["trace"] = trace_objects
print(f"  Loaded {len(trace_objects)} traces")

# Bypass bias/flat/scatter calibration - not needed for the FP fiber.
pipe._data["mask"] = None
pipe._data["bias"] = None
pipe._data["norm_flat"] = None
pipe._data["scatter"] = None

# --- Extract ---
print(f"\nExtracting FP fiber {fp_fiber} from {os.path.basename(fp_file)}...")
results = pipe.extract([fp_file]).run()

# results["science"] = (heads, list-per-file of list[Spectrum])
spectra = results["science"][1][0]
spectra = [s for s in spectra if str(s.fiber_idx) == fp_fiber]
spectra.sort(key=lambda s: (s.m if s.m is not None else 0))
print(f"  Got {len(spectra)} order spectra for fiber {fp_fiber}")
print(f"  Orders (m): {[s.m for s in spectra]}")


def measure_comb(flux):
    """Detect FP lines and return (peak_positions, spacings, fwhm_widths)."""
    flux = np.nan_to_num(flux, nan=0.0)
    # FP lines sit well above the inter-line baseline; median tracks the
    # baseline, so a multiple of it is a robust, blaze-tolerant threshold.
    thr = 5.0 * np.nanmedian(flux[flux > 0])
    peaks, _ = find_peaks(flux, height=thr, distance=3)
    spacings = np.diff(peaks)
    fwhm = peak_widths(flux, peaks, rel_height=0.5)[0]
    return peaks, spacings, fwhm


# --- Measure one order (middle) ---
sp = spectra[len(spectra) // 2]
peaks, spacings, fwhm = measure_comb(sp.spec)

print(f"\nFP comb in order m={sp.m}:")
print(f"  Detected {len(peaks)} FP lines")
if spacings.size:
    print(
        f"  Line spacing:    median {np.median(spacings):.2f} px "
        f"(mean {spacings.mean():.2f}, std {spacings.std():.2f})"
    )
if fwhm.size:
    print(
        f"  Line width FWHM: median {np.median(fwhm):.2f} px "
        f"(mean {fwhm.mean():.2f}, std {fwhm.std():.2f})"
    )

# --- Plot ---
if plot:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

    for s in spectra:
        ax1.plot(np.arange(s.spec.size), s.spec, lw=0.5)
    ax1.set_title(
        f"ANDES {channel}-band FP (fiber {fp_fiber}) - {len(spectra)} extracted orders"
    )
    ax1.set_xlabel("pixel")
    ax1.set_ylabel("flux")

    flux = np.nan_to_num(sp.spec, nan=0.0)
    ax2.plot(np.arange(flux.size), flux, lw=0.7, color="k")
    ax2.plot(peaks, flux[peaks], "r.", ms=6, label=f"{len(peaks)} lines")
    title = f"Order m={sp.m}"
    if spacings.size:
        title += f": spacing {np.median(spacings):.1f} px"
    if fwhm.size:
        title += f", FWHM {np.median(fwhm):.1f} px"
    ax2.set_title(title)
    ax2.set_xlabel("pixel")
    ax2.set_ylabel("flux")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_png = os.path.join(output_dir, "andes_h_fp_fiber.png")
    fig.savefig(out_png, dpi=120)
    print(f"\nSaved plot: {out_png}")
    if os.environ.get("PYREDUCE_PLOT_SHOW", "block") != "off":
        plt.show()
