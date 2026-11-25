"""
SOXS (NTT) example: reduce the public commissioning data of both arms.

Data: the 94 public raw frames of ESO programme 114.28K3.001 (the soxspipe
quickstart set), decompressed into $REDUCE_DATA/SOXS/raw. Download from
http://archive.eso.org/wdb/wdb/eso/packages/query?pkg_id=167586 or with

    pyvo TAP on https://archive.eso.org/tap_obs, instrument='SOXS', then
    https://dataportal.eso.org/dataportal_new/file/<dp_id>, gunzip the .fits.Z

VIS (2025-10-12/13): bias, 5" slit QTH flats, pinhole/multi-pinhole ThAr,
    Feige 110 stare and nod through 5.0/1.5/1.0" slits.
NIR (2026-01-11): QTH flats, pinhole/multi-pinhole ArHgNeXe, CD-32 5613
    (= LTT 3218) stare and nod (5" slit).

The initial wavelength solution (wavecal_{vis,nir}.npz) comes from the lines
soxspipe identified on the single-pinhole arc, see tools/soxs_from_soxspipe.py,
so wavecal_init is not run.

Slit tilt: SOXS has no slit-filling arc, only the 9-pinhole mask. To measure the
tilt we smooth the multi-pinhole arc along the slit with a boxcar of one pinhole
pitch, which turns each dotted line into a continuous (staircase) line whose
mean slope is the tilt.

Nodding: PyReduce has no sky model along the slit, so for the nod frames we
extract A-B (and B-A) differences, with the trace moved to the object.

Usage:
    uv run python examples/soxs_example.py [VIS|NIR] [step,step,...]
"""

import os
import sys
from collections import defaultdict
from copy import deepcopy
from glob import glob
from os.path import basename, join

import numpy as np
from astropy.io import fits
from scipy.ndimage import uniform_filter1d

from pyreduce.configuration import load_config
from pyreduce.pipeline import Pipeline

arm = sys.argv[1].upper() if len(sys.argv) > 1 else "VIS"
steps = (
    sys.argv[2].split(",")
    if len(sys.argv) > 2
    else [
        "bias",
        "flat",
        "trace",
        "curvature",
        "scatter",
        "norm_flat",
        "wavecal_master",
        "wavecal",
        "science",
        "nod",
    ]
)

data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
raw_dir = join(data_dir, "SOXS", "raw")
output_dir = join(data_dir, "SOXS", "reduced", arm)
plot = int(os.environ.get("PYREDUCE_PLOT", "0"))
os.makedirs(output_dir, exist_ok=True)

# Multi-pinhole pitch along the slit (arcsec) and plate scale (arcsec/px)
PINHOLE_PITCH = {"VIS": 1.375, "NIR": 1.4375}
PLATE_SCALE = {"VIS": 0.272, "NIR": 0.33}


# --- File selection from the headers ---
def header_table(files):
    keys = {
        "arm": "ESO SEQ ARM",
        "catg": "ESO DPR CATG",
        "type": "ESO DPR TYPE",
        "tech": "ESO DPR TECH",
        "speed": "ESO DET READ CURID",
        "slit": "ESO INS VISE NAME" if arm == "VIS" else "ESO INS NISE NAME",
        "object": "OBJECT",
        "exptime": "EXPTIME",
        "mjd": "MJD-OBS",
    }
    rows = []
    for f in sorted(files):
        h = fits.getheader(f)
        rows.append({"file": f, **{k: h.get(v) for k, v in keys.items()}})
    return rows


def select(rows, **conds):
    return [r["file"] for r in rows if all(r[k] == v for k, v in conds.items())]


rows = [r for r in header_table(glob(join(raw_dir, "SOXS.*.fits"))) if r["arm"] == arm]
if arm == "VIS":
    # slow readout; the 5" slit flats match the science slit and illuminate all orders
    kw = {"speed": 1}
    files = {
        "bias": select(rows, type="BIAS", **kw),
        "flat": select(
            rows, type="LAMP,FLAT", tech="ECHELLE,SLIT", slit="SLIT5.0", **kw
        ),
        "mph": select(rows, type="LAMP,WAVE", tech="ECHELLE,MULTI-PINHOLE", **kw),
        "arc": select(rows, type="LAMP,WAVE", tech="ECHELLE,PINHOLE", **kw),
        "stare": select(rows, tech="ECHELLE,SLIT,STARE", **kw),
        "nod": select(rows, catg="SCIENCE", tech="ECHELLE,SLIT,NODDING", **kw),
    }
else:
    files = {
        "bias": [],
        "flat": select(rows, type="LAMP,FLAT", tech="ECHELLE,SLIT"),
        "mph": select(rows, type="LAMP,WAVE", tech="ECHELLE,MULTI-PINHOLE"),
        "arc": select(rows, type="LAMP,WAVE", tech="ECHELLE,PINHOLE"),
        "stare": select(rows, tech="ECHELLE,SLIT,STARE"),
        "nod": select(rows, catg="SCIENCE", tech="ECHELLE,SLIT,NODDING"),
    }
for k, v in files.items():
    print(f"{arm} {k:6s}: {len(v)} files")


def smoothed_mph(fname):
    """Multi-pinhole arc boxcar-smoothed along the slit by one pinhole pitch."""
    out = join(output_dir, basename(fname).replace(".fits", "_mph_smoothed.fits"))
    if not os.path.exists(out):
        with fits.open(fname) as hdu:
            data = hdu[0].data.astype(np.float32)
            width = int(round(PINHOLE_PITCH[arm] / PLATE_SCALE[arm]))
            # raw VIS has the slit along the columns, raw NIR along the rows
            axis = 1 if arm == "VIS" else 0
            data = uniform_filter1d(data, width, axis=axis)
            fits.writeto(out, data, hdu[0].header, overwrite=True)
    return out


# --- Pipeline ---
config = load_config(None, "SOXS", channel=arm)
pipe = Pipeline(
    instrument="SOXS",
    output_dir=output_dir,
    target="SOXS",
    channel=arm,
    config=config,
    plot=plot,
    plot_dir=join(output_dir, "plots") if plot else None,
)
pipe._data["mask"] = None
if arm == "NIR":
    # the NIR raw frames are ramp fits with the reference pixels removed: no bias
    pipe._data["bias"] = None

if "bias" in steps and files["bias"]:
    pipe.bias(files["bias"])
if "flat" in steps:
    pipe.flat(files["flat"])
if "trace" in steps:
    pipe.trace(files["flat"])
if "curvature" in steps:
    pipe.curvature([smoothed_mph(files["mph"][0])])
if "scatter" in steps:
    pipe.scatter(files["flat"])
if "norm_flat" in steps:
    pipe.normalize_flat()
if "wavecal_master" in steps:
    pipe.wavecal_master(files["arc"][:1])
if "wavecal" in steps:
    pipe._files["wavecal_master"] = files["arc"][:1]
    pipe.wavecal()
if "science" in steps:
    pipe.extract(files["stare"])
pipe.run()


# --- Nodding: extract A-B / B-A with the trace shifted onto the object ---
def object_offsets(image, traces, half=25):
    """Offset (px) of the positive object image from each trace centre.

    Measured per trace: the VIS quasi-orders do not all image the slit in the
    same direction, so A and B swap sides between orders.
    """
    ncol = image.shape[1]
    x = np.arange(ncol // 4, 3 * ncol // 4)
    rows = np.arange(-half, half + 1)
    offsets = []
    for t in traces:
        yc = np.polyval(t.pos, x)
        yy = np.clip(
            np.round(yc[None, :] + rows[:, None]).astype(int), 0, image.shape[0] - 1
        )
        profile = np.ma.filled(np.ma.median(image[yy, x[None, :]], axis=1), 0)
        i = int(np.argmax(profile))
        # sub-pixel centre from a parabola through the peak
        if 0 < i < len(rows) - 1:
            a, b, c = profile[i - 1 : i + 2]
            i = i + 0.5 * (a - c) / (a - 2 * b + c)
        offsets.append(rows[0] + i)
    return np.array(offsets)


def shift_trace(trace, dy, ncol):
    """Copy of trace moved dy rows along the slit, with its wavelength solution.

    The wavelength solution belongs to the slit centre. Along the tilted slit,
    column x at offset dy sees the wavelength of column x - c1*dy - c2*dy**2 on
    the centre (NIR: c1 up to 0.24, so the A and B images are ~3 px apart).
    """
    t = deepcopy(trace)
    t.pos = np.array(t.pos, dtype=float)
    t.pos[-1] += dy
    if t.wave is not None and t.slit is not None and t.wave.ndim == 1:
        x = np.arange(ncol, dtype=float)
        c = t.slit_at_x(x)
        dx = sum(c[i] * dy**i for i in range(1, len(c)))
        wave = trace.wlen(x - dx)
        t.wave = np.polyfit(x, wave, len(t.wave) + 1)
    return t


if "nod" in steps:
    from pyreduce.instruments.instrument_info import load_instrument

    inst = load_instrument("SOXS")
    pairs = defaultdict(list)
    for f in files["nod"]:
        h = fits.getheader(f)
        pairs[
            (
                h["ESO OBS ID"],
                h["ESO INS VISE NAME" if arm == "VIS" else "ESO INS NISE NAME"],
            )
        ].append(f)

    traces = pipe._run_step("trace", None, load_only=True)
    norm_flat = pipe._data.get("norm_flat") or pipe._run_step(
        "norm_flat", None, load_only=True
    )
    pipe._data["norm_flat"] = norm_flat
    pipe._data["scatter"] = None
    pipe._data["bias"] = None  # A-B removes bias and dark
    # the aperture must stay clear of the negative image one nod throw (~4.5") away
    pipe.config["science"]["extraction_height"] = 16

    for (_obid, slit), group in pairs.items():
        group = sorted(group)
        cumoff = [fits.getheader(f)["ESO SEQ CUMOFF Y"] for f in group]
        a_frames = [f for f, c in zip(group, cumoff, strict=True) if c > 0]
        b_frames = [f for f, c in zip(group, cumoff, strict=True) if c < 0]
        for fa, fb in zip(a_frames, b_frames, strict=False):
            for fpos, fneg in ((fa, fb), (fb, fa)):
                with fits.open(fpos) as hp, fits.open(fneg) as hn:
                    diff = hp[0].data.astype(np.float32) - hn[0].data.astype(np.float32)
                    head = hp[0].header.copy()
                out = join(
                    output_dir,
                    basename(fpos).replace(
                        ".fits", f"_minus_{basename(fneg)[5:28]}_nod.fits"
                    ),
                )
                fits.writeto(out, diff, head, overwrite=True)

                image, _ = inst.load_fits(out, arm)
                dy = object_offsets(image, traces)
                shifted = [
                    shift_trace(t, d, image.shape[1])
                    for t, d in zip(traces, dy, strict=True)
                ]
                pipe._data["trace"] = shifted
                pipe._steps = []
                pipe.extract([out])
                pipe._data.pop("science", None)
                print(
                    f"nod {slit}: {basename(fpos)} - {basename(fneg)}, object at",
                    np.round(dy, 1),
                    "px",
                )
                pipe.run()
