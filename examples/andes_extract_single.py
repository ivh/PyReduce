"""Extract a single order from an ANDES V-band frame.

Usage:
    uv run python examples/andes_extract_single.py
"""

import os

from pyreduce.configuration import load_config
from pyreduce.pipeline import Pipeline
from pyreduce.trace_model import load_traces

data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
band_dir = os.path.join(data_dir, "ANDES", "reduced_allbands", "V")
output_dir = os.path.join(band_dir, "single_extract")
os.makedirs(output_dir, exist_ok=True)

input_file = os.path.join(data_dir, "ANDES", "V_FF_slitA_1s_wl545-561.fits")
trace_file = os.path.join(band_dir, "andes_ubv_v.traces.fits")

traces, _ = load_traces(trace_file)
traces = [t for t in traces if t.m == 84 and t.group == "A"]
print(f"Using {len(traces)} trace(s): m={traces[0].m}, group={traces[0].group}")

config = load_config(None, "ANDES_UBV", channel="V")
plot = int(os.environ.get("PYREDUCE_PLOT", "1"))

pipe = Pipeline(
    instrument="ANDES_UBV",
    output_dir=output_dir,
    target="single_extract",
    channel="V",
    config=config,
    plot=plot,
    plot_dir=output_dir,
)
pipe._data["mask"] = None
pipe._data["bias"] = None
pipe._data["norm_flat"] = None
pipe._data["scatter"] = None
pipe._data["trace"] = traces
pipe.extract([input_file])
results = pipe.run()

spectra = results.get("science", (None, None))[1]
if spectra:
    print(f"Extracted {len(spectra[0])} spectrum/spectra")
