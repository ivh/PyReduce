"""
MOSAIC NIR LR-J starsky example.

Uses the newer LR-J simulations:
- LR-J_FF_all_1s.fits        : flat field (all fibers illuminated) for tracing
- LR-J_LFC_all_1s.fits       : laser frequency comb (all fibers) for slit curvature
- LR-J_starsky_combined.fits : science frame; only bundle 45 (fibers 309-315) illuminated

The MOSAIC config groups 630 fibers into 90 bundles of 7 and extracts the
bundle centers. After extraction, only bundle 45 carries signal; the other
89 bundles are essentially zero.
"""

import os
from os.path import join

from pyreduce import util
from pyreduce.configuration import load_config
from pyreduce.pipeline import Pipeline

instrument_name = "MOSAIC"
target = "MOSAIC_NIR_LRJ_starsky"
night = ""
channel = "LR-J"
plot = 1

# Re-run trace / curvature, or load saved results from a previous run.
# Set False to skip and reuse cached output for fast iteration on extraction.
rerun_trace = False
rerun_curvature = False

if "PYREDUCE_PLOT" in os.environ:
    plot = int(os.environ["PYREDUCE_PLOT"])
plot_dir = os.environ.get("PYREDUCE_PLOT_DIR")
util.set_plot_dir(plot_dir)

data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
base_dir = join(data_dir, "MOSAIC")
output_dir = join(data_dir, "MOSAIC", "reduced", "NIR_LRJ_starsky")

flat_file = join(base_dir, "LR-J_FF_all_1s.fits")
lfc_file = join(base_dir, "LR-J_LFC_all_1s.fits")
science_file = join(base_dir, "LR-J_starsky_combined.fits")

for fpath in [flat_file, lfc_file, science_file]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")

print(f"FLAT:    {flat_file}")
print(f"LFC:     {lfc_file}")
print(f"SCIENCE: {science_file}")

config = load_config(None, instrument_name, plot, channel=channel)

pipe = Pipeline(
    instrument=instrument_name,
    output_dir=output_dir,
    target=target,
    channel=channel,
    night=night,
    config=config,
    plot=plot,
)

if rerun_trace:
    pipe.trace([flat_file])
if rerun_curvature:
    pipe.curvature([lfc_file])
pipe.extract([science_file])

# Only bundle 45 (fibers 309-315) is illuminated in the starsky frame, so
# restrict science extraction to that bundle. Curvature still uses all
# bundles since the LFC illuminates everything.
pipe.instrument.config.fibers.use["science"] = ["bundle_45"]

print("\n=== Running Pipeline ===")
results = pipe.run()

print("\n=== Results ===")
traces = results["trace"]
print(f"Traces: {len(traces)}")
for t in traces[:3]:
    print(
        f"  m={t.m}, group={t.group}, fiber_idx={t.fiber_idx}, columns={t.column_range}"
    )
