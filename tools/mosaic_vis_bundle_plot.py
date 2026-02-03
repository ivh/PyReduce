"""
Plot MOSAIC VIS quadrants with bundle centers overlaid.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

import pyreduce.instruments.MOSAIC as mosaic_module
from pyreduce.instruments.instrument_info import load_instrument

inst = load_instrument("MOSAIC")
data_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
flat_file = os.path.join(
    data_dir,
    "MOSAIC/REF_E2E/VIS/E2E_as_built_FLAT_DIT_20s_MOSAIC_VIS_c01_FOCAL_PLANE_000.fits",
)

# Directory containing bundle_centers files
mosaic_dir = os.path.dirname(mosaic_module.__file__)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for ax, channel in zip(axes, ["VIS1", "VIS2", "VIS3", "VIS4"], strict=False):
    # Load quadrant data
    data, _ = inst.load_fits(flat_file, channel)
    ny, nx = data.shape

    # Load bundle centers
    bundle_file = os.path.join(mosaic_dir, f"bundle_centers_{channel.lower()}.yaml")
    with open(bundle_file) as f:
        centers = yaml.safe_load(f)
    y_positions = [v for k, v in centers.items() if isinstance(k, int)]

    # Plot image
    vmin, vmax = np.nanpercentile(data, [5, 95])
    ax.imshow(data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")

    # Plot bundle centers at x = center of image
    x_center = nx // 2
    ax.scatter(
        [x_center] * len(y_positions),
        y_positions,
        c="red",
        marker="_",
        s=100,
        linewidths=2,
        label=f"{len(y_positions)} bundles",
    )

    ax.set_title(f"{channel}: {ny} x {nx}, {len(y_positions)} bundles")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("/tmp/mosaic_vis_bundle_centers.png", dpi=150)
print("Saved to /tmp/mosaic_vis_bundle_centers.png")
plt.show()
