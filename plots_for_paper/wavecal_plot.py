# -*- coding: utf-8 -*-
"""
Simple usage example for PyReduce
Loads a sample UVES dataset, and runs the full extraction
"""

import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import gaussian

import pyreduce
from pyreduce import datasets
from pyreduce.combine_frames import combine_flat
from pyreduce.extract import fix_parameters
from pyreduce.util import make_index


def plot_comparison(
    original, orders, spectrum, slitf, extraction_width, column_range
):  # pragma: no cover
    nrow, ncol = original.shape
    nord = len(orders)
    output = np.zeros((np.sum(extraction_width) + nord, ncol))
    pos = [0]
    x = np.arange(ncol)
    for i in range(nord):
        ycen = np.polyval(orders[i], x)
        yb = ycen - extraction_width[i, 0]
        yt = ycen + extraction_width[i, 1]
        xl, xr = column_range[i]
        index = make_index(yb, yt, xl, xr)
        yl = pos[i]
        yr = pos[i] + index[0].shape[0]
        output[yl:yr, xl:xr] = original[index]

        vmin, vmax = np.percentile(output[yl:yr, xl:xr], (5, 95))
        output[yl:yr, xl:xr] = np.clip(output[yl:yr, xl:xr], vmin, vmax)
        output[yl:yr, xl:xr] -= vmin
        output[yl:yr, xl:xr] /= vmax - vmin

        pos += [yr]

    plt.imshow(output, origin="lower", aspect="auto", cmap="Greys")

    for i in range(nord):
        tmp = spectrum[i] - np.min(spectrum[i, column_range[i, 0] : column_range[i, 1]])
        # np.log(tmp, out=tmp, where=tmp > 0)
        tmp = tmp / np.max(tmp) * 0.9 * (pos[i + 1] - pos[i])
        tmp += pos[i]
        tmp[tmp < pos[i]] = pos[i]
        plt.plot(x, tmp, "r")

    locs = np.sum(extraction_width, axis=1) + 1
    locs = np.array([0, *np.cumsum(locs)[:-1]])
    locs[:-1] += (np.diff(locs) * 0.5).astype(int)
    locs[-1] += ((output.shape[0] - locs[-1]) * 0.5).astype(int)
    plt.yticks(locs, range(len(locs)))

    # plt.title("Extracted Spectrum vs. Input Image")

    plt.xlim(1050.5, 2205.5)
    plt.ylim(271.5, 355.5)

    plt.tight_layout()

    plt.xlabel("x [pixel]")
    plt.ylabel("order")
    plt.show()


def make_reference_spectrum(lines, nrow, ncol, nord):
    ref_image = np.zeros((nord, ncol))
    for iord in range(nord):
        for line in lines[lines["order"] == iord]:
            first = int(np.clip(line["xfirst"], 0, ncol))
            last = int(np.clip(line["xlast"], 0, ncol))
            ref_image[iord, first:last] += np.log(1 + line["height"]) * gaussian(
                last - first, line["width"]
            )

    return ref_image


# define parameters
instrument = "HARPS"
target = "HD109200"
night = "2015-04-09"
mode = "red"
extension = 2
steps = ["wavecal"]

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
base_dir = datasets.HARPS("/DATA/PyReduce")
input_dir = "raw"
output_dir = "reduced_{mode}"

# Path to the configuration parameters, that are to be used for this reduction
config = pyreduce.configuration.get_configuration_for_instrument(instrument, plot=0)

data = pyreduce.reduce.main(
    instrument,
    target,
    night,
    mode,
    steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    configuration=config,
    # order_range=(0, 25),
)
files = data[0]["files"]
mask = data[0]["mask"]
bias, bhead = data[0]["bias"]
orders, column_range = data[0]["orders"]
wave, coef, linelist = data[0]["wavecal"]
xwd = config["wavecal"]["extraction_width"]
nrow, ncol = bias.shape
nord = len(orders)
xwd, column_range, orders = fix_parameters(xwd, column_range, orders, nrow, ncol, nord)

orig, _ = combine_flat(
    files["wavecal_master"],
    instrument,
    mode,
    extension,
    mask=mask,
    bias=bias,
    bhead=bhead,
)

ref = make_reference_spectrum(linelist, nrow, ncol, nord)

plot_comparison(orig, orders, ref, None, xwd, column_range)
