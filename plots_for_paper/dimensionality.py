"""
Simple usage example for PyReduce
Loads a sample UVES dataset, and runs the full extraction
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path
import pyreduce
from pyreduce import datasets


# define parameters
instrument = "HARPS"
target = "HD109200"
night = "2015-04-09"
mode = "red"
steps = ("wavecal", "freq_comb")

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
base_dir = "/DATA/PyReduce/datasets/HARPS"
input_dir = "raw"
output_dir = f"reduced_{mode}"

# relevant configuration settings
config = {
    "__instrument__": instrument,
    "instrument": {"polarimetry": False, "fiber": "A"},
    "wavecal": {
        "threshold": 100,
        "dimensionality": "2D",
        "degree": [2, 5],
        "manual": False,
        "extraction_width": 10,
        "plot": False,
    },
    "freq_comb": {
        "dimensionality": "2D",
        "degree": [5, 5],
        "plot": False,
        "extraction_width": 10,
    },
}

pyreduce.reduce.main(
    instrument,
    target,
    night,
    mode,
    steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    configuration=config,
)

fname_lfc = os.path.join(base_dir, output_dir, f"harps_{mode}.comb.npz")
lfc = np.load(fname_lfc)
comb = lfc["wave"]

fname2D = os.path.join(base_dir, output_dir, f"harps_{mode}.thar.npz")
wave2D = np.load(fname2D)
wave2D = wave2D["wave"]


steps = ["wavecal"]
config["wavecal"] = {
    "threshold": 100,
    "dimensionality": "1D",
    "degree": 2,
    "manual": False,
    "extraction_width": 10,
    "plot": False,
}

pyreduce.reduce.main(
    instrument,
    target,
    night,
    mode,
    steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    configuration=config,
)

fname1D = os.path.join(base_dir, output_dir, f"harps_{mode}.thar.npz")
wave1D = np.load(fname1D)
wave1D = wave1D["wave"]


# plot
# 2D
gauss = lambda x, A, mu, sig: A * np.exp(-(x-mu)**2 / (2*sig**2))

xlim = (-0.004, 0.004)
ylim = (0, 4200)
bins = 100
x = np.linspace(xlim[0], xlim[1], 1000)

plt.subplot(121)
residual = wave1D - comb
residual = residual.ravel()
mean = np.median(residual)
std = np.percentile(residual, 68) - mean

A = plt.hist(residual, bins=bins, range=xlim)[0]
A = A.sum() * (xlim[1] - xlim[0]) / bins
A /= np.sqrt(2 * np.pi * std**2)
plt.plot(x, gauss(x, A, mean, std), "--")
plt.plot()
plt.title("1D")
plt.xlabel(r"$\Delta\lambda$ [Å]")
plt.ylabel("N")
plt.ylim(ylim)

plt.subplot(122)
residual = wave2D - comb
residual = residual.ravel()
mean = np.median(residual)
std = np.percentile(residual, 68) - mean
A = plt.hist(residual, bins=bins, range=xlim)[0]
A = A.sum() * (xlim[1] - xlim[0]) / bins
A /= np.sqrt(2 * np.pi * std**2)
plt.plot(x, gauss(x, A, mean, std), "--")
plt.title("2D")
plt.xlabel(r"$\Delta\lambda$ [Å]")
plt.ylabel("N")
plt.ylim(ylim)

plt.suptitle("ThAr - LFC")
plt.tight_layout()
plt.subplots_adjust(top=0.87)

plt.show()


plt.subplot(121)
residual = np.abs(wave1D - comb)
residual = residual.max(axis=1)
plt.plot(residual)
plt.xlabel("Order")
plt.ylabel(r"max($\Delta\lambda$) [Å]")
plt.title("1D")

plt.subplot(122)
residual = np.abs(wave2D - comb)
residual = residual.max(axis=1)
plt.plot(residual)
plt.xlabel("Order")
plt.ylabel(r"max($\Delta\lambda$) [Å]")
plt.title("2D")

plt.suptitle("Maximum difference in each order")
plt.subplots_adjust(wspace=0.45, right=0.96)
plt.show()
