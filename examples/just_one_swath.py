"""
Debug script for extracting a single swath from a specific trace.

Demonstrates manual swath extraction using the trace_model interface.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from pyreduce import cwrappers, extract
from pyreduce.trace_model import load_traces

input_dir = "../DATA/datasets/UVES/reduced/2010-04-01/middle/"
raw_dir = "../DATA/datasets/UVES/raw/"

hdu = fits.open(input_dir + "uves_middle.flat.fits")
img = hdu[0].data
nrow, ncol = img.shape

# Load traces from FITS file
traces, _ = load_traces(input_dir + "uves_middle.traces.fits")
print(f"Loaded {len(traces)} traces")

i = 5
ix = np.arange(ncol)

ylow, yhigh = 50, 50
ibeg, iend = 1500, 2000
ycen = np.polyval(traces[i].pos, ix)
ycen_int = ycen.astype(int)

index = extract.make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
swath_img = img[index]
swath_ycen = ycen[ibeg:iend] - ycen_int[ibeg:iend]
osample = 10

np.savetxt("image.txt", swath_img)
np.savetxt("ycen.txt", swath_ycen)

data2 = cwrappers.slitfunc_curved(
    swath_img, swath_ycen, 0, 0, 0.0, 0.5, osample=osample, yrange=(ylow, yhigh)
)

data = data2
spec = data[0]
slitf = data[1]
model = data[2]
unc = data[3]
mask = data[4]

np.savetxt("spectrum.txt", spec)
np.savetxt("slitfunction.txt", slitf)
np.savetxt("model.txt", model)
np.savetxt("unc.txt", unc)
np.savetxt("mask.txt", mask)


# Input image
plt.subplot(121)
plt.imshow(swath_img, aspect="auto", origin="lower")
plt.xlabel("x")
plt.ylabel("y")

# Spectrum
plt.subplot(222)
x = np.indices(swath_img.shape)[1].ravel()

nsf = len(slitf)
nrow, ncol = swath_img.shape
sf = np.zeros(swath_img.shape)
for i in range(ncol):
    sf[:, i] = np.interp(
        np.linspace(0, nrow - 1, nrow),
        np.linspace(-1, nrow - 1 + 1, nsf) + (swath_ycen[i] - 0.5) * 1,
        slitf,
    )

y = swath_img / sf
y = y.ravel() * np.mean(spec) / np.mean(y)
plt.plot(x, y, ".", label="Observation")
plt.plot(spec, label="Extraction")
plt.title("Spectrum")
plt.xlabel("x [Pixel]")
plt.ylabel("Intensity [au]")
plt.legend()

# Slitfunction
plt.subplot(224)
x = np.indices(swath_img.shape)[0].ravel()
y = swath_img / spec[None, :]
y = y.ravel() * np.mean(slitf) / np.mean(y)

sf = np.interp(np.linspace(0, nrow - 1, nrow), np.linspace(-1, nrow, nsf), slitf)
plt.plot(x, y, ".", label="Observation")
plt.plot(sf, label="Extraction")
plt.title("Slitfunction")
plt.xlabel("y [Pixel]")
plt.ylabel("Intensity [au]")
plt.legend()

plt.show()

pass
