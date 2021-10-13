# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from pyreduce import cwrappers, extract

input_dir = "./datasets/reduced/UVES/HD132205/2010-04-02/middle/"
raw_dir = "./datasets/HD132205/"

hdu = fits.open(input_dir + "uves_middle.flat.fits")
img = hdu[0].data
nrow, ncol = img.shape

data = np.load(input_dir + "uves_middle.ord_default.npz", allow_pickle=True)
orders = data["orders"]
column_range = data["column_range"]

i = 5
ix = np.arange(ncol)

ylow, yhigh = 50, 50
ibeg, iend = 1500, 2000
ycen = np.polyval(orders[i], ix)
ycen_int = ycen.astype(int)
# yrange = extract.get_y_scale(ycen, [400, 600], [5, 5], nrow)

index = extract.make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
swath_img = img[index]
swath_ycen = ycen[ibeg:iend] - ycen_int[ibeg:iend]
shear = 0
osample = 10

np.savetxt("image.txt", swath_img)
np.savetxt("ycen.txt", swath_ycen)

# return sp, sl, model, unc, mask
# data1 = cwrappers.slitfunc(swath_img, swath_ycen, osample=osample)
data2 = cwrappers.slitfunc_curved(
    swath_img, swath_ycen, shear, osample=osample, yrange=(ylow, yhigh)
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

    # plt.subplot(111)
    # plt.plot(swath_img[:, i] / np.mean(swath_img[:, i]), label="Image")
    # plt.plot(sf[:, i] / np.mean(sf[:, i]), label="Interpolated")
    # plt.plot(np.linspace(-1, nrow, nsf), slitf / np.mean(slitf), label="Slitfunction")
    # plt.legend()
    # plt.show()

# plt.subplot(122)
# plt.imshow(swath_img / sf, aspect="auto", origin="lower")
# plt.show()

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
