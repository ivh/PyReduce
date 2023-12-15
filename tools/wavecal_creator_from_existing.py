# -*- coding: utf-8 -*-
"""
This script creates a new linelist file based on an existing wavelength solution and an atlas for a specific element of the gas lamp
used in the wavelength calibration.
"""

from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from pymultispec import readmultispec

from pyreduce.configuration import get_configuration_for_instrument
from pyreduce.instruments import instrument_info
from pyreduce.wavelength_calibration import LineAtlas, LineList, WavelengthCalibration

# Load Existing wavelength solution and ThAr file
# Here from IRAF
fname = join(dirname(__file__), "wef10006.ec.fits")
spec = readmultispec(fname)
wave = spec["wavelen"]
flux = np.nansum(spec["flux"], axis=0)
nord, ncol = flux.shape

# Load the lineatlas for the used element
# Here we use ThAr in vaccuum
atlas = LineAtlas("thar", "vac")

# Fill the linelist with the expected values
linelist = {
    "wlc": [],
    "wll": [],
    "posc": [],
    "posm": [],
    "xfirst": [],
    "xlast": [],
    "approx": [],
    "width": [],
    "height": [],
    "order": [],
    "flag": [],
}
# Adjust this threshold to what makes sense in your spectra
# it is used to differentiate between background and line
threshold = 1000

for i in range(nord):
    wmin, wmax = wave[i].min(), wave[i].max()
    # Find only useful lines
    idx_list = (atlas.linelist.wave > wmin) & (atlas.linelist.wave < wmax)
    height = np.interp(atlas.linelist[idx_list].wave, wave[i], flux[i])
    idx_list[idx_list] &= height > threshold
    nlines = np.sum(idx_list)

    linelist["wlc"] += [atlas.linelist[idx_list].wave]
    linelist["wll"] += [atlas.linelist[idx_list].wave]
    linelist["posc"] += [
        np.interp(atlas.linelist[idx_list].wave, wave[i], np.linspace(0, ncol, ncol))
    ]
    linelist["posm"] += [
        np.interp(atlas.linelist[idx_list].wave, wave[i], np.linspace(0, ncol, ncol))
    ]
    linelist["xfirst"] += [np.clip(linelist["posc"][-1] - 5, 0, ncol).astype(int)]
    linelist["xlast"] += [np.clip(linelist["posc"][-1] + 5, 0, ncol).astype(int)]
    linelist["approx"] += [np.full(nlines, "G")]
    linelist["width"] += [np.full(nlines, 1, float)]
    linelist["height"] += [
        np.interp(atlas.linelist[idx_list].wave, wave[i], flux[i] / np.nanmax(flux[i]))
    ]
    linelist["order"] += [np.full(nlines, i)]
    linelist["flag"] += [np.full(nlines, True)]

# Combine the data from the different orders
linelist = {k: np.concatenate(v) for k, v in linelist.items()}
linelist = np.rec.fromarrays(
    list(linelist.values()), names=list(linelist.keys()), dtype=LineList.dtype
)

# Run the lines through the wavecal
# This updates the linelist inplace and flags bad ones
# You need to check if this is a good solution
# And update the settings in the config file accordingly
# here: pyreduce/settings/settings_MCDONALD.json
# or after the
# config = get_configuration_for_instrument(instrument, plot=1)
# line in your script

# Setup the wavelength calibration module of PyReduce
instrument = instrument_info.load_instrument("MCDONALD")
module = WavelengthCalibration(
    plot=1,
    manual=True,
    degree=8,
    threshold=500,
    iterations=3,
    dimensionality="1D",
    nstep=0,
    shift_window=0.1,
    element="thar",
    medium="vac",
)
result = module.execute(flux, linelist)

# Save the linelist
linelist = LineList(linelist)
linelist.save("mcdonald.npz")
