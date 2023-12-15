# -*- coding: utf-8 -*-
"""
with our powers combined we increase snr
"""
import logging
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from spectres import spectres
from tqdm import tqdm

from .. import echelle

logger = logging.getLogger(__name__)


def combine(files, output, plot=None):
    # Create a Wavelength grid that will be used for all spectra
    # Based on the one in the "first" fits file
    e = echelle.read(files[0], continuum_normalization=False)
    nord, ncol = e.spec.shape

    # Prepare some empty arrays for storage of all the data
    # TODO what if this becomes to large to handle?
    waves = np.zeros((len(files), nord, ncol))
    specs = np.zeros((len(files), nord, ncol))
    sigms = np.zeros((len(files), nord, ncol))
    conts = np.zeros((len(files), nord, ncol))

    mask = np.full(len(files), True)

    # Load all the data from all files
    # And resample the spectrum and the continuum onto the shared grid
    for k, file in tqdm(enumerate(files), desc="File", total=len(files)):
        try:
            e = echelle.read(file, continuum_normalization=False)
            specs[k] = np.ma.filled(e.spec, 0)
            waves[k] = np.ma.getdata(e.wave)
            sigms[k] = np.ma.filled(e.sig, 1)
            conts[k] = np.ma.filled(e.cont, 0)
        except ValueError as ex:
            logger.warning("Error in loading file %s. %s", file, ex)
            mask[k] = False

    waves = waves[mask]
    specs = specs[mask]
    sigms = sigms[mask]
    conts = conts[mask]

    # wmin, wmax = waves.min(axis=(0, 2)), waves.max(axis=(0, 2))
    # wnew = np.geomspace(wmin, wmax, ncol, endpoint=True).T
    # TODO something weird happens when changing the wavelength grid, that also depends on the wavelength
    # Maybe points in the grid are interpreted differently, i.e. in the rebinning they are the center of the
    # bin, but later on they are the edges?
    wnew = np.copy(waves[0])

    for k in tqdm(range(specs.shape[0]), desc="File"):
        for i in tqdm(range(nord), desc="Order", leave=False):
            conts[k, i], _ = spectres(wnew[i], waves[k, i], conts[k, i], sigms[k, i])
            specs[k, i], sigms[k, i] = spectres(
                wnew[i], waves[k, i], specs[k, i], sigms[k, i]
            )

    # These are just for plotting
    if plot:
        sold = np.copy(specs[:, plot])
        cold = np.copy(conts[:, plot])

    # Median and MAD
    # We use the MAD over the whole range though, since we need some data points to properly evaluate it
    arr = specs / conts
    mean = np.nanmedian(arr, axis=0)
    std = np.nanmedian(np.abs(arr - mean), axis=[0, 2])[:, None]
    vmin, vmax = mean - 5 * std, mean + 5 * std

    # Disregard all values outside of 5 * MAD
    where = (arr < vmin) | (arr > vmax)
    specs[where] = 0
    conts[where] = 1
    sigms[where] = np.nan
    weights = 1 / sigms
    weights[np.isposinf(weights)] = np.sqrt(2)  # TODO Why
    weights[where] = 0

    w2 = np.sum(weights, axis=0) == 0
    weights[:, w2] = 1

    # Take the average of the spectrum and the continuum
    snew = np.average(specs, weights=weights, axis=0)
    cnew = np.average(conts, weights=weights, axis=0)
    snew = np.nan_to_num(snew, copy=False)
    # unew = 1 / np.sqrt(np.nansum((conts/sigms)**2, axis=0))

    # This is the uncertainty from the scatter
    unew = np.sqrt(np.nansum(weights * (arr - snew) ** 2, axis=0) / len(files))
    unew[unew == 0] = np.nansum(sigms, axis=0)[unew == 0]

    snew /= cnew
    cnew = np.ones_like(snew)

    if plot:
        for i in range(sold.shape[0]):
            plt.plot(wnew[plot], sold[i] / cold[i])
        plt.plot(wnew[plot], snew[plot], "--")
        plt.fill_between(wnew[plot], vmin[plot], vmax[plot], alpha=0.5)
        plt.show()

    e.spec = snew
    e.sig = unew
    e.cont = cnew
    e.mask = (snew == 0) | (cnew == 0)
    del e["columns"]

    e.header["barycorr"] = 0.0
    e.save(output)

    logger.info("Created combined file: %s", output)
