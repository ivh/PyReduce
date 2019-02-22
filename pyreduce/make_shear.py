"""
Calculate the shear based on a reference spectrum with high SNR, e.g. Wavelength calibration image

Authors
-------
Nikolai Piskunov
Ansgar Wehrhahn

Version
--------
0.9 - NP - IDL Version
1.0 - AW - Python Version

License
-------
....
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

from .extract import fix_extraction_width
from .util import make_index, gaussfit2 as gaussfit


def make_shear(
    extracted, original, orders, extraction_width=0.5, column_range=None, plot=False
):
    """ Calculate the shear/tilt of the slit along each order
    Determine strong spectral lines
    For each line:
    - Determine center of line along a number of rows around the center
    - Fit line to the centers along the rows == shear at that position
    - Fit curve to the shears along the order (2nd degree polynomial)
    - Calculate slit for all points

    Parameters
    ----------
    extracted : array[nord, ncol]
        already extracted image
    original : array[nrow, ncol]
        original image, the basis for extracted
    orders : array[nord, degree]
        order trace coefficients
    extraction_width : {float, array[nord, 2]}, optional
        extraction width per order, values below 1.5 are considered fractions (default: 0.5)
    column_range : array[nord, 2], optional
        columns that are part of the order (default: use all columns)
    plot : bool, optional
        wether to plot the results (default: False)
    
    Returns
    -------
    shear : array[nord, ncol]
        shear along the slit
    """

    logging.info("Extract shear of the slit")

    nord = orders.shape[0]
    nrow, ncol = original.shape
    threshold = (
        10
    )  # how much SNR should a peak have to contribute (were Noise = median(img - min(img)))

    if np.isscalar(extraction_width):
        extraction_width = np.tile([extraction_width], [nord, 2])
    if column_range is None:
        column_range = np.tile([0, ncol], [nord, 1])

    extraction_width = fix_extraction_width(
        extraction_width, orders, column_range, ncol
    )

    # Fit shear with parabola
    shear_x = np.zeros((nord, 3))

    if plot:
        fig, axes = plt.subplots(nrows=nord // 2, ncols=2)
        fig.suptitle("Peaks")
        fig2, axes2 = plt.subplots(nrows=nord // 2, ncols=2)
        fig2.suptitle("Shear")
        plt.subplots_adjust(hspace=0)

    for iord in range(nord):
        cr = np.where(extracted[iord] > 0)[0][[0, -1]]
        cr = np.clip(cr, column_range[iord, 0], column_range[iord, 1])

        xwd = extraction_width[iord]
        height = np.sum(xwd) + 1
        ycen = np.polyval(orders[iord], np.arange(ncol)).astype(int)

        # This should probably be the same as in the wavelength calibration
        vec = extracted[iord, cr[0] : cr[1]]
        vec -= np.ma.min(vec)
        locmax, _ = signal.find_peaks(
            vec, height=np.ma.median(vec) * threshold, distance=10
        )

        # Remove peaks at the edge
        locmax = locmax[(locmax >= 10) & (locmax < len(vec) - 10)]

        # Keep not more than 20 strongest lines per order
        i = np.argsort(vec[locmax])[::-1]
        i = i[:20]
        nmax = len(i)
        locmax = locmax[i]
        locmax = locmax[::-1]  # sort again

        if plot:
            axes[iord // 2, iord % 2].plot(vec)
            axes[iord // 2, iord % 2].plot(
                np.arange(len(vec))[locmax], vec[locmax], "+"
            )

        # Remove the offset, due to vec being a subset of extracted
        locmax += cr[0]

        # look at +- 9 pixels around the line
        xx = np.arange(-9, 10)
        xcen = np.zeros(height)
        xind = np.arange(-xwd[0], xwd[1] + 1)
        shear = np.zeros(nmax)

        # Determine shear for each order seperately
        for iline in range(nmax):
            # Extract short horizontal strip for each row in extraction width
            # Then fit a gaussian to each row, to find the center of the line
            x = locmax[iline] + xx
            x = x[(x >= 0) & (x < ncol)]
            for irow in xind:
                idx = make_index(ycen + irow, ycen + irow, x[0], x[-1] + 1)
                s = original[idx][0]
                if np.all(np.ma.getmask(s)):
                    # If this row is masked, this will happen
                    xcen[irow + xwd[0]] = np.mean(x)
                else:
                    s -= np.min(s)
                    coef = gaussfit(x, s)
                    xcen[irow + xwd[0]] = coef[1]  # Store line center

            # Fit a line through the line centers, along the rows
            perc = int(len(xind) * 0.8)
            xind_loc, xcen_loc = xind, xcen
            for _ in range(2):
                # Linear fit to slit image
                coef = np.polyfit(xind_loc, xcen_loc, 1)
                line = np.polyval(coef, xind)
                # Remove outliers
                j = np.argsort(np.abs(line - xcen))
                xind_loc = xind[j[:perc]]
                xcen_loc = xcen[j[:perc]]

            # Final fit using best 80%
            coef = np.polyfit(xind_loc, xcen_loc, 1)
            shear[iline] = coef[0]  # Store line shear

        # Fit a line through all individual shears along the order
        perc = int(len(shear) * 0.8)
        locmax_loc, shear_loc = locmax, shear
        for _ in range(2):
            coef = np.polyfit(locmax_loc, shear_loc, 1)
            line = np.polyval(coef, locmax)
            j = np.argsort(np.abs(line - shear))
            locmax_loc = locmax[j[:perc]]
            shear_loc = shear[j[:perc]]

        a = np.polyfit(locmax_loc, shear_loc, 2)
        shear_x[iord] = a

        if plot:
            x = np.arange(cr[0], cr[1])
            axes2[iord // 2, iord % 2].plot(x, np.polyval(a, x))
            axes2[iord // 2, iord % 2].set_xlim(0, ncol)

    if plot:
        plt.show()

    shear = np.zeros((nord, ncol))
    for iord in range(nord):
        shear[iord] = np.polyval(shear_x[iord], np.arange(ncol))
    return shear
