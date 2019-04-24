"""
Calculate the tilt based on a reference spectrum with high SNR, e.g. Wavelength calibration image

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
from scipy.optimize import least_squares

from skimage.filters import threshold_otsu

from .extract import fix_extraction_width, extend_orders, fix_column_range
from .util import make_index, gaussfit3 as gaussfit, gaussval2 as gaussval


def make_shear(
    extracted,
    original,
    orders,
    extraction_width=0.5,
    column_range=None,
    order_range=None,
    plot=False,
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

    logging.info("Extract tilt of the slit")

    nord = orders.shape[0]
    nrow, ncol = original.shape
    # how much SNR should a peak have to contribute (were Noise = median(img - min(img)))
    threshold = 10
    # The width around the line that is supposed to be extracted
    width = 9

    if order_range is None:
        order_range = (0, nord)
    if np.isscalar(extraction_width):
        extraction_width = np.tile([extraction_width], [nord, 2])
    if column_range is None:
        column_range = np.tile([0, ncol], [nord, 1])

    orders = extend_orders(orders, nrow)
    extraction_width = np.array(
        [extraction_width[0], *extraction_width, extraction_width[-1]]
    )
    column_range = np.array([column_range[0], *column_range, column_range[-1]])

    # Fix column range, so that all extractions are fully within the image
    extraction_width = fix_extraction_width(
        extraction_width, orders, column_range, ncol
    )
    column_range = fix_column_range(original, orders, extraction_width, column_range)

    orders = orders[1:-1]
    extraction_width = extraction_width[1:-1]
    column_range = column_range[1:-1]

    # Fit tilt with parabola
    n = order_range[1] - order_range[0]
    tilt_x = np.zeros((n, 3))
    shear_x = np.zeros((n, 3))

    if plot:
        fig, axes = plt.subplots(nrows=n // 2, ncols=2, squeeze=False)
        fig.suptitle("Peaks")
        fig2, axes2 = plt.subplots(nrows=n // 2, ncols=2, squeeze=False)
        fig2.suptitle("tilt")
        plt.subplots_adjust(hspace=0)

    for j, iord in enumerate(range(order_range[0], order_range[1])):
        if n < 10 or j % 5 == 0:
            logging.info("Calculating tilt of order %i out of %i", j + 1, n)
        else:
            logging.debug("Calculating tilt of order %i out of %i", j + 1, n)

        cr = np.where(extracted[j] > 0)[0][[0, -1]]
        cr = np.clip(cr, column_range[iord, 0], column_range[iord, 1])

        xwd = extraction_width[iord]
        height = np.sum(xwd) + 1
        ycen = np.polyval(orders[iord], np.arange(ncol)).astype(int)

        # This should probably be the same as in the wavelength calibration
        vec = extracted[j, cr[0] : cr[1]]
        vec -= np.ma.min(vec)
        vec = np.ma.filled(vec, 0)
        locmax, _ = signal.find_peaks(
            vec, height=np.ma.median(vec) * threshold, distance=10
        )

        # Remove peaks at the edge
        locmax = locmax[(locmax >= width + 1) & (locmax < len(vec) - width - 1)]

        if plot:
            axes[j // 2, j % 2].plot(vec)
            axes[j // 2, j % 2].plot(np.arange(len(vec))[locmax], vec[locmax], "+")
            axes[j // 2, j % 2].set_xlim([0, ncol])
            if j not in (order_range[1] - 1, order_range[1] - 2):
                axes2[j // 2, j % 2].get_xaxis().set_ticks([])

        # Remove the offset, due to vec being a subset of extracted
        locmax += cr[0]

        # look at +- 9 pixels around the line
        nmax = len(locmax)
        xx = np.arange(-width, width + 1)
        xcen = np.zeros(height)
        xind = np.arange(-xwd[0], xwd[1] + 1)
        tilt = np.zeros(nmax)
        shear = np.zeros(nmax)

        deviation = np.zeros(xind.size)
        # plt.show()

        # Determine tilt for each line seperately
        for iline in range(nmax):
            # Extract short horizontal strip for each row in extraction width
            # Then fit a gaussian to each row, to find the center of the line
            x = locmax[iline] + xx
            x = x[(x >= 0) & (x < ncol)]
            for i, irow in enumerate(xind):
                if np.any((ycen + irow)[x[0] : x[-1] + 1] >= original.shape[0]):
                    # This should never happen after we fixed the extraction width
                    xcen[i] = np.mean(x)
                    deviation[i] = 0
                    continue
                idx = make_index(ycen + irow, ycen + irow, x[0], x[-1] + 1)
                s = original[idx][0]

                if np.all(np.ma.getmask(s)):
                    # If this row is masked, this will happen
                    xcen[i] = np.mean(x)
                    deviation[i] = 0
                else:
                    coef = gaussfit(x, s)
                    xcen[i] = coef[1]  # Store line center
                    deviation[i] = np.ma.std(s)  # Store the variation within the row

            #     _s = s - np.mean(s)
            #     _s /= np.max(_s)
            #     _s *= 5
            #     _v = 5
            #     plt.plot(x, _s + irow)
            #     plt.plot(coef[1], _v + irow, "rx")
            #     plt.xlabel("Column")
            #     plt.ylabel("Row")
            # plt.show()

            # Seperate in order pixels from out of order pixels
            # TODO: actually we want to weight them by the slitfunction?
            idx = deviation > threshold_otsu(deviation)

            # Linear fit to slit image
            coef = np.polyfit(xind[idx], xcen[idx], 2)

            # plt.plot(xind, xcen, ".")
            # plt.plot(xind[idx], xcen[idx], "rx")
            # plt.plot(xind, line)
            # plt.show()

            tilt[iline] = coef[1]
            shear[iline] = coef[0]

        # Fit a 2nd order polynomial through all individual lines
        # And discard obvious outliers
        for _ in range(2):
            func = lambda c: np.polyval(c, locmax) - tilt
            res = least_squares(func, np.zeros(3), loss="soft_l1")
            coef_tilt = res.x

            line = np.polyval(coef_tilt, locmax)
            diff = np.abs(line - tilt)
            idx = diff < np.std(diff) * 5
            locmax = locmax[idx]
            tilt = tilt[idx]
            shear = shear[idx]
            if np.all(idx):
                break

        func = lambda c: np.polyval(c, locmax) - shear
        res = least_squares(func, np.zeros(3), loss="soft_l1")
        coef_shear = res.x

        # Fit a line through all individual shears along the order
        # coef = np.polyfit(locmax, shear, 2)
        tilt_x[j] = coef_tilt
        shear_x[j] = coef_shear

        if plot:
            x = np.arange(cr[0], cr[1])
            axes2[j // 2, j % 2].plot(locmax, tilt, "rx")
            axes2[j // 2, j % 2].plot(x, np.polyval(coef_tilt, x))
            axes2[j // 2, j % 2].set_xlim(0, ncol)
            if j not in (order_range[1] - 1, order_range[1] - 2):
                axes2[j // 2, j % 2].get_xaxis().set_ticks([])

    if plot:
        plt.show()

    tilt = np.zeros((n, ncol))
    shear = np.zeros((n, ncol))
    for j in range(order_range[0], order_range[1]):
        tilt[j] = np.polyval(tilt_x[j], np.arange(ncol))
        shear[j] = np.polyval(shear_x[j], np.arange(ncol))
    return tilt, shear
