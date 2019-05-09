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
from .util import make_index, gaussfit4 as gaussfit, gaussval2 as gaussval


def make_shear(
    extracted,
    original,
    orders,
    extraction_width=0.5,
    column_range=None,
    order_range=None,
    width=9,
    threshold=10,
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
    width : int, optional
        The width around each individual line peak that is used; approx hwhm (default: 9)
        The full area is 2 * width + 1  (one for the central column)
    threshold: float, optional
        how much SNR should a peak have to contribute (were Noise = median(img - min(img)))
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

        cr = column_range[iord]
        xwd = extraction_width[iord]
        height = np.sum(xwd) + 1
        ycen = np.polyval(orders[iord], np.arange(ncol)).astype(int)

        # This should probably be the same as in the wavelength calibration
        vec = extracted[j, cr[0] : cr[1]]
        vec -= np.ma.min(vec)
        vec = np.ma.filled(vec, 0)
        peaks, _ = signal.find_peaks(
            vec, height=np.ma.median(vec) * threshold, distance=10
        )

        # Remove peaks at the edge
        peaks = peaks[(peaks >= width + 1) & (peaks < len(vec) - width - 1)]
        # Remove the offset, due to vec being a subset of extracted
        peaks += cr[0]

        if plot:
            axes[j // 2, j % 2].plot(vec)
            axes[j // 2, j % 2].plot(
                np.arange(len(vec))[peaks - cr[0]], vec[peaks - cr[0]], "+"
            )
            axes[j // 2, j % 2].set_xlim([0, ncol])
            if j not in (n - 1, n - 2):
                axes2[j // 2, j % 2].get_xaxis().set_ticks([])

        # look at +- width pixels around the line
        #:array of shape (2*width + 1,): indices of the pixels to the left and right of the line peak
        index_x = np.arange(-width, width + 1)
        #:array of shape (height,): stores the peak positions of the fits to each row
        xcen = np.zeros(height)
        #:array of shape (height,): indices of the rows in the order, with 0 being the central row
        xind = np.arange(-xwd[0], xwd[1] + 1)
        #:array of shape (height,): Scatter of the values within the row, to seperate in order and out of order rows
        deviation = np.zeros(height)

        npeaks = len(peaks)
        #:array of shape (npeaks,): 1st order curvature for each peak
        tilt = np.zeros(npeaks)
        #:array of shape (npeaks,): 2nd order curvature for each peak
        shear = np.zeros(npeaks)

        # Determine tilt for each line seperately
        for ipeak, peak in enumerate(peaks):
            # Extract short horizontal strip for each row in extraction width
            # Then fit a gaussian to each row, to find the center of the line
            x = peak + index_x
            x = x[(x >= 0) & (x < ncol)]
            for i, irow in enumerate(xind):
                # Trying to access values outside the image
                assert not np.any((ycen + irow)[x[0] : x[-1] + 1] >= original.shape[0])

                # Just cutout this one row
                idx = make_index(ycen + irow, ycen + irow, x[0], x[-1] + 1)
                s = original[idx][0]

                if np.all(np.ma.getmask(s)):
                    # If this row is masked, this will happen
                    # It will be ignored by the thresholding anyway
                    xcen[i] = np.mean(x)
                    deviation[i] = 0
                else:
                    try:
                        coef = gaussfit(x, s)
                        # Store line center
                        xcen[i] = coef[1]
                        # Store the variation within the row
                        deviation[i] = np.ma.std(s)
                    except:
                        xcen[i] = np.mean(x)
                        deviation[i] = 0

            # Seperate in order pixels from out of order pixels
            # TODO: actually we want to weight them by the slitfunction?
            idx = deviation > threshold_otsu(deviation)

            # Linear fit to slit image
            coef = np.polyfit(xind[idx], xcen[idx], 2)

            tilt[ipeak] = coef[1]
            shear[ipeak] = coef[0]

        # Fit a 2nd order polynomial through all individual lines
        # And discard obvious outliers
        for _ in range(2):
            func = lambda c: np.polyval(c, peaks) - tilt
            res = least_squares(func, np.zeros(3), loss="soft_l1")
            coef_tilt = res.x

            line = np.polyval(coef_tilt, peaks)
            diff = np.abs(line - tilt)
            idx = diff < np.std(diff) * 5
            peaks = peaks[idx]
            tilt = tilt[idx]
            shear = shear[idx]
            if np.all(idx):
                break

        func = lambda c: np.polyval(c, peaks) - shear
        res = least_squares(func, np.zeros(3), loss="soft_l1")
        coef_shear = res.x

        # Fit a line through all individual shears along the order
        # coef = np.polyfit(peaks, shear, 2)
        tilt_x[j] = coef_tilt
        shear_x[j] = coef_shear

        if plot:
            x = np.arange(cr[0], cr[1])
            axes2[j // 2, j % 2].plot(peaks, tilt, "rx")
            axes2[j // 2, j % 2].plot(x, np.polyval(coef_tilt, x))
            axes2[j // 2, j % 2].set_xlim(0, ncol)
            if j not in (n - 1, n - 2):
                axes2[j // 2, j % 2].get_xaxis().set_ticks([])

    if plot:
        plt.show()

    tilt = np.zeros((n, ncol))
    shear = np.zeros((n, ncol))
    for j in range(n):
        tilt[j] = np.polyval(tilt_x[j], np.arange(ncol))
        shear[j] = np.polyval(shear_x[j], np.arange(ncol))
    return tilt, shear
