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


def fix_inputs(original, orders, extraction_width, column_range):
    nord = len(orders)
    nrow, ncol = original.shape

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

    extraction_width = extraction_width[1:-1]
    column_range = column_range[1:-1]

    return extraction_width, column_range


def find_peaks(vec, cr, threshold, width):
    # This should probably be the same as in the wavelength calibration
    vec -= np.ma.min(vec)
    vec = np.ma.filled(vec, 0)
    height = np.quantile(vec, 0.1) * threshold
    peaks, _ = signal.find_peaks(vec, height=height)

    # Remove peaks at the edge
    peaks = peaks[(peaks >= width + 1) & (peaks < len(vec) - width - 1)]
    # Remove the offset, due to vec being a subset of extracted
    peaks += cr[0]
    return vec, peaks


def determine_curvature_single_line(original, peak, ycen, width, xwd):
    nrow, ncol = original.shape
    height = np.sum(xwd) + 1

    # look at +- width pixels around the line
    #:array of shape (2*width + 1,): indices of the pixels to the left and right of the line peak
    index_x = np.arange(-width, width + 1)
    #:array of shape (height,): stores the peak positions of the fits to each row
    xcen = np.zeros(height)
    #:array of shape (height,): indices of the rows in the order, with 0 being the central row
    xind = np.arange(-xwd[0], xwd[1] + 1)
    #:array of shape (height,): Scatter of the values within the row, to seperate in order and out of order rows
    deviation = np.zeros(height)

    # Extract short horizontal strip for each row in extraction width
    # Then fit a gaussian to each row, to find the center of the line
    x = peak + index_x
    x = x[(x >= 0) & (x < ncol)]
    for i, irow in enumerate(xind):
        # Trying to access values outside the image
        assert not np.any((ycen + irow)[x[0] : x[-1] + 1] >= nrow)

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

    tilt, shear = coef[1], coef[0]
    return tilt, shear


def polyfit(x, y, deg):
    # func = lambda c: np.polyval(c, x) - y
    # res = least_squares(func, np.zeros(deg + 1), loss="soft_l1")
    # coef = res.x
    coef = np.ma.polyfit(x, y, deg)
    return coef


def fit_curvature_single_order(peaks, tilt, shear, fit_degree, max_iter, sigma=3):

    # Make them masked arrays to avoid copying the data all the time
    # Updating the mask updates all of them (as it is not copied)
    mask = np.full(peaks.shape, False)
    peaks = np.ma.masked_array(peaks, mask=mask)
    tilt = np.ma.masked_array(tilt, mask=mask)
    shear = np.ma.masked_array(shear, mask=mask)

    # Fit a 2nd order polynomial through all individual lines
    # And discard obvious outliers
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        coef_tilt = polyfit(peaks, tilt, fit_degree)
        coef_shear = polyfit(peaks, shear, fit_degree)

        diff = np.polyval(coef_tilt, peaks) - tilt
        idx1 = np.ma.abs(diff) >= np.ma.std(diff) * sigma
        mask |= idx1

        diff = np.polyval(coef_shear, peaks) - shear
        idx2 = np.ma.abs(diff) >= np.ma.std(diff) * sigma
        mask |= idx2

        # if no maximum iteration is given, go on forever
        if np.ma.all(~idx1) and np.ma.all(~idx2):
            break
        if np.all(mask):
            raise ValueError("Could not fit polynomial to the data")

    coef_tilt = polyfit(peaks, tilt, fit_degree)
    coef_shear = polyfit(peaks, shear, fit_degree)

    # x = np.arange(4096)
    # plt.plot(peaks, tilt, "rx")
    # plt.plot(x, np.polyval(coef_tilt, x))
    # plt.show()

    return coef_tilt, coef_shear, peaks


def plot_results(
    ncol,
    order_range,
    column_range,
    plot_peaks,
    plot_vec,
    plot_tilt,
    plot_shear,
    tilt_x,
    shear_x,
):
    n = order_range[1] - order_range[0]

    fig, axes = plt.subplots(nrows=n // 2, ncols=2, squeeze=False)
    fig.suptitle("Peaks")
    fig1, axes1 = plt.subplots(nrows=n // 2, ncols=2, squeeze=False)
    fig1.suptitle("1st Order Curvature")
    fig2, axes2 = plt.subplots(nrows=n // 2, ncols=2, squeeze=False)
    fig2.suptitle("2nd Order Curvature")
    plt.subplots_adjust(hspace=0)

    for j, iord in enumerate(range(order_range[0], order_range[1])):
        cr = column_range[iord]
        peaks = plot_peaks[j]
        vec = plot_vec[j]
        tilt = plot_tilt[j]
        shear = plot_shear[j]
        x = np.arange(cr[0], cr[1])

        # Figure Peaks found (and used)
        axes[j // 2, j % 2].plot(vec)
        axes[j // 2, j % 2].plot(
            np.arange(len(vec))[peaks - cr[0]], vec[peaks - cr[0]], "+"
        )
        axes[j // 2, j % 2].set_xlim([0, ncol])
        if j not in (n - 1, n - 2):
            axes2[j // 2, j % 2].get_xaxis().set_ticks([])

        # Figure 1st order
        axes1[j // 2, j % 2].plot(peaks, tilt, "rx")
        axes1[j // 2, j % 2].plot(x, np.polyval(tilt_x[j], x))
        axes1[j // 2, j % 2].set_xlim(0, ncol)
        if j not in (n - 1, n - 2):
            axes2[j // 2, j % 2].get_xaxis().set_ticks([])

        # Figure 2nd order
        axes2[j // 2, j % 2].plot(peaks, shear, "rx")
        axes2[j // 2, j % 2].plot(x, np.polyval(shear_x[j], x))
        axes2[j // 2, j % 2].set_xlim(0, ncol)
        if j not in (n - 1, n - 2):
            axes2[j // 2, j % 2].get_xaxis().set_ticks([])

    plt.show()


def make_shear(
    extracted,
    original,
    orders,
    extraction_width=0.5,
    column_range=None,
    order_range=None,
    width=9,
    threshold=10,
    fit_degree=2,
    sigma_cutoff=3,
    max_iter=None,
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

    if max_iter is None:
        max_iter = np.inf

    nord = orders.shape[0]
    nrow, ncol = original.shape

    if order_range is None:
        order_range = (0, nord)

    extraction_width, column_range = fix_inputs(
        original, orders, extraction_width, column_range
    )

    # Fit tilt with parabola
    n = order_range[1] - order_range[0]
    tilt_x = np.zeros((n, fit_degree + 1))
    shear_x = np.zeros((n, fit_degree + 1))

    # plotting
    plot_vec = []
    plot_peaks = []
    plot_tilt = []
    plot_shear = []

    for j, iord in enumerate(range(order_range[0], order_range[1])):
        if n < 10 or j % 5 == 0:
            logging.info("Calculating tilt of order %i out of %i", j + 1, n)
        else:
            logging.debug("Calculating tilt of order %i out of %i", j + 1, n)

        cr = column_range[iord]
        xwd = extraction_width[iord]
        ycen = np.polyval(orders[iord], np.arange(ncol)).astype(int)

        # This should probably be the same as in the wavelength calibration
        vec = extracted[j, cr[0] : cr[1]]
        vec, peaks = find_peaks(vec, cr, threshold, width)

        npeaks = len(peaks)
        if npeaks < fit_degree + 1:
            raise ValueError(
                f"Not enough peaks found to fit a polynomial of degree {fit_degree}"
            )
        # 1st order curvature for each peak
        tilt = np.zeros(npeaks)
        # 2nd order curvature for each peak
        shear = np.zeros(npeaks)

        # Determine tilt for each line seperately
        for ipeak, peak in enumerate(peaks):
            tilt[ipeak], shear[ipeak] = determine_curvature_single_line(
                original, peak, ycen, width, xwd
            )

        tilt_x[j], shear_x[j], peaks = fit_curvature_single_order(
            peaks, tilt, shear, fit_degree, max_iter, sigma=sigma_cutoff
        )

        if plot:
            plot_vec += [vec]
            plot_peaks += [peaks]
            plot_tilt += [tilt]
            plot_shear += [shear]

    if plot:
        plot_results(
            ncol,
            order_range,
            column_range,
            plot_peaks,
            plot_vec,
            plot_tilt,
            plot_shear,
            tilt_x,
            shear_x,
        )

    # TODO do a 2D fit instead of a 1D fit, as all orders should have similar curvature?

    tilt = np.zeros((n, ncol))
    shear = np.zeros((n, ncol))
    for j in range(n):
        tilt[j] = np.polyval(tilt_x[j], np.arange(ncol))
        shear[j] = np.polyval(shear_x[j], np.arange(ncol))
    return tilt, shear
