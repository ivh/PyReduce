"""
Module that estimates the background scatter

Authors
-------
Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)

Version
-------
1.0 - 2d polynomial background scatter

License
-------
....

"""

import logging

import matplotlib.pyplot as plt
import numpy as np

import extract
from util import polyfit2d


def estimate_background_scatter(img, orders, column_range=None, extraction_width=0.1, scatter_degree=4, plot=False):
    """Estimate the background by fitting a 2d polynomial to interorder data

    Parameters
    ----------
    img : array[nrow, ncol]
        (flat) image data
    orders : array[nord, degree]
        order polynomial coefficients
    column_range : array[nord, 2], optional
        range of columns to use in each order (default: None == all columns)
    extraction_width : float, array[nord, 2], optional
        extraction width for each order, values below 1.5 are considered fractional, others as number of pixels (default: 0.1)
    scatter_degree : int, optional
        polynomial degree of the 2d fit for the background scatter (default: 4)
    plot : bool, optional
        wether to plot the fitted polynomial and the data or not (default: False)

    Returns
    -------
    array[nord+1, ncol]
        background scatter between orders
    array[nord+1, ncol]
        y positions of the interorder lines, the scatter values are taken from
    """

    nrow, ncol = img.shape
    nord, _ = orders.shape

    if np.isscalar(extraction_width):
        extraction_width = np.tile([extraction_width, extraction_width], (nord, 1))
    if column_range is None:
        column_range = np.tile([0, ncol], (nord, 1))
    # Extend orders above and below orders
    orders = extract.extend_orders(orders, nrow)
    extraction_width = np.array(
        [extraction_width[0], *extraction_width, extraction_width[-1]]
    )
    column_range = np.array([column_range[0], *column_range, column_range[-1]])

    extraction_width = extract.fix_extraction_width(
        extraction_width, orders, column_range, ncol
    )
    column_range = extract.fix_column_range(img, orders, extraction_width, column_range)

    # determine points inbetween orders
    x_inbetween = [None for _ in range(nord + 1)]
    y_inbetween = [None for _ in range(nord + 1)]
    orders_inbetween = [None for _ in range(nord + 1)]

    for i, j in zip(range(nord + 1), range(1, nord + 2)):
        left = max(column_range[[i, j], 0])
        right = min(column_range[[i, j], 1])

        x_order = np.arange(left, right)
        y_below = np.polyval(orders[i], x_order)
        y_above = np.polyval(orders[j], x_order)
        y_order = (y_below + y_above) / 2

        x_inbetween[i] = x_order[(y_order > 0) & (y_order < nrow)]
        y_inbetween[i] = y_order[(y_order > 0) & (y_order < nrow)]
        orders_inbetween[i] = np.polyfit(x_order, y_order, 2)

    # Sanitize input into desired flat shape
    x = np.concatenate(x_inbetween)
    y = np.concatenate(y_inbetween).astype(int)
    z = img[y, x].flatten()

    coeff = polyfit2d(x, y, z, degree=scatter_degree, plot=plot)
    logging.debug("Background scatter coefficients: %s", str(coeff))

    # Calculate scatter at interorder positions
    x = np.arange(ncol)
    y = np.array([np.polyval(order, x) for order in orders_inbetween])
    x = x[None, :] * np.full(nord+1, 1)[:, None]
    back = np.polynomial.polynomial.polyval2d(x, y, coeff)
    yback = y

    if plot:
        plt.subplot(211)
        plt.imshow(img, vmax=np.max(back))
        for i in range(x.shape[0]):
            plt.plot(x[i], y[i])
        plt.subplot(212)
        plt.imshow(back, aspect="auto")
        plt.show()

    return back, yback
