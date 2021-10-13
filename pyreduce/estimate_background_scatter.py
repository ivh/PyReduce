# -*- coding: utf-8 -*-
"""
Module that estimates the background scatter
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from .extract import fix_extraction_width, fix_parameters
from .util import make_index, polyfit2d, polyfit2d_2

logger = logging.getLogger(__name__)


def estimate_background_scatter(
    img,
    orders,
    column_range=None,
    extraction_width=0.1,
    scatter_degree=4,
    sigma_cutoff=2,
    border_width=10,
    plot=False,
    plot_title=None,
):
    """
    Estimate the background by fitting a 2d polynomial to interorder data

    Interorder data is all pixels minus the orders +- the extraction width

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

    extraction_width, column_range, orders = fix_parameters(
        extraction_width,
        column_range,
        orders,
        nrow,
        ncol,
        nord,
        ignore_column_range=True,
    )

    # Method 1: Select all pixels, but those known to be in orders
    bw = border_width
    mask = np.full(img.shape, True)
    if bw is not None and bw != 0:
        mask[:bw] = mask[-bw:] = mask[:, :bw] = mask[:, -bw:] = False
    for i in range(nord):
        left, right = column_range[i]
        left -= extraction_width[i, 1] * 2
        right += extraction_width[i, 0] * 2
        left = max(0, left)
        right = min(ncol, right)

        x_order = np.arange(left, right)
        y_order = np.polyval(orders[i], x_order)

        y_above = y_order + extraction_width[i, 1]
        y_below = y_order - extraction_width[i, 0]

        y_above = np.floor(y_above)
        y_below = np.ceil(y_below)

        index = make_index(y_below, y_above, left, right, zero=True)
        np.clip(index[0], 0, nrow - 1, out=index[0])

        mask[index] = False

    mask &= ~np.ma.getmask(img)

    y, x = np.indices(mask.shape)
    y, x = y[mask].ravel(), x[mask].ravel()
    z = np.ma.getdata(img[mask]).ravel()

    mask = z <= np.median(z) + sigma_cutoff * z.std()
    y, x, z = y[mask], x[mask], z[mask]

    coeff = polyfit2d(x, y, z, degree=scatter_degree, plot=plot, plot_title=plot_title)
    logger.debug("Background scatter coefficients: %s", str(coeff))

    if plot:  # pragma: no cover
        # Calculate scatter at interorder positionsq
        yp, xp = np.indices(img.shape)
        back = np.polynomial.polynomial.polyval2d(xp, yp, coeff)

        plt.subplot(121)
        plt.title("Input Image + In-between Order traces")
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        vmin, vmax = np.percentile(img - back, (5, 95))
        plt.imshow(img - back, vmin=vmin, vmax=vmax, aspect="equal", origin="lower")
        plt.plot(x, y, ",")

        plt.subplot(122)
        plt.title("2D fit to the scatter between orders")
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        plt.imshow(back, vmin=0, vmax=abs(np.max(back)), aspect="equal", origin="lower")

        if plot_title is not None:
            plt.suptitle(plot_title)
        plt.show()

    return coeff
