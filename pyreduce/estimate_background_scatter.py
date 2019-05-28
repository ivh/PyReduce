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
TODO

"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from . import extract
from .util import polyfit2d, polyfit2d_2, make_index


def estimate_background_scatter(
    img,
    orders,
    column_range=None,
    extraction_width=0.1,
    scatter_degree=4,
    plot=False,
    **kwargs
):
    """
    Estimate the background by fitting a 2d polynomial to interorder data

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

    if not isinstance(scatter_degree, (int, np.integer, tuple)):
        raise TypeError(
            "Expected integer value for scatter polynomial degree, got %s"
            % type(scatter_degree)
        )
    if isinstance(scatter_degree, tuple):
        if len(scatter_degree) != 2:
            raise ValueError(
                "Expected tuple of length 2, but got length %i" % len(scatter_degree)
            )
        types = [isinstance(i, (int, np.integer)) for i in scatter_degree]
        if not all(types):
            raise TypeError(
                "Expected integer value for scatter polynomial degree, got %s"
                % type(scatter_degree)
            )
        values = [i < 0 for i in scatter_degree]
        if any(values):
            raise ValueError(
                "Expected positive value for scatter polynomial degree, got %s"
                % str(scatter_degree)
            )
    elif scatter_degree < 0:
        raise ValueError(
            "Expected positive value for scatter polynomial degree, got %i"
            % scatter_degree
        )

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

    # column_range = extract.fix_column_range(img, orders, extraction_width, column_range)

    # determine points inbetween orders
    x_inbetween = [None for _ in range(nord + 1)]
    y_inbetween = [None for _ in range(nord + 1)]
    z_inbetween = [None for _ in range(nord + 1)]

    for i, j in zip(range(nord + 1), range(1, nord + 2)):
        left = max(column_range[[i, j], 0])
        right = min(column_range[[i, j], 1])

        x_order = np.arange(left, right)
        y_below = np.polyval(orders[i], x_order)
        y_above = np.polyval(orders[j], x_order)

        y_below += extraction_width[i, 1]
        y_above -= extraction_width[j, 0]

        y_above = np.floor(y_above)
        y_below = np.ceil(y_below)

        within_img = (y_below < nrow) & (y_above >= 0)
        left, right = x_order[within_img][[0, -1]]
        y_below = np.clip(y_below[within_img], 0, nrow - 1)
        y_above = np.clip(y_above[within_img], 0, nrow - 1)

        index = make_index(y_below, y_above, left, right, zero=True)

        y = np.concatenate(index[0])
        x = np.concatenate(index[1])
        sub_img = img[(y, x)]

        threshold = np.ma.median(sub_img) + 5 * np.ma.std(sub_img)

        mask = (~np.ma.getmaskarray(sub_img)) & (sub_img <= threshold)
        x_inbetween[i] = x[mask]
        y_inbetween[i] = y[mask]
        z_inbetween[i] = np.ma.getdata(sub_img[mask]).ravel()

        # plt.title("Between %i and %i" % (i, j))
        # plt.imshow(sub_img, aspect="auto")
        # plt.show()

    # Sanitize input into desired flat shape
    x = np.concatenate(x_inbetween)
    y = np.concatenate(y_inbetween)
    z = np.concatenate(z_inbetween)

    coeff = polyfit2d(x, y, z, degree=scatter_degree, plot=plot)
    logging.debug("Background scatter coefficients: %s", str(coeff))

    if plot:
        # Calculate scatter at interorder positionsq
        y, x = np.indices(img.shape)
        back = np.polynomial.polynomial.polyval2d(x, y, coeff)

        plt.subplot(211)
        plt.title("Input Image + In-between Order traces")
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        plt.imshow(
            img - back, vmin=0, vmax=np.max(back), aspect="equal", origin="lower"
        )
        for i in range(len(x_inbetween)):
            plt.plot(x_inbetween[i], y_inbetween[i], ".")

        plt.subplot(212)
        plt.title("2D fit to the scatter between orders")
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        plt.imshow(back, vmin=0, vmax=np.max(back), aspect="equal", origin="lower")
        plt.show()

    return coeff
