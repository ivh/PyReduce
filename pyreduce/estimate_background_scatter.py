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
    sigma_cutoff=2,
    border_width=10,
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
    column_range = column_range[1:-1]
    orders = orders[1:-1]
    extraction_width = extraction_width[1:-1]

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

    coeff = polyfit2d(x, y, z, degree=scatter_degree, plot=plot)
    logging.debug("Background scatter coefficients: %s", str(coeff))

    if plot:
        # Calculate scatter at interorder positionsq
        yp, xp = np.indices(img.shape)
        back = np.polynomial.polynomial.polyval2d(xp, yp, coeff)

        plt.subplot(121)
        plt.title("Input Image + In-between Order traces")
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        plt.imshow(img - back, aspect="equal", origin="lower")
        plt.plot(x, y, ",")

        plt.subplot(122)
        plt.title("2D fit to the scatter between orders")
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        plt.imshow(back, vmin=0, vmax=np.max(back), aspect="equal", origin="lower")
        plt.show()

    return coeff
