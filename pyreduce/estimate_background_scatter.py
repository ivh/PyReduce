"""
Module that estimates the background scatter
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from . import util
from .extract import fix_parameters
from .util import make_index, polyfit2d

logger = logging.getLogger(__name__)


def estimate_background_scatter(
    img,
    traces,
    column_range=None,
    extraction_height=0.1,
    scatter_degree=4,
    sigma_cutoff=2,
    border_width=10,
    plot=False,
    plot_title=None,
):
    """
    Estimate the background by fitting a 2d polynomial to inter-trace data

    Inter-trace data is all pixels minus the traces +- the extraction width

    Parameters
    ----------
    img : array[nrow, ncol]
        (flat) image data
    traces : array[ntrace, degree]
        trace polynomial coefficients
    column_range : array[ntrace, 2], optional
        range of columns to use in each trace (default: None == all columns)
    extraction_height : float, array[ntrace, 2], optional
        extraction width for each trace, values below 1.5 are considered fractional, others as number of pixels (default: 0.1)
    scatter_degree : int, optional
        polynomial degree of the 2d fit for the background scatter (default: 4)
    plot : bool, optional
        wether to plot the fitted polynomial and the data or not (default: False)

    Returns
    -------
    array[ntrace+1, ncol]
        background scatter between traces
    array[ntrace+1, ncol]
        y positions of the inter-trace lines, the scatter values are taken from
    """

    nrow, ncol = img.shape
    ntrace, _ = traces.shape

    extraction_height, column_range, traces = fix_parameters(
        extraction_height,
        column_range,
        traces,
        nrow,
        ncol,
        ntrace,
        ignore_column_range=True,
    )

    # Method 1: Select all pixels, but those known to be in traces
    bw = border_width
    mask = np.full(img.shape, True)
    if bw is not None and bw != 0:
        mask[:bw] = mask[-bw:] = mask[:, :bw] = mask[:, -bw:] = False
    for i in range(ntrace):
        left, right = column_range[i]
        left -= extraction_height[i, 1] * 2
        right += extraction_height[i, 0] * 2
        left = max(0, left)
        right = min(ncol, right)

        x_trace = np.arange(left, right)
        y_trace = np.polyval(traces[i], x_trace)

        y_above = y_trace + extraction_height[i, 1]
        y_below = y_trace - extraction_height[i, 0]

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
        plt.figure()
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
        util.show_or_save("scatter")

    return coeff
