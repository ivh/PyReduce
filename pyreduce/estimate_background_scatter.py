"""
Module that estimates the background scatter
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from . import util
from .util import make_index, polyfit2d

logger = logging.getLogger(__name__)


def estimate_background_scatter(
    img,
    traces,
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
    traces : list[Trace]
        Trace objects with pos, column_range attributes
    extraction_height : float, optional
        extraction full height, values below 2 are considered fractional (default: 0.1)
    scatter_degree : int, optional
        polynomial degree of the 2d fit for the background scatter (default: 4)
    plot : bool, optional
        wether to plot the fitted polynomial and the data or not (default: False)

    Returns
    -------
    coeff : array
        2D polynomial coefficients for background scatter
    """

    nrow, ncol = img.shape

    # Compute extraction height in pixels if fractional
    xwd = extraction_height
    if xwd is not None and xwd < 3:
        # Fraction of order spacing - estimate from trace separation
        x_mid = ncol // 2
        y_mids = np.array([np.polyval(t.pos, x_mid) for t in traces])
        if len(y_mids) > 1:
            spacing = np.median(np.abs(np.diff(np.sort(y_mids))))
            xwd = xwd * spacing
        else:
            xwd = 10  # fallback

    # Method 1: Select all pixels, but those known to be in traces
    bw = border_width
    mask = np.full(img.shape, True)
    if bw is not None and bw != 0:
        mask[:bw] = mask[-bw:] = mask[:, :bw] = mask[:, -bw:] = False

    for trace in traces:
        left, right = trace.column_range
        left = int(max(0, left))
        right = int(min(ncol, right))

        x_trace = np.arange(left, right)
        y_trace = np.polyval(trace.pos, x_trace)

        # Compute aperture with fixed height
        half = int(np.ceil(xwd / 2))
        height = 2 * half  # constant height
        y_center = np.round(y_trace).astype(int)
        y_below = y_center - half
        y_above = y_below + height - 1  # ensures constant height

        # Find columns where full aperture fits within image
        valid = (y_below >= 0) & (y_above < nrow)
        if not np.any(valid):
            continue

        valid_idx = np.where(valid)[0]
        new_left = left + valid_idx[0]
        new_right = left + valid_idx[-1] + 1

        y_below = y_below[valid]
        y_above = y_above[valid]

        index = make_index(y_below, y_above, new_left, new_right, zero=True)

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
