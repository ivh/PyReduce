"""
Module that estimates the background scatter
"""

import logging
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from . import util
from .util import polyfit2d

logger = logging.getLogger(__name__)


@dataclass
class ScatterModel:
    """Background scatter model, together with the recipe that produced it.

    Scattered light scales with the illumination of the frame it was measured on,
    so ``coeff`` is only valid for ``reference``. A consumer that wants to correct a
    *different* frame must call :meth:`refit` on that frame rather than reuse
    ``coeff`` -- a master flat is typically hundreds of times brighter than a single
    science exposure, and nothing in the stored coefficients records that.
    """

    #:array: 2D polynomial coefficients, in the flux units of ``reference``
    coeff: np.ndarray
    #:dict: fit parameters, so any consumer can reproduce the estimate on its own frame
    params: dict = field(default_factory=dict)
    #:str: description of the frame ``coeff`` was measured on, for logging
    reference: str = "unknown"

    def refit(self, img, traces, **kwargs):
        """Re-estimate the background on ``img`` using the same fit parameters.

        Parameters
        ----------
        img : array[nrow, ncol]
            the frame to be corrected, after the same calibration it will be
            extracted with (so that the model is in its flux units)
        traces : list[Trace]
            all traces on the detector, so order flux is masked out of the fit

        Returns
        -------
        coeff : array
            2D polynomial coefficients valid for ``img``
        """
        params = {**self.params, **kwargs}
        return estimate_background_scatter(img, traces, **params)


def as_scatter_coeff(scatter, img, traces, context=""):
    """Coefficients valid for ``img``, from whatever the scatter dependency holds.

    A :class:`ScatterModel` is refitted on ``img``; a bare coefficient array cannot be
    (its flux scale is unknown) and is used unchanged, with a warning.

    Returns
    -------
    coeff : array or None
    """
    if scatter is None:
        return None
    if isinstance(scatter, ScatterModel):
        coeff = scatter.refit(img, traces)
        logger.info(
            "Re-estimated background scatter on %s (model measured on %s)",
            context or "this frame",
            scatter.reference,
        )
        return coeff
    logger.warning(
        "Background scatter given as bare coefficients; cannot re-estimate for %s. "
        "These were fitted on another frame and are applied without rescaling, which "
        "over- or under-subtracts by the ratio of the two exposure levels.",
        context or "this frame",
    )
    return scatter


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

        # Find columns where full aperture fits within image. These need not be
        # contiguous: a trace that bows out of the frame in the middle is in
        # bounds at both edges, so index the valid columns directly rather than
        # spanning first to last.
        valid = (y_below >= 0) & (y_above < nrow)
        if not np.any(valid):
            continue

        x_valid = x_trace[valid]
        yy = y_below[valid][None, :] + np.arange(height)[:, None]
        mask[yy, np.broadcast_to(x_valid, yy.shape)] = False

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
