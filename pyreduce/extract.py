# -*- coding: utf-8 -*-
"""Module for extracting data from observations

Authors
-------

Version
-------

License
-------
"""

import logging

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.ndimage import convolve, median_filter
from scipy.ndimage.morphology import binary_hit_or_miss
from tqdm import tqdm

from .cwrappers import slitfunc, slitfunc_curved
from .util import make_index, resample

logger = logging.getLogger(__name__)


class ProgressPlot:  # pragma: no cover
    def __init__(self, nrow, ncol, nslitf, nbad=1000, title=None):
        self.nrow = nrow
        self.ncol = ncol
        self.nslitf = nslitf

        self.nbad = nbad

        plt.ion()
        self.fig = plt.figure(figsize=(12, 4))

        # self.ax1 = self.fig.add_subplot(231, projection="3d")
        self.ax1 = self.fig.add_subplot(231)
        self.ax1.set_title("Swath")
        self.ax1.set_ylabel("y [pixel]")
        self.ax2 = self.fig.add_subplot(132)
        self.ax2.set_title("Spectrum")
        self.ax2.set_xlabel("x [pixel]")
        self.ax2.set_ylabel("flux [arb. unit]")
        self.ax2.set_xlim((0, ncol))
        self.ax3 = self.fig.add_subplot(133)
        self.ax3.set_title("Slit")
        self.ax3.set_xlabel("y [pixel]")
        self.ax3.set_ylabel("contribution [1]")
        self.ax3.set_xlim((0, nrow))
        # self.ax4 = self.fig.add_subplot(234, projection="3d")
        self.ax4 = self.fig.add_subplot(234)
        self.ax4.set_title("Model")
        self.ax4.set_xlabel("x [pixel]")
        self.ax4.set_ylabel("y [pixel]")

        self.title = title
        if title is not None:
            self.fig.suptitle(title)

        self.fig.tight_layout()

        # Just plot empty pictures, to create the plots
        # Update the data later
        img = np.ones((nrow, ncol))
        # y, x = np.indices((nrow, ncol))
        # self.im_obs = self.ax1.plot_surface(x, y, img)
        # self.im_model = self.ax4.plot_surface(x, y, img)
        self.im_obs = self.ax1.imshow(img)
        self.im_model = self.ax4.imshow(img)

        (self.dots_spec,) = self.ax2.plot(
            np.zeros(nrow * ncol), np.zeros(nrow * ncol), ".r", ms=2, alpha=0.6
        )
        (self.line_spec,) = self.ax2.plot(np.zeros(ncol), "-k")
        (self.mask_spec,) = self.ax2.plot(np.zeros(self.nbad), "Pg")
        (self.dots_slit,) = self.ax3.plot(
            np.zeros(nrow * ncol), np.zeros(nrow * ncol), ".r", ms=2, alpha=0.6
        )
        (self.line_slit,) = self.ax3.plot(np.zeros(nrow), "-k", lw=2)
        (self.mask_slit,) = self.ax3.plot(np.zeros(self.nbad), "Pg")

        # self.ax1.set_zscale("log")
        # self.ax4.set_zscale("log")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def fix_linear(self, data, limit, fill=0):
        """Assures the size of the 1D array data is equal to limit"""

        if len(data) > limit:
            data = data[:limit]
        elif len(data) < limit:
            padding = np.full(limit - len(data), fill, dtype=data.dtype)
            data = np.concatenate((data, padding))
        return data

    def plot(self, img, spec, slitf, model, ycen, mask, ord_num, left, right):
        img = np.copy(img)
        spec = np.copy(spec)
        slitf = np.copy(slitf)
        ycen = np.copy(ycen)

        ny = img.shape[0]
        nspec = img.shape[1]
        x_spec, y_spec = self.get_spec(img, spec, slitf, ycen)
        x_slit, y_slit = self.get_slitf(img, spec, slitf, ycen)
        ycen = ycen + ny / 2

        old = np.linspace(-1, ny, len(slitf))

        # Fix Sizes
        mask_spec_x = self.fix_linear(x_spec[mask.ravel()], self.nbad, fill=np.nan)
        mask_spec = self.fix_linear(y_spec[mask.ravel()], self.nbad, fill=np.nan)
        mask_slit_x = self.fix_linear(x_slit[mask.ravel()], self.nbad, fill=np.nan)
        mask_slit = self.fix_linear(y_slit[mask.ravel()], self.nbad, fill=np.nan)

        ycen = self.fix_linear(ycen, self.ncol)
        x_spec = self.fix_linear(x_spec, self.ncol * self.nrow)
        y_spec = self.fix_linear(y_spec, self.ncol * self.nrow)
        spec = self.fix_linear(spec, self.ncol)
        x_slit = self.fix_linear(x_slit, self.ncol * self.nrow)
        y_slit = self.fix_linear(y_slit, self.ncol * self.nrow)
        old = self.fix_linear(old, self.nslitf)
        sf = self.fix_linear(slitf, self.nslitf)

        # Update Data
        model = np.clip(model, 0, np.max(model[5:-5, 5:-5]) * 1.1)
        self.im_obs.remove()
        img = np.clip(img, 0, np.max(model) * 1.1)
        # y, x = np.indices(img.shape)
        # self.im_obs = self.ax1.plot_surface(x, y, img)
        self.im_obs = self.ax1.imshow(img, aspect="auto", origin="lower")
        vmin, vmax = self.im_obs.norm.vmin, self.im_obs.norm.vmax
        self.im_model.remove()
        # y, x = np.indices(model.shape)
        # self.im_model = self.ax4.plot_surface(x, y, model)
        self.im_model = self.ax4.imshow(
            model, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
        )

        # self.line_ycen.set_ydata(ycen)
        self.dots_spec.set_xdata(x_spec)
        self.dots_spec.set_ydata(y_spec)
        self.line_spec.set_ydata(spec)

        self.mask_spec.set_xdata(mask_spec_x)
        self.mask_spec.set_ydata(mask_spec)

        self.dots_slit.set_xdata(x_slit)
        self.dots_slit.set_ydata(y_slit)
        self.line_slit.set_xdata(old)
        self.line_slit.set_ydata(sf)

        self.mask_slit.set_xdata(mask_slit_x)
        self.mask_slit.set_ydata(mask_slit)

        self.ax2.set_xlim((0, nspec - 1))
        limit = np.nanmax(spec[5:-5]) * 1.1
        if not np.isnan(limit):
            self.ax2.set_ylim((0, limit))

        self.ax3.set_xlim((0, ny - 1))
        limit = np.nanmax(sf) * 1.1
        if not np.isnan(limit):
            self.ax3.set_ylim((0, limit))

        title = f"Order {ord_num}, Columns {left} - {right}"
        if self.title is not None:
            title = f"{self.title}\n{title}"
        self.fig.suptitle(title)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close()

    def get_spec(self, img, spec, slitf, ycen):
        """get the spectrum corrected by the slit function"""
        nrow, ncol = img.shape
        x, y = np.indices(img.shape)
        ycen = ycen - ycen.astype(int)

        x = x - ycen + 0.5
        old = np.linspace(-1, nrow - 1 + 1, len(slitf))
        sf = np.interp(x, old, slitf)

        x = img / sf

        x = x.ravel()
        y = y.ravel()
        return y, x

    def get_slitf(self, img, spec, slitf, ycen):
        """get the slit function"""
        x = np.indices(img.shape)[0]
        ycen = ycen - ycen.astype(int)

        if np.any(spec == 0):
            i = np.arange(len(spec))
            try:
                spec = interp1d(
                    i[spec != 0], spec[spec != 0], fill_value="extrapolate"
                )(i)
            except ValueError:
                spec[spec == 0] = np.median(spec)
        y = img / spec[None, :]
        y = y.ravel()

        x = x - ycen + 0.5
        x = x.ravel()
        return x, y


class Swath:
    def __init__(self, nswath):
        self.nswath = nswath
        self.spec = [None] * nswath
        self.slitf = [None] * nswath
        self.model = [None] * nswath
        self.unc = [None] * nswath
        self.mask = [None] * nswath
        self.info = [None] * nswath

    def __len__(self):
        return self.nswath

    def __getitem__(self, key):
        return (
            self.spec[key],
            self.slitf[key],
            self.model[key],
            self.unc[key],
            self.mask[key],
            self.info[key],
        )

    def __setitem__(self, key, value):
        self.spec[key] = value[0]
        self.slitf[key] = value[1]
        self.model[key] = value[2]
        self.unc[key] = value[3]
        self.mask[key] = value[4]
        self.info[key] = value[5]


def fix_parameters(xwd, cr, orders, nrow, ncol, nord, ignore_column_range=False):
    """Fix extraction width and column range, so that all pixels used are within the image.
    I.e. the column range is cut so that the everything is within the image

    Parameters
    ----------
    xwd : float, array
        Extraction width, either one value for all orders, or the whole array
    cr : 2-tuple(int), array
        Column range, either one value for all orders, or the whole array
    orders : array
        polynomial coefficients that describe each order
    nrow : int
        Number of rows in the image
    ncol : int
        Number of columns in the image
    nord : int
        Number of orders in the image
    ignore_column_range : bool, optional
        if true does not change the column range, however this may lead to problems with the extraction, by default False

    Returns
    -------
    xwd : array
        fixed extraction width
    cr : array
        fixed column range
    orders : array
        the same orders as before
    """

    if xwd is None:
        xwd = 0.5
    if np.isscalar(xwd):
        xwd = np.tile([xwd, xwd], (nord, 1))
    else:
        xwd = np.asarray(xwd)
        if xwd.ndim == 1:
            xwd = np.tile(xwd, (nord, 1))

    if cr is None:
        cr = np.tile([0, ncol], (nord, 1))
    else:
        cr = np.asarray(cr)
        if cr.ndim == 1:
            cr = np.tile(cr, (nord, 1))

    orders = np.asarray(orders)

    xwd = np.array([xwd[0], *xwd, xwd[-1]])
    cr = np.array([cr[0], *cr, cr[-1]])
    orders = extend_orders(orders, nrow)

    xwd = fix_extraction_width(xwd, orders, cr, ncol)
    if not ignore_column_range:
        cr = fix_column_range(cr, orders, xwd, nrow, ncol)

    orders = orders[1:-1]
    xwd = xwd[1:-1]
    cr = cr[1:-1]

    return xwd, cr, orders


def extend_orders(orders, nrow):
    """Extrapolate extra orders above and below the existing ones

    Parameters
    ----------
    orders : array[nord, degree]
        order tracing coefficients
    nrow : int
        number of rows in the image

    Returns
    -------
    orders : array[nord + 2, degree]
        extended orders
    """

    nord, ncoef = orders.shape

    if nord > 1:
        order_low = 2 * orders[0] - orders[1]
        order_high = 2 * orders[-1] - orders[-2]
    else:
        order_low = [0 for _ in range(ncoef)]
        order_high = [0 for _ in range(ncoef - 1)] + [nrow]

    return np.array([order_low, *orders, order_high])


def fix_extraction_width(xwd, orders, cr, ncol):
    """Convert fractional extraction width to pixel range

    Parameters
    ----------
    extraction_width : array[nord, 2]
        current extraction width, in pixels or fractions (for values below 1.5)
    orders : array[nord, degree]
        order tracing coefficients
    column_range : array[nord, 2]
        column range to use
    ncol : int
        number of columns in image

    Returns
    -------
    extraction_width : array[nord, 2]
        updated extraction width in pixels
    """

    if not np.all(xwd > 1.5):
        # if extraction width is in relative scale transform to pixel scale
        x = np.arange(ncol)
        for i in range(1, len(xwd) - 1):
            for j in [0, 1]:
                if xwd[i, j] < 1.5:
                    k = i - 1 if j == 0 else i + 1
                    left = max(cr[[i, k], 0])
                    right = min(cr[[i, k], 1])

                    if right < left:
                        raise ValueError(
                            f"Check your column ranges. Orders {i} and {k} are weird"
                        )

                    current = np.polyval(orders[i], x[left:right])
                    below = np.polyval(orders[k], x[left:right])
                    xwd[i, j] *= np.min(np.abs(current - below))

        xwd[0] = xwd[1]
        xwd[-1] = xwd[-2]

    xwd = np.ceil(xwd).astype(int)

    return xwd


def fix_column_range(column_range, orders, extraction_width, nrow, ncol):
    """Fix the column range, so that no pixels outside the image will be accessed (Thus avoiding errors)

    Parameters
    ----------
    img : array[nrow, ncol]
        image
    orders : array[nord, degree]
        order tracing coefficients
    extraction_width : array[nord, 2]
        extraction width in pixels, (below, above)
    column_range : array[nord, 2]
        current column range
    no_clip : bool, optional
        if False, new column range will be smaller or equal to current column range, otherwise it can also be larger (default: False)

    Returns
    -------
    column_range : array[nord, 2]
        updated column range
    """

    ix = np.arange(ncol)
    # Loop over non extension orders
    for i, order in zip(range(1, len(orders) - 1), orders[1:-1]):
        # Shift order trace up/down by extraction_width
        coeff_bot, coeff_top = np.copy(order), np.copy(order)
        coeff_bot[-1] -= extraction_width[i, 0]
        coeff_top[-1] += extraction_width[i, 1]

        y_bot = np.polyval(coeff_bot, ix)  # low edge of arc
        y_top = np.polyval(coeff_top, ix)  # high edge of arc

        # find regions of pixels inside the image
        # then use the region that most closely resembles the existing column range (from order tracing)
        # but clip it to the existing column range (order tracing polynomials are not well defined outside the original range)
        points_in_image = np.where((y_bot >= 0) & (y_top < nrow))[0]

        if len(points_in_image) == 0:
            raise ValueError(
                f"No pixels are completely within the extraction width for order {i}"
            )

        regions = np.where(np.diff(points_in_image) != 1)[0]
        regions = [(r, r + 1) for r in regions]
        regions = [
            points_in_image[0],
            *points_in_image[(regions,)].ravel(),
            points_in_image[-1],
        ]
        regions = [[regions[i], regions[i + 1] + 1] for i in range(0, len(regions), 2)]
        overlap = [
            min(reg[1], column_range[i, 1]) - max(reg[0], column_range[i, 0])
            for reg in regions
        ]
        iregion = np.argmax(overlap)
        column_range[i] = np.clip(
            regions[iregion], column_range[i, 0], column_range[i, 1]
        )

    column_range[0] = column_range[1]
    column_range[-1] = column_range[-2]

    return column_range


def make_bins(swath_width, xlow, xhigh, ycen):
    """Create bins for the swathes
    Bins are roughly equally sized, have roughly length swath width (if given)
    and overlap roughly half-half with each other

    Parameters
    ----------
    swath_width : {int, None}
        initial value for the swath_width, bins will have roughly that size, but exact value may change
        if swath_width is None, determine a good value, from the data
    xlow : int
        lower bound for x values
    xhigh : int
        upper bound for x values
    ycen : array[ncol]
        center of the order trace

    Returns
    -------
    nbin : int
        number of bins
    bins_start : array[nbin]
        left(beginning) side of the bins
    bins_end : array[nbin]
        right(ending) side of the bins
    """

    if swath_width is None:
        ncol = len(ycen)
        i = np.unique(ycen.astype(int))  # Points of row crossing
        # ni = len(i)  # This is how many times this order crosses to the next row
        if len(i) > 1:  # Curved order crosses rows
            i = np.sum(i[1:] - i[:-1]) / (len(i) - 1)
            nbin = np.clip(
                int(np.round(ncol / i)) // 3, 3, 20
            )  # number of swaths along the order
        else:  # Perfectly aligned orders
            nbin = np.clip(ncol // 400, 3, None)  # Still follow the changes in PSF
        nbin = nbin * (xhigh - xlow) // ncol  # Adjust for the true order length
    else:
        nbin = np.clip(int(np.round((xhigh - xlow) / swath_width)), 1, None)

    bins = np.linspace(xlow, xhigh, 2 * nbin + 1)  # boundaries of bins
    bins_start = np.ceil(bins[:-2]).astype(int)  # beginning of each bin
    bins_end = np.floor(bins[2:]).astype(int)  # end of each bin

    return nbin, bins_start, bins_end


def calc_telluric_correction(telluric, img):  # pragma: no cover
    """Calculate telluric correction

    If set to specific integer larger than 1 is used as the
    offset from the order center line. The sky is then estimated by computing
    median signal between this offset and the upper/lower limit of the
    extraction window.

    Parameters
    ----------
    telluric : int
        telluric correction parameter
    img : array
        image of the swath

    Returns
    -------
    tell : array
        telluric correction
    """
    width, height = img.shape

    tel_lim = telluric if telluric > 5 and telluric < height / 2 else min(5, height / 3)
    tel = np.sum(img, axis=0)
    itel = np.arange(height)
    itel = itel[np.abs(itel - height / 2) >= tel_lim]
    tel = img[itel, :]
    sc = np.zeros(width)

    for itel in range(width):
        sc[itel] = np.ma.median(tel[itel])

    return sc


def calc_scatter_correction(scatter, index):
    """Calculate scatter correction
    by interpolating between values?

    Parameters
    ----------
    scatter : array of shape (degree_x, degree_y)
        2D polynomial coefficients of the background scatter
    index : tuple (array, array)
        indices of the swath within the overall image

    Returns
    -------
    scatter_correction : array of shape (swath_width, swath_height)
        correction for scattered light
    """

    # The indices in the image are switched
    y, x = index
    scatter_correction = np.polynomial.polynomial.polyval2d(x, y, scatter)
    return scatter_correction


def extract_spectrum(
    img,
    ycen,
    yrange,
    xrange,
    gain=1,
    readnoise=0,
    lambda_sf=0.1,
    lambda_sp=0,
    osample=1,
    swath_width=None,
    maxiter=20,
    telluric=None,
    scatter=None,
    normalize=False,
    threshold=0,
    tilt=None,
    shear=None,
    plot=False,
    plot_title=None,
    im_norm=None,
    im_ordr=None,
    out_spec=None,
    out_sunc=None,
    out_slitf=None,
    out_mask=None,
    progress=None,
    ord_num=0,
    **kwargs,
):
    """
    Extract the spectrum of a single order from an image
    The order is split into several swathes of roughly swath_width length, which overlap half-half
    For each swath a spectrum and slitfunction are extracted
    overlapping sections are combined using linear weights (centrum is strongest, falling off to the edges)
    Here is the layout for the bins:

    ::

           1st swath    3rd swath    5th swath      ...
        /============|============|============|============|============|

                  2nd swath    4th swath    6th swath
               |------------|------------|------------|------------|
               |.....|
               overlap

               +     ******* 1
                +   *
                 + *
                  *            weights (+) previous swath, (*) current swath
                 * +
                *   +
               *     +++++++ 0

    Parameters
    ----------
    img : array[nrow, ncol]
        observation (or similar)
    ycen : array[ncol]
        order trace of the current order
    yrange : tuple(int, int)
        extraction width in pixles, below and above
    xrange : tuple(int, int)
        columns range to extract (low, high)
    gain : float, optional
        adu to electron, amplifier gain (default: 1)
    readnoise : float, optional
        read out noise factor (default: 0)
    lambda_sf : float, optional
        slit function smoothing parameter, usually very small (default: 0.1)
    lambda_sp : int, optional
        spectrum smoothing parameter, usually very small (default: 0)
    osample : int, optional
        oversampling factor, i.e. how many subpixels to create per pixel (default: 1, i.e. no oversampling)
    swath_width : int, optional
        swath width suggestion, actual width depends also on ncol, see make_bins (default: None, which will determine the width based on the order tracing)
    telluric : {float, None}, optional
        telluric correction factor (default: None, i.e. no telluric correction)
    scatter : {array, None}, optional
        background scatter as 2d polynomial coefficients (default: None, no correction)
    normalize : bool, optional
        whether to create a normalized image. If true, im_norm and im_ordr are used as output (default: False)
    threshold : int, optional
        threshold for normalization (default: 0)
    tilt : array[ncol], optional
        The tilt (1st order curvature) of the slit in this order for the curved extraction (default: None, i.e. tilt = 0)
    shear : array[ncol], optional
        The shear (2nd order curvature) of the slit in this order for the curved extraction (default: None, i.e. shear = 0)
    plot : bool, optional
        wether to plot the progress, plotting will slow down the procedure significantly (default: False)
    ord_num : int, optional
        current order number, just for plotting (default: 0)
    im_norm : array[nrow, ncol], optional
        normalized image, only output if normalize is True (default: None)
    im_ordr : array[nrow, ncol], optional
        image of the order blaze, only output if normalize is True (default: None)

    Returns
    -------
    spec : array[ncol]
        extracted spectrum
    slitf : array[nslitf]
        extracted slitfunction
    mask : array[ncol]
        mask of the column range to use in the spectrum
    unc : array[ncol]
        uncertainty on the spectrum
    """

    _, ncol = img.shape
    ylow, yhigh = yrange
    xlow, xhigh = xrange
    nslitf = osample * (ylow + yhigh + 2) + 1
    height = yhigh + ylow + 1

    ycen_int = np.floor(ycen).astype(int)

    spec = np.zeros(ncol) if out_spec is None else out_spec
    sunc = np.zeros(ncol) if out_sunc is None else out_sunc
    mask = np.full(ncol, False) if out_mask is None else out_mask
    slitf = np.zeros(nslitf) if out_slitf is None else out_slitf

    nbin, bins_start, bins_end = make_bins(swath_width, xlow, xhigh, ycen)
    nswath = 2 * nbin - 1
    swath = Swath(nswath)
    margin = np.zeros((nswath, 2), int)

    if normalize:
        norm_img = [None] * nswath
        norm_model = [None] * nswath

    # Perform slit decomposition within each swath stepping through the order with
    # half swath width. Spectra for each decomposition are combined with linear weights.
    with tqdm(
        enumerate(zip(bins_start, bins_end)),
        total=len(bins_start),
        leave=False,
        desc="Swath",
    ) as t:
        for ihalf, (ibeg, iend) in t:
            logger.debug("Extracting Swath %i, Columns: %i - %i", ihalf, ibeg, iend)

            # Cut out swath from image
            index = make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
            swath_img = img[index]
            swath_ycen = ycen[ibeg:iend]

            # Corrections
            # TODO: what is it even supposed to do?
            if telluric is not None:  # pragma: no cover
                telluric_correction = calc_telluric_correction(telluric, swath_img)
            else:
                telluric_correction = 0

            if scatter is not None:
                scatter_correction = calc_scatter_correction(scatter, index)
            else:
                scatter_correction = 0

            swath_img -= scatter_correction + telluric_correction

            # Do Slitfunction extraction
            swath_tilt = tilt[ibeg:iend] if tilt is not None else 0
            swath_shear = shear[ibeg:iend] if shear is not None else 0
            swath[ihalf] = slitfunc_curved(
                swath_img,
                swath_ycen,
                swath_tilt,
                swath_shear,
                lambda_sp=lambda_sp,
                lambda_sf=lambda_sf,
                osample=osample,
                yrange=yrange,
                maxiter=maxiter,
                gain=gain,
            )
            t.set_postfix(chi=f"{swath[ihalf][5][1]:1.2f}")

            if normalize:
                # Save image and model for later
                # Use np.divide to avoid divisions by zero
                where = swath.model[ihalf] > threshold / gain
                norm_img[ihalf] = np.ones_like(swath.model[ihalf])
                np.divide(
                    np.abs(swath_img),
                    swath.model[ihalf],
                    where=where,
                    out=norm_img[ihalf],
                )
                norm_model[ihalf] = swath.model[ihalf]

            if plot >= 2 and not np.all(np.isnan(swath_img)):  # pragma: no cover
                if progress is None:
                    progress = ProgressPlot(
                        swath_img.shape[0], swath_img.shape[1], nslitf, title=plot_title
                    )
                progress.plot(
                    swath_img,
                    swath.spec[ihalf],
                    swath.slitf[ihalf],
                    swath.model[ihalf],
                    swath_ycen,
                    swath.mask[ihalf],
                    ord_num,
                    ibeg,
                    iend,
                )

    # Remove points at the border of the each swath, if order has tilt
    # as those pixels have bad information
    for i in range(nswath):
        margin[i, :] = int(swath.info[i][4]) + 1

    # Weight for combining swaths
    weight = [np.ones(bins_end[i] - bins_start[i]) for i in range(nswath)]
    weight[0][: margin[0, 0]] = 0
    weight[-1][len(weight[-1]) - margin[-1, 1] :] = 0
    for i, j in zip(range(0, nswath - 1), range(1, nswath)):
        width = bins_end[i] - bins_start[i]
        overlap = bins_end[i] - bins_start[j]

        # Start and end indices for the two swaths
        start_i = width - overlap + margin[j, 0]
        end_i = width - margin[i, 1]

        start_j = margin[j, 0]
        end_j = overlap - margin[i, 1]

        # Weights for one overlap from 0 to 1, but do not include those values (whats the point?)
        triangle = np.linspace(0, 1, overlap + 1, endpoint=False)[1:]
        # Cut away the margins at the corners
        triangle = triangle[margin[j, 0] : len(triangle) - margin[i, 1]]

        # Set values
        weight[i][start_i:end_i] = 1 - triangle
        weight[j][start_j:end_j] = triangle

        # Don't use the pixels at the egdes (due to curvature)
        weight[i][end_i:] = 0
        weight[j][:start_j] = 0

    # Update column range
    xrange[0] += margin[0, 0]
    xrange[1] -= margin[-1, 1]
    mask[: xrange[0]] = True
    mask[xrange[1] :] = True

    # Apply weights
    for i, (ibeg, iend) in enumerate(zip(bins_start, bins_end)):
        spec[ibeg:iend] += swath.spec[i] * weight[i]
        sunc[ibeg:iend] += swath.unc[i] * weight[i]

    if normalize:
        for i, (ibeg, iend) in enumerate(zip(bins_start, bins_end)):
            index = make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
            im_norm[index] += norm_img[i] * weight[i]
            im_ordr[index] += norm_model[i] * weight[i]

    slitf[:] = np.mean(swath.slitf, axis=0)
    sunc[:] = np.sqrt(sunc ** 2 + (readnoise / gain) ** 2)
    return spec, slitf, mask, sunc


def model(spec, slitf):
    return spec[None, :] * slitf[:, None]


def get_y_scale(ycen, xrange, extraction_width, nrow):
    """Calculate the y limits of the order
    This is especially important at the edges

    Parameters
    ----------
    ycen : array[ncol]
        order trace
    xrange : tuple(int, int)
        column range
    extraction_width : tuple(int, int)
        extraction width in pixels below and above the order
    nrow : int
        number of rows in the image, defines upper edge

    Returns
    -------
    y_low, y_high : int, int
        lower and upper y bound for extraction
    """
    ycen = ycen[xrange[0] : xrange[1]]

    ymin = ycen - extraction_width[0]
    ymin = np.floor(ymin)
    if min(ymin) < 0:
        ymin = ymin - min(ymin)  # help for orders at edge
    if max(ymin) >= nrow:
        ymin = ymin - max(ymin) + nrow - 1  # helps at edge

    ymax = ycen + extraction_width[1]
    ymax = np.ceil(ymax)
    if max(ymax) >= nrow:
        ymax = ymax - max(ymax) + nrow - 1  # helps at edge

    # Define a fixed height area containing one spectral order
    y_lower_lim = int(np.min(ycen - ymin))  # Pixels below center line
    y_upper_lim = int(np.min(ymax - ycen))  # Pixels above center line

    return y_lower_lim, y_upper_lim


def optimal_extraction(
    img,
    orders,
    extraction_width,
    column_range,
    tilt,
    shear,
    plot=False,
    plot_title=None,
    **kwargs,
):
    """Use optimal extraction to get spectra

    This functions just loops over the orders, the actual work is done in extract_spectrum

    Parameters
    ----------
    img : array[nrow, ncol]
        image to extract
    orders : array[nord, degree]
        order tracing coefficients
    extraction_width : array[nord, 2]
        extraction width in pixels
    column_range : array[nord, 2]
        column range to use
    scatter : array[nord, 4, ncol]
        background scatter (or None)
    **kwargs
        other parameters for the extraction (see extract_spectrum)

    Returns
    -------
    spectrum : array[nord, ncol]
        extracted spectrum
    slitfunction : array[nord, nslitf]
        recovered slitfunction
    uncertainties: array[nord, ncol]
        uncertainties on the spectrum
    """

    logger.info("Using optimal extraction to produce spectrum")

    nrow, ncol = img.shape
    nord = len(orders)

    spectrum = np.zeros((nord, ncol))
    uncertainties = np.zeros((nord, ncol))
    slitfunction = [None for _ in range(nord)]

    if tilt is None:
        tilt = [None for _ in range(nord)]
    if shear is None:
        shear = [None for _ in range(nord)]

    # Add mask as defined by column ranges
    mask = np.full((nord, ncol), True)
    for i in range(nord):
        mask[i, column_range[i, 0] : column_range[i, 1]] = False
    spectrum = np.ma.array(spectrum, mask=mask)
    uncertainties = np.ma.array(uncertainties, mask=mask)

    ix = np.arange(ncol)
    if plot >= 2:  # pragma: no cover
        ncol_swath = kwargs.get("swath_width", img.shape[1] // 400)
        nrow_swath = np.sum(extraction_width, axis=1).max()
        nslitf_swath = (nrow_swath + 2) * kwargs.get("osample", 1) + 1
        progress = ProgressPlot(nrow_swath, ncol_swath, nslitf_swath, title=plot_title)
    else:
        progress = None

    for i in tqdm(range(nord), desc="Order"):
        logger.debug("Extracting relative order %i out of %i", i + 1, nord)

        # Define a fixed height area containing one spectral order
        ycen = np.polyval(orders[i], ix)
        yrange = get_y_scale(ycen, column_range[i], extraction_width[i], nrow)

        osample = kwargs.get("osample", 1)
        slitfunction[i] = np.zeros(osample * (sum(yrange) + 2) + 1)

        # Return values are set by reference, as the out parameters
        # Also column_range is adjusted depending on the shear
        # This is to avoid large chunks of memory of essentially duplicates
        extract_spectrum(
            img,
            ycen,
            yrange,
            column_range[i],
            tilt=tilt[i],
            shear=shear[i],
            out_spec=spectrum[i],
            out_sunc=uncertainties[i],
            out_slitf=slitfunction[i],
            out_mask=mask[i],
            progress=progress,
            ord_num=i + 1,
            plot=plot,
            plot_title=plot_title,
            **kwargs,
        )

    if plot >= 2:  # pragma: no cover
        progress.close()

    if plot:  # pragma: no cover
        plot_comparison(
            img,
            orders,
            spectrum,
            slitfunction,
            extraction_width,
            column_range,
            title=plot_title,
        )

    return spectrum, slitfunction, uncertainties


def correct_for_curvature(img_order, tilt, shear, xwd):
    # img_order = np.ma.filled(img_order, np.nan)
    mask = ~np.ma.getmaskarray(img_order)

    xt = np.arange(img_order.shape[1])
    for y, yt in zip(range(xwd[0] + xwd[1]), range(-xwd[0], xwd[1])):
        xi = xt + yt * tilt + yt ** 2 * shear
        img_order[y] = np.interp(
            xi, xt[mask[y]], img_order[y][mask[y]], left=0, right=0
        )

    xt = np.arange(img_order.shape[0])
    for x in range(img_order.shape[1]):
        img_order[:, x] = np.interp(
            xt, xt[mask[:, x]], img_order[:, x][mask[:, x]], left=0, right=0
        )

    return img_order


def model_image(img, xwd, tilt, shear):
    # Correct image for curvature
    height = img.shape[0]
    img = correct_for_curvature(img, tilt, shear, xwd)
    # Find slitfunction using the median to avoid outliers
    slitf = np.ma.median(img, axis=1)
    slitf /= np.ma.sum(slitf)
    # Use the slitfunction to find spectrum
    spec = np.ma.median(img / slitf[:, None], axis=0)
    # Create model from slitfunction and spectrum
    model = spec[None, :] * slitf[:, None]
    # Reapply curvature to the model
    model = correct_for_curvature(model, -tilt, -shear, xwd)
    return model, spec, slitf


def get_mask(img, model):
    # 99.73 = 3 sigma, 2 * 3 = 6 sigma
    residual = np.ma.abs(img - model)
    median, vmax = np.percentile(np.ma.compressed(residual), (50, 99.73))
    vmax = median + 2 * (vmax - median)
    return residual > vmax


def arc_extraction(
    img,
    orders,
    extraction_width,
    column_range,
    gain=1,
    readnoise=0,
    dark=0,
    plot=False,
    plot_title=None,
    tilt=None,
    shear=None,
    collapse_function="median",
    **kwargs,
):
    """Use "simple" arc extraction to get a spectrum
    Arc extraction simply takes the sum orthogonal to the order for extraction width pixels

    This extraction makes a few rough assumptions and does not provide the most accurate results,
    but rather a good approximation

    Parameters
    ----------
    img : array[nrow, ncol]
        image to extract
    orders : array[nord, order]
        order tracing coefficients
    extraction_width : array[nord, 2]
        extraction width in pixels
    column_range : array[nord, 2]
        column range to use
    gain : float, optional
        adu to electron, amplifier gain (default: 1)
    readnoise : float, optional
        read out noise (default: 0)
    dark : float, optional
        dark current noise (default: 0)
    plot : bool, optional
        wether to plot the results (default: False)

    Returns
    -------
    spectrum : array[nord, ncol]
        extracted spectrum
    uncertainties : array[nord, ncol]
        uncertainties on extracted spectrum
    """

    logger.info("Using arc extraction to produce spectrum")
    _, ncol = img.shape
    nord, _ = orders.shape

    spectrum = np.zeros((nord, ncol))
    uncertainties = np.zeros((nord, ncol))

    # Add mask as defined by column ranges
    mask = np.full((nord, ncol), True)
    for i in range(nord):
        mask[i, column_range[i, 0] : column_range[i, 1]] = False
    spectrum = np.ma.array(spectrum, mask=mask)
    uncertainties = np.ma.array(uncertainties, mask=mask)

    x = np.arange(ncol)

    for i in tqdm(range(nord), desc="Order"):
        logger.debug("Calculating order %i out of %i", i + 1, nord)

        x_left_lim = column_range[i, 0]
        x_right_lim = column_range[i, 1]

        # Rectify the image, i.e. remove the shape of the order
        # Then the center of the order is within one pixel variations
        ycen = np.polyval(orders[i], x).astype(int)
        yb, yt = ycen - extraction_width[i, 0], ycen + extraction_width[i, 1]
        height = extraction_width[i, 0] + extraction_width[i, 1] + 1
        index = make_index(yb, yt, x_left_lim, x_right_lim)
        img_order = img[index]

        # Correct for tilt and shear
        # For each row of the rectified order, interpolate onto the shifted row
        # Masked pixels are set to 0, similar to the summation
        if tilt is not None and shear is not None:
            img_order = correct_for_curvature(
                img_order,
                tilt[i, x_left_lim:x_right_lim],
                shear[i, x_left_lim:x_right_lim],
                extraction_width[i],
            )

        # Sum over the prepared image
        if collapse_function == "sum":
            arc = np.ma.sum(img_order, axis=0)
        elif collapse_function == "mean":
            arc = np.ma.mean(img_order, axis=0) * img_order.shape[0]
        elif collapse_function == "median":
            arc = np.ma.median(img_order, axis=0) * img_order.shape[0]
        else:
            raise ValueError(
                "Could not determine the arc method, expected one of ('sum', 'mean', 'median'), but got %s"
                % collapse_function
            )

        # Store results
        spectrum[i, x_left_lim:x_right_lim] = arc
        uncertainties[i, x_left_lim:x_right_lim] = (
            np.sqrt(np.abs(arc * gain + dark + readnoise ** 2)) / gain
        )

    if plot:  # pragma: no cover
        plot_comparison(
            img,
            orders,
            spectrum,
            None,
            extraction_width,
            column_range,
            title=plot_title,
        )

    return spectrum, uncertainties


def plot_comparison(
    original, orders, spectrum, slitf, extraction_width, column_range, title=None
):  # pragma: no cover
    nrow, ncol = original.shape
    nord = len(orders)
    output = np.zeros((np.sum(extraction_width) + nord, ncol))
    pos = [0]
    x = np.arange(ncol)
    for i in range(nord):
        ycen = np.polyval(orders[i], x)
        yb = ycen - extraction_width[i, 0]
        yt = ycen + extraction_width[i, 1]
        xl, xr = column_range[i]
        index = make_index(yb, yt, xl, xr)
        yl = pos[i]
        yr = pos[i] + index[0].shape[0]
        output[yl:yr, xl:xr] = original[index]

        vmin, vmax = np.percentile(output[yl:yr, xl:xr], (5, 95))
        output[yl:yr, xl:xr] = np.clip(output[yl:yr, xl:xr], vmin, vmax)
        output[yl:yr, xl:xr] -= vmin
        output[yl:yr, xl:xr] /= vmax - vmin

        pos += [yr]

    plt.imshow(output, origin="lower", aspect="auto")

    for i in range(nord):
        try:
            tmp = spectrum[i, column_range[i, 0] : column_range[i, 1]]
            # if len(tmp)
            vmin = np.min(tmp[tmp != 0])
            tmp = np.copy(spectrum[i])
            tmp[tmp != 0] -= vmin
            np.log(tmp, out=tmp, where=tmp > 0)
            tmp = tmp / np.max(tmp) * 0.9 * (pos[i + 1] - pos[i])
            tmp += pos[i]
            tmp[tmp < pos[i]] = pos[i]
            plt.plot(x, tmp, "r")
        except:
            pass

    locs = np.sum(extraction_width, axis=1) + 1
    locs = np.array([0, *np.cumsum(locs)[:-1]])
    locs[:-1] += (np.diff(locs) * 0.5).astype(int)
    locs[-1] += ((output.shape[0] - locs[-1]) * 0.5).astype(int)
    plt.yticks(locs, range(len(locs)))

    plot_title = "Extracted Spectrum vs. Rectified Image"
    if title is not None:
        plot_title = f"{title}\n{plot_title}"
    plt.title(plot_title)
    plt.xlabel("x [pixel]")
    plt.ylabel("order")
    plt.show()


def extract(
    img,
    orders,
    column_range=None,
    order_range=None,
    extraction_width=0.5,
    extraction_type="optimal",
    tilt=None,
    shear=None,
    sigma_cutoff=0,
    **kwargs,
):
    """
    Extract the spectrum from an image

    Parameters
    ----------
    img : array[nrow, ncol](float)
        observation to extract
    orders : array[nord, degree](float)
        polynomial coefficients of the order tracing
    column_range : array[nord, 2](int), optional
        range of pixels to use for each order (default: use all)
    order_range : array[2](int), optional
        range of orders to extract, orders have to be consecutive (default: use all)
    extraction_width : array[nord, 2]({float, int}), optional
        extraction width above and below each order, values below 1.5 are considered relative, while values above are absolute (default: 0.5)
    extraction_type : {"optimal", "arc", "normalize"}, optional
        which extracttion algorithm to use, "optimal" uses optimal extraction, "arc" uses simple arc extraction, and "normalize" also uses optimal extraction, but returns the normalized image (default: "optimal")
    tilt : float or array[nord, ncol], optional
        The tilt (1st order curvature) of the slit for curved extraction. Will use vertical extraction if no tilt is set. (default: None, i.e. tilt = 0)
    shear : float or array[nord, ncol], optional
        The shear (2nd order curvature) of the slit for curved extraction (default: None, i.e. shear = 0)
    polarization : bool, optional
        if true, pairs of orders are considered to belong to the same order, but different polarization. Only affects the scatter (default: False)
    **kwargs, optional
        parameters for extraction functions

    Returns
    -------
    spec : array[nord, ncol](float)
        extracted spectrum for each order
    uncertainties : array[nord, ncol](float)
        uncertainties on the spectrum

    if extraction_type == "normalize" instead return

    im_norm : array[nrow, ncol](float)
        normalized image
    im_ordr : array[nrow, ncol](float)
        image with just the orders
    blaze : array[nord, ncol](float)
        extracted spectrum (equals blaze if img was the flat field)
    """

    nrow, ncol = img.shape
    nord, _ = orders.shape
    if order_range is None:
        order_range = (0, nord)
    if np.isscalar(tilt):
        n = order_range[1] - order_range[0]
        tilt = np.full((n, ncol), tilt)
    if np.isscalar(shear):
        n = order_range[1] - order_range[0]
        shear = np.full((n, ncol), shear)

    # Fix the input parameters
    extraction_width, column_range, orders = fix_parameters(
        extraction_width, column_range, orders, nrow, ncol, nord
    )
    # Limit orders (and related properties) to orders in range
    nord = order_range[1] - order_range[0]
    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]
    extraction_width = extraction_width[order_range[0] : order_range[1]]

    # if sigma_cutoff > 0:
    #     # Blur the image and mask outliers
    #     img = np.ma.masked_invalid(img, copy=False)
    #     img.data[img.mask] = 0
    #     # Use the median of the sorounding pixels (excluding the pixel itself)
    #     footprint = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    #     dilated = median_filter(img, footprint=footprint)
    #     diff = np.ma.abs(img - dilated)
    #     # median = 50%; 3 sigma = 99.73 %
    #     median, std = np.percentile(diff.compressed(), (50, 99.73))
    #     mask = diff > median + sigma_cutoff * std / 3
    #     img[mask] = np.ma.masked

    if extraction_type == "optimal":
        # the "normal" case, except for wavelength calibration files
        spectrum, slitfunction, uncertainties = optimal_extraction(
            img,
            orders,
            extraction_width,
            column_range,
            tilt=tilt,
            shear=shear,
            **kwargs,
        )
    elif extraction_type == "normalize":
        # TODO
        # Prepare normalized flat field image if necessary
        # These will be passed and "returned" by reference
        # I dont like it, but it works for now
        im_norm = np.zeros_like(img)
        im_ordr = np.zeros_like(img)

        blaze, _, _ = optimal_extraction(
            img,
            orders,
            extraction_width,
            column_range,
            tilt=tilt,
            shear=shear,
            normalize=True,
            im_norm=im_norm,
            im_ordr=im_ordr,
            **kwargs,
        )
        threshold_lower = kwargs.get("threshold_lower", 0)
        im_norm[im_norm <= threshold_lower] = 1
        im_ordr[im_ordr <= threshold_lower] = 1
        return im_norm, im_ordr, blaze, column_range
    elif extraction_type == "arc":
        # Simpler extraction, just summing along the arc of the order
        spectrum, uncertainties = arc_extraction(
            img,
            orders,
            extraction_width,
            column_range,
            tilt=tilt,
            shear=shear,
            **kwargs,
        )
        slitfunction = None
    else:
        raise ValueError(
            f"Parameter 'extraction_type' not understood. Expected 'optimal', 'normalize', or 'arc' bug got {extraction_type}."
        )

    return spectrum, uncertainties, slitfunction, column_range
