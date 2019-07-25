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
from scipy.interpolate import interp1d
from PIL import Image

from .cwrappers import slitfunc, slitfunc_curved
from .util import make_index

# TODO put the plotting somewhere else
# np.seterr(all="raise")

def imresize(img, newsize):
    return np.array(Image.fromarray(img).resize(newsize))


class ProgressPlot:
    def __init__(self, nrow, ncol, nbad=1000):
        self.nrow = nrow
        self.ncol = ncol

        self.nbad = nbad

        plt.ion()
        self.fig = plt.figure(figsize=(12, 4))
        self.fig.tight_layout(pad=0.05)

        self.ax1 = self.fig.add_subplot(231)
        self.ax1.set_title("Swath")
        self.ax2 = self.fig.add_subplot(132)
        self.ax2.set_title("Spectrum")
        self.ax2.set_xlim((0, ncol))
        self.ax3 = self.fig.add_subplot(133)
        self.ax3.set_title("Slit")
        self.ax3.set_xlim((0, nrow))
        self.ax4 = self.fig.add_subplot(234)
        self.ax4.set_title("Model")

        # Just plot empty pictures, to create the plots
        # Update the data later
        self.im_obs = self.ax1.imshow(
            np.zeros((nrow, ncol)), aspect="auto", origin="lower"
        )
        self.line_ycen, = self.ax1.plot(np.zeros(ncol), "-r")
        self.im_model = self.ax4.imshow(
            np.zeros((nrow, ncol)), aspect="auto", origin="lower"
        )
        self.dots_spec, = self.ax2.plot(
            np.zeros(nrow * ncol), np.zeros(nrow * ncol), ".r", ms=2, alpha=0.6
        )
        self.line_spec, = self.ax2.plot(np.zeros(ncol), "-k")
        self.mask_spec, = self.ax2.plot(np.zeros(self.nbad), "+g")
        self.dots_slit, = self.ax3.plot(
            np.zeros(nrow * ncol), np.zeros(nrow * ncol), ".r", ms=2, alpha=0.6
        )
        self.line_slit, = self.ax3.plot(np.zeros(nrow), "-k", lw=3)
        self.mask_slit, = self.ax3.plot(np.zeros(self.nbad), "+g")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def fix_image(self, img):
        """ Assures that the shape of img is equal to self.nrow, self.ncol """
        img = imresize(img, (self.nrow, self.ncol))
        return img

    def fix_linear(self, data, limit, fill=0):
        """ Assures the size of the 1D array data is equal to limit """

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
        x_spec, y_spec = self.get_spec(img, spec, slitf, ycen)
        x_slit, y_slit = self.get_slitf(img, spec, slitf)
        ycen = ycen + ny / 2

        new = np.linspace(0, ny - 1, ny)
        old = np.linspace(-1, ny, len(slitf))
        sf = np.interp(new, old, slitf)

        # Fix Sizes
        mask_spec_x = self.fix_linear(x_spec[mask.ravel()], self.nbad, fill=np.nan)
        mask_spec = self.fix_linear(y_spec[mask.ravel()], self.nbad, fill=np.nan)
        mask_slit_x = self.fix_linear(x_slit[mask.ravel()], self.nbad, fill=np.nan)
        mask_slit = self.fix_linear(y_slit[mask.ravel()], self.nbad, fill=np.nan)

        img = self.fix_image(img)
        model = self.fix_image(model)
        ycen = self.fix_linear(ycen, self.ncol)
        x_spec = self.fix_linear(x_spec, self.ncol * self.nrow)
        y_spec = self.fix_linear(y_spec, self.ncol * self.nrow)
        spec = self.fix_linear(spec, self.ncol)
        x_slit = self.fix_linear(x_slit, self.ncol * self.nrow)
        y_slit = self.fix_linear(y_slit, self.ncol * self.nrow)
        sf = self.fix_linear(sf, self.nrow)

        # Update Data
        self.im_obs.set_data(img)
        self.im_model.set_data(model)
        self.line_ycen.set_ydata(ycen)
        self.dots_spec.set_xdata(x_spec)
        self.dots_spec.set_ydata(y_spec)
        self.line_spec.set_ydata(spec)

        self.mask_spec.set_xdata(mask_spec_x)
        self.mask_spec.set_ydata(mask_spec)

        self.dots_slit.set_xdata(x_slit)
        self.dots_slit.set_ydata(y_slit)
        self.line_slit.set_ydata(sf)

        self.mask_slit.set_xdata(mask_slit_x)
        self.mask_slit.set_ydata(mask_slit)

        self.im_obs.set_norm(
            mcolors.Normalize(vmin=np.nanpercentile(img, 5), vmax=np.nanpercentile(img, 95))
        )
        self.im_model.set_norm(
            mcolors.Normalize(vmin=np.nanmin(model), vmax=np.nanmax(model))
        )


        limit = np.nanpercentile(y_spec, 95) * 1.1
        if not np.isnan(limit):
            self.ax2.set_ylim((0, limit))
        
        limit = np.nanpercentile(y_slit, 95) * 1.1
        if not np.isnan(limit):
            self.ax3.set_ylim((0, limit))

        self.fig.suptitle("Order %i, Columns %i - %i" % (ord_num, left, right))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close()

    def get_spec(self, img, spec, slitf, ycen):
        """ get the spectrum corrected by the slit function """
        x = np.indices(img.shape)[1].ravel()
        ycen = ycen - ycen.astype(int)

        nsf = len(slitf)
        nrow, ncol = img.shape
        sf = np.zeros(img.shape)
        new = np.linspace(0, nrow - 1, nrow)
        for i in range(ncol):
            old = np.linspace(-1, nrow - 1 + 1, nsf) + (ycen[i] - 0.5)
            sf[:, i] = np.interp(new, old, slitf)

        y = img / sf
        y = y.ravel() * np.mean(spec) / np.mean(y)
        return x, y

    def get_slitf(self, img, spec, slitf):
        """ get the slit function """
        x = np.indices(img.shape)[0].ravel()
        if np.any(spec == 0):
            i = np.arange(len(spec))
            try:
                spec = interp1d(i[spec != 0], spec[spec != 0], fill_value="extrapolate")(i)
            except ValueError:
                spec[spec == 0] = np.mean(spec)
        y = img / spec[None, :]
        y = y.ravel() * np.mean(slitf) / np.mean(y)
        return x, y


def make_bins(swath_width, xlow, xhigh, ycen, ncol):
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
    ncol : int
        number of columns in the image

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


def calc_telluric_correction(telluric, img):
    """ Calculate telluric correction

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
    """ Calculate scatter correction
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
    telluric=None,
    scatter=None,
    normalize=False,
    threshold=0,
    tilt=None,
    shear=None,
    plot=False,
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
        wether to create a normalized image. If true, im_norm and im_ordr are used as output (default: False)
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
    model : array[ncol, nslitf]
        model of the image, based on the spectrum and slitfunction
    unc : array[ncol]
        uncertainty on the spectrum
    """

    _, ncol = img.shape
    ylow, yhigh = yrange
    xlow, xhigh = xrange
    nslitf = osample * (ylow + yhigh + 2) + 1
    height = yhigh + ylow + 1

    ycen_int = np.floor(ycen).astype(int)

    if tilt is None:
        tilt = np.zeros(ncol)
    if shear is None:
        shear = np.zeros(ncol)

    if out_spec is None:
        spec = np.zeros(ncol)
    else:
        spec = out_spec
    if out_sunc is None:
        sunc = np.zeros(ncol)
    else:
        sunc = out_sunc

    nbin, bins_start, bins_end = make_bins(swath_width, xlow, xhigh, ycen, ncol)
    nswath = 2 * nbin - 1
    swath_slitf = np.zeros((nswath, nslitf))
    margin = np.zeros((nswath, 2), int)

    swath_spec = [None] * nswath
    swath_unc = [None] * nswath
    if normalize:
        norm_img = [None] * nswath
        norm_model = [None] * nswath

    # Perform slit decomposition within each swath stepping through the order with
    # half swath width. Spectra for each decomposition are combined with linear weights.
    for ihalf, (ibeg, iend) in enumerate(zip(bins_start, bins_end)):
        logging.debug("Extracting Swath %i, Columns: %i - %i", ihalf, ibeg, iend)

        # Cut out swath from image
        index = make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
        swath_img = img[index]
        swath_ycen = ycen[ibeg:iend]

        # Corrections
        # TODO: what is it even supposed to do?
        if telluric is not None:
            telluric_correction = calc_telluric_correction(telluric, swath_img)
        else:
            telluric_correction = 0

        if scatter is not None:
            scatter_correction = calc_scatter_correction(scatter, index)
        else:
            scatter_correction = 0

        swath_img -= scatter_correction + telluric_correction
        swath_img = np.clip(swath_img, 0, None)

        # Do Slitfunction extraction
        swath_tilt = tilt[ibeg:iend]
        swath_shear = shear[ibeg:iend]
        swath_spec[ihalf], swath_slitf[ihalf], swath_model, swath_unc[
            ihalf
        ], swath_mask = slitfunc_curved(
            swath_img,
            swath_ycen,
            swath_tilt,
            swath_shear,
            lambda_sp=lambda_sp,
            lambda_sf=lambda_sf,
            osample=osample,
        )

        if not np.all(np.isfinite(swath_spec[ihalf])):
            # TODO: Why does this happen?
            logging.warning("Curved extraction failed, using Tilt=Shear=0 instead")
            swath_spec[ihalf], swath_slitf[ihalf], swath_model, swath_unc[
                ihalf
            ], swath_mask = slitfunc_curved(
                swath_img,
                swath_ycen,
                0,
                0,
                lambda_sp=lambda_sp,
                lambda_sf=lambda_sf,
                osample=osample,
            )

        if normalize:
            # Save image and model for later
            # Use np.divide to avoid divisions by zero
            where = swath_model > threshold / gain
            norm_img[ihalf] = np.ones_like(swath_model)
            np.divide(swath_img, swath_model, where=where, out=norm_img[ihalf])
            norm_model[ihalf] = swath_model

        if plot:
            if not np.all(np.isnan(swath_img)):
                if progress is None:
                    progress = ProgressPlot(swath_img.shape[0], swath_img.shape[1])
                progress.plot(
                    swath_img,
                    swath_spec[ihalf],
                    swath_slitf[ihalf],
                    swath_model,
                    swath_ycen,
                    swath_mask,
                    ord_num,
                    ibeg,
                    iend,
                )

    # Remove points at the border of the each swath, if order has tilt
    # as those pixels have bad information
    if tilt is not None:
        for i, (ibeg, iend) in enumerate(zip(bins_start, bins_end)):
            swath_tilt = tilt[ibeg:iend]
            swath_shear = shear[ibeg:iend]
            tilt_first, tilt_last = swath_tilt[[0, -1]]
            shear_first, shear_last = swath_shear[[0, -1]]

            excess = np.polyval([shear_first, tilt_first, 0], [ylow, -yhigh])
            margin[i, 0] = abs(int(np.ceil(excess).max()))

            excess = np.polyval([shear_last, tilt_last, 0], [-ylow, yhigh])
            margin[i, 1] = abs(int(np.ceil(excess).max()))

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

    # control = np.zeros(ncol)
    # for i, (ibeg, iend) in enumerate(zip(bins_start, bins_end)):
    #     control[ibeg:iend] += weight[i]
    # assert np.all(control[bins_start[0] + margin[0, 0] : bins_end[-1] - margin[-1, 1]] == 1)

    # Update column range
    xrange[0] += margin[0, 0]
    xrange[1] -= margin[-1, 1]
    if out_mask is not None:
        out_mask[: xrange[0]] = out_mask[xrange[1] :] = True

    # Apply weights
    for i, (ibeg, iend) in enumerate(zip(bins_start, bins_end)):
        spec[ibeg:iend] += swath_spec[i] * weight[i]
        sunc[ibeg:iend] += swath_unc[i] * weight[i]

    if normalize:
        for i, (ibeg, iend) in enumerate(zip(bins_start, bins_end)):
            index = make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
            im_norm[index] += norm_img[i] * weight[i]
            im_ordr[index] += norm_model[i] * weight[i]

    slitf = np.mean(swath_slitf, axis=0)
    if out_slitf is not None:
        out_slitf[:] = slitf

    sunc[:] = np.sqrt(sunc ** 2 + (readnoise / gain) ** 2)

    model = spec[None, :] * slitf[:, None]
    return spec, slitf, model, sunc


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
    img, orders, extraction_width, column_range, tilt, shear, **kwargs
):
    """ Use optimal extraction to get spectra

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

    logging.info("Using optimal extraction to produce spectrum")

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
    if kwargs.get("plot", False):
        ncol_swath = kwargs.get("swath_width", img.shape[1] // 400)
        nrow_swath = np.sum(extraction_width, axis=1).max()
        progress = ProgressPlot(nrow_swath, ncol_swath)
    else:
        progress = None

    for i in range(nord):
        if nord < 10 or i % 5 == 0:
            logging.info("Extracting relative order %i out of %i", i + 1, nord)
        else:
            logging.debug("Extracting relative order %i out of %i", i + 1, nord)

        # Define a fixed height area containing one spectral order
        ycen = np.polyval(orders[i], ix)
        yrange = get_y_scale(ycen, column_range[i], extraction_width[i], nrow)

        # Return values are set by reference, as the out parameters
        # Also column_range is adjusted depending on the shear
        # This is to avoid large chunks of memory of essentially duplicates
        _, slitfunction[i], _, _ = extract_spectrum(
            img,
            ycen,
            yrange,
            column_range[i],
            tilt=tilt[i],
            shear=shear[i],
            out_spec=spectrum[i],
            out_sunc=uncertainties[i],
            out_mask=mask[i],
            progress=progress,
            ord_num=i + 1,
            **kwargs,
        )

    if kwargs.get("plot", False):
        progress.close()

    slitfunction = np.asarray(slitfunction)
    return spectrum, slitfunction, uncertainties


def arc_extraction(
    img,
    orders,
    extraction_width,
    column_range,
    gain=1,
    readnoise=0,
    dark=0,
    plot=False,
    **kwargs,
):
    """ Use "simple" arc extraction to get a spectrum
    Arc extraction simply takes the sum orthogonal to the order for extraction width pixels

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

    logging.info("Using arc extraction to produce spectrum.")
    _, ncol = img.shape
    nord, _ = orders.shape

    if plot:
        # Prepare output image
        output = np.zeros((np.sum(extraction_width) + nord, ncol))
        pos = [0]

    spectrum = np.zeros((nord, ncol))
    uncertainties = np.zeros((nord, ncol))

    # Add mask as defined by column ranges
    mask = np.full((nord, ncol), True)
    for i, onum in enumerate(range(1, nord - 1)):
        mask[i, column_range[onum, 0] : column_range[onum, 1]] = False
    spectrum = np.ma.array(spectrum, mask=mask)
    uncertainties = np.ma.array(uncertainties, mask=mask)

    x = np.arange(ncol)

    for i in range(nord):  # loop thru orders
        x_left_lim = column_range[i, 0]  # First column to extract
        x_right_lim = column_range[i, 1]  # Last column to extract

        ycen = np.polyval(orders[i], x).astype(int)
        yb, yt = ycen - extraction_width[i, 0], ycen + extraction_width[i, 1]
        index = make_index(yb, yt, x_left_lim, x_right_lim)

        # Sum over the prepared index
        arc = np.sum(img[index], axis=0)

        spectrum[i, x_left_lim:x_right_lim] = arc  # store total counts
        uncertainties[i, x_left_lim:x_right_lim] = (
            np.sqrt(np.abs(arc * gain + dark + readnoise ** 2)) / gain
        )  # estimate uncertainty

        if plot:
            output[pos[i] : pos[i] + index[0].shape[0], x_left_lim:x_right_lim] = img[
                index
            ]
            pos += [pos[i] + index[0].shape[0]]

    if plot:
        plt.title("Extracted Spectrum vs. Input Image")
        plt.xlabel("x [pixel]")
        plt.ylabel("order")
        locs = np.sum(extraction_width, axis=1) + 1
        locs = [0, *np.cumsum(locs)[:-1]]
        plt.yticks(locs, range(len(locs)))
        plt.imshow(
            output,
            vmin=0,
            vmax=np.mean(output) + 5 * np.std(output),
            origin="lower",
            aspect="auto",
        )

        for i in range(nord):
            tmp = spectrum[i] - np.min(
                spectrum[i, column_range[i, 0] : column_range[i, 1]]
            )
            tmp = tmp / np.max(tmp) * 0.9 * (pos[i + 1] - pos[i])
            tmp += pos[i]
            tmp[tmp < pos[i]] = pos[i]
            plt.plot(x, tmp)

        plt.show()

    return spectrum, uncertainties


def fix_column_range(img, orders, extraction_width, column_range, no_clip=False):
    """ Fix the column range, so that no pixels outside the image will be accessed (Thus avoiding errors)

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

    nrow, ncol = img.shape
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
        if not no_clip:
            column_range[i] = np.clip(
                regions[iregion], column_range[i, 0], column_range[i, 1]
            )
        else:
            column_range[i] = regions[iregion]

    column_range[0] = column_range[1]
    column_range[-1] = column_range[-2]

    return column_range


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


def fix_extraction_width(
    extraction_width, orders, column_range, ncol, img=None, plot=False
):
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

    if not np.all(extraction_width > 1.5):
        # if extraction width is in relative scale transform to pixel scale
        x = np.arange(ncol)
        for i in range(1, len(extraction_width) - 1):
            for j in [0, 1]:
                if extraction_width[i, j] < 1.5:
                    k = i - 1 if j == 0 else i + 1
                    left = max(column_range[[i, k], 0])
                    right = min(column_range[[i, k], 1])

                    current = np.polyval(orders[i], x[left:right])
                    below = np.polyval(orders[k], x[left:right])
                    extraction_width[i, j] *= np.abs(np.mean(current - below))

        extraction_width[0] = extraction_width[1]
        extraction_width[-1] = extraction_width[-2]

    extraction_width = np.ceil(extraction_width).astype(int)

    if plot and img is not None:
        plt.imshow(img, aspect="auto", origin="lower")
        for i in range(len(extraction_width)):
            left, right = column_range[i]
            xwd = extraction_width[i]
            current = np.polyval(orders[i], x[left:right])

            plt.plot(x[left:right], current, "k-")
            plt.plot(x[left:right], np.round(current - xwd[0]), "k--")
            plt.plot(x[left:right], np.round(current + xwd[1]), "k--")
        plt.show()

    return extraction_width


def extract(
    img,
    orders,
    column_range=None,
    order_range=None,
    extraction_width=0.5,
    extraction_type="optimal",
    tilt=None,
    shear=None,
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
    if column_range is None:
        column_range = np.tile([0, ncol], (nord, 1))
    if np.isscalar(extraction_width):
        extraction_width = np.tile([extraction_width, extraction_width], (nord, 1))
    else:
        extraction_width = np.asarray(extraction_width)
        if extraction_width.ndim == 1:
            extraction_width = np.tile(extraction_width, (nord, 1))
        

    # Limit orders (and related properties) to orders in range
    nord = order_range[1] - order_range[0]
    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]
    extraction_width = extraction_width[order_range[0] : order_range[1]]

    # Extend orders and related properties
    orders = extend_orders(orders, nrow)
    extraction_width = np.array(
        [extraction_width[0], *extraction_width, extraction_width[-1]]
    )
    column_range = np.array([column_range[0], *column_range, column_range[-1]])

    # Fix column range, so that all extractions are fully within the image
    extraction_width = fix_extraction_width(
        extraction_width, orders, column_range, ncol
    )
    column_range = fix_column_range(img, orders, extraction_width, column_range)

    orders = orders[1:-1]
    extraction_width = extraction_width[1:-1]
    column_range = column_range[1:-1]

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
        im_norm[im_norm == 0] = 1
        im_ordr[im_ordr == 0] = 1
        return im_norm, im_ordr, blaze, column_range
    elif extraction_type == "arc":
        # Simpler extraction, just summing along the arc of the order
        spectrum, uncertainties = arc_extraction(
            img, orders, extraction_width, column_range, **kwargs
        )
        slitfunction = None
    else:
        raise ValueError(
            f"Parameter 'extraction_type' not understood. Expected 'optimal', 'normalize', or 'arc' bug got {extraction_type}."
        )

    return spectrum, uncertainties, slitfunction, column_range


class Extraction:
    def __init__(
        self,
        orders,
        tilt=None,
        shear=None,
        column_range=None,
        order_range=None,
        extraction_width=0.5,
        extraction_type="optimal",
        oversampling=1,
        gain=1,
        readnoise=0,
        dark=0,
        plot=False,
    ):
        self.extraction_type = extraction_type
        self.orders = orders
        self._order_range = order_range
        self._column_range = column_range
        self._extraction_width = extraction_width
        self._tilt = tilt
        self._shear = shear

        self.oversampling = oversampling
        self.gain = gain
        self.readnoise = readnoise
        self.dark = dark
        self.plot = plot

        self.nrow = self.ncol = None

    def _fix_column_range(
        self, img, orders, extraction_width, column_range, no_clip=False
    ):
        """ Fix the column range, so that no pixels outside the image will be accessed (Thus avoiding errors)

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

        nrow, ncol = img.shape
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
            regions = [
                [regions[i], regions[i + 1] + 1] for i in range(0, len(regions), 2)
            ]
            overlap = [
                min(reg[1], column_range[i, 1]) - max(reg[0], column_range[i, 0])
                for reg in regions
            ]
            iregion = np.argmax(overlap)
            if not no_clip:
                column_range[i] = np.clip(
                    regions[iregion], column_range[i, 0], column_range[i, 1]
                )
            else:
                column_range[i] = regions[iregion]

        column_range[0] = column_range[1]
        column_range[-1] = column_range[-2]

        return column_range

    def _extend_orders(self, orders):
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
            order_high = [0 for _ in range(ncoef - 1)] + [self.nrow]

        return np.array([order_low, *orders, order_high])

    def _extend_column_range(self, cr):
        """ Pad the column range with the first and last element of itself """
        return np.array([cr[0], *cr, cr[-1]])

    def _extend_extraction_width(self, xwd):
        return np.array([xwd[0], *xwd, xwd[-1]])

    def _fix_extraction_width(
        self, extraction_width, orders, column_range, img=None, plot=False
    ):
        """Convert fractional extraction width to pixel range

        Parameters
        ----------
        extraction_width : array[nord, 2]
            current extraction width, in pixels or fractions (for values below 1.5)
        orders : array[nord, degree]
            order tracing coefficients
        column_range : array[nord, 2]
            column range to use

        Returns
        -------
        extraction_width : array[nord, 2]
            updated extraction width in pixels
        """

        if not np.all(extraction_width > 1.5):
            # if extraction width is in relative scale transform to pixel scale
            x = np.arange(self.ncol)
            for i in range(1, len(extraction_width) - 1):
                for j in [0, 1]:
                    if extraction_width[i, j] < 1.5:
                        k = i - 1 if j == 0 else i + 1
                        left = max(column_range[[i, k], 0])
                        right = min(column_range[[i, k], 1])

                        current = np.polyval(orders[i], x[left:right])
                        below = np.polyval(orders[k], x[left:right])
                        extraction_width[i, j] *= np.abs(np.mean(current - below))

            extraction_width[0] = extraction_width[1]
            extraction_width[-1] = extraction_width[-2]

        extraction_width = np.ceil(extraction_width).astype(int)

        if plot and img is not None:
            plt.imshow(img, aspect="auto", origin="lower")
            for i in range(len(extraction_width)):
                left, right = column_range[i]
                xwd = extraction_width[i]
                current = np.polyval(orders[i], x[left:right])

                plt.plot(x[left:right], current, "k-")
                plt.plot(x[left:right], np.round(current - xwd[0]), "k--")
                plt.plot(x[left:right], np.round(current + xwd[1]), "k--")
            plt.show()

        return extraction_width

    def _fix(self, img):
        self.nrow, self.ncol = img.shape

        # Limit orders (and related properties) to orders in range
        slices = slice(self.order_range[0], self.order_range[1], None)
        orders = self.orders[slices]
        cr = self.column_range[slices]
        xwd = self.extraction_width[slices]

        # Extend orders and related properties
        # For extrapolation
        orders = self._extend_orders(orders)
        xwd = self._extend_extraction_width(xwd)
        cr = self._extend_column_range(cr)

        # Fix column range and extraction width, so that all extractions are fully within the image
        xwd = self._fix_extraction_width(xwd, orders, cr)
        cr = self._fix_column_range(img, orders, xwd, cr)

        # Remove temporary extended orders
        orders = orders[1:-1]
        xwd = xwd[1:-1]
        cr = cr[1:-1]

        return orders, cr, xwd

    @property
    def nord(self):
        return self.order_range[-1] - self.order_range[0]

    @property
    def order_range(self):
        if self._order_range is None:
            return (0, len(self.orders))
        else:
            return self._order_range

    @property
    def column_range(self):
        if self._column_range is None:
            if self.ncol is not None and self.nord is not None:
                return np.tile([0, self.ncol], (self.nord, 1))
            else:
                return None
        else:
            return self._column_range[self.order_range[0]:self.order_range[1]]

    @property
    def extraction_width(self):
        if np.isscalar(self._extraction_width):
            if self.nord is not None:
                return np.tile(
                    [self._extraction_width, self._extraction_width], (self.nord, 1)
                )
        return self._extraction_width

    @property
    def tilt(self):
        if np.isscalar(self._tilt):
            if self.ncol is not None:
                return np.full((self.nord, self.ncol), self._tilt)
            else:
                return self._tilt
        else:
            return self._tilt

    @property
    def shear(self):
        if np.isscalar(self._shear):
            if self.ncol is not None:
                return np.full((self.nord, self.ncol), self._shear)
            else:
                return self._shear
        else:
            return self._shear

    def execute_optimal(
        self, img, orders, tilt, shear, extraction_width, column_range, **kwargs
    ):
        spectrum, slitfunction, uncertainties = optimal_extraction(
            img,
            orders,
            extraction_width,
            column_range,
            tilt=tilt,
            shear=shear,
            **kwargs,
        )
        return spectrum, slitfunction, uncertainties

    def execute_normalize(
        self, img, orders, tilt, shear, extraction_width, column_range, **kwargs
    ):
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
        im_norm[im_norm == 0] = 1
        im_ordr[im_ordr == 0] = 1
        return im_norm, im_ordr, blaze, column_range

    def execute_arc(self, img, orders, extraction_width, column_range, **kwargs):
        """ Use "simple" arc extraction to get a spectrum
        Arc extraction simply takes the sum orthogonal to the order for extraction width pixels

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

        Returns
        -------
        spectrum : array[nord, ncol]
            extracted spectrum
        uncertainties : array[nord, ncol]
            uncertainties on extracted spectrum
        """

        logging.info("Using arc extraction to produce spectrum.")

        if self.plot:
            # Prepare output image
            output = np.zeros((np.sum(extraction_width) + self.nord, self.ncol))
            pos = [0]

        spectrum = np.zeros((self.nord, self.ncol))
        uncertainties = np.zeros((self.nord, self.ncol))

        # Add mask as defined by column ranges
        mask = np.full((self.nord, self.ncol), True)
        for i, onum in enumerate(range(1, self.nord - 1)):
            mask[i, column_range[onum, 0] : column_range[onum, 1]] = False
        spectrum = np.ma.array(spectrum, mask=mask)
        uncertainties = np.ma.array(uncertainties, mask=mask)

        x = np.arange(self.ncol)

        for i in range(self.nord):  # loop thru orders
            x_left_lim = column_range[i, 0]  # First column to extract
            x_right_lim = column_range[i, 1]  # Last column to extract

            ycen = np.polyval(orders[i], x).astype(int)
            yb, yt = ycen - extraction_width[i, 0], ycen + extraction_width[i, 1]
            index = make_index(yb, yt, x_left_lim, x_right_lim)

            # Sum over the prepared index
            arc = np.sum(img[index], axis=0)

            spectrum[i, x_left_lim:x_right_lim] = arc  # store total counts
            uncertainties[i, x_left_lim:x_right_lim] = (
                np.sqrt(np.abs(arc * self.gain + self.dark + self.readnoise ** 2)) / self.gain
            )  # estimate uncertainty

            if self.plot:
                output[pos[i] : pos[i] + index[0].shape[0], x_left_lim:x_right_lim] = img[
                    index
                ]
                pos += [pos[i] + index[0].shape[0]]

        if self.plot:
            plt.title("Extracted Spectrum vs. Input Image")
            plt.xlabel("x [pixel]")
            plt.ylabel("order")
            locs = np.sum(extraction_width, axis=1) + 1
            locs = [0, *np.cumsum(locs)[:-1]]
            plt.yticks(locs, range(len(locs)))
            plt.imshow(
                output,
                vmin=0,
                vmax=np.mean(output) + 5 * np.std(output),
                origin="lower",
                aspect="auto",
            )

            for i in range(self.nord):
                tmp = spectrum[i] - np.min(
                    spectrum[i, column_range[i, 0] : column_range[i, 1]]
                )
                tmp = tmp / np.max(tmp) * 0.9 * (pos[i + 1] - pos[i])
                tmp += pos[i]
                tmp[tmp < pos[i]] = pos[i]
                plt.plot(x, tmp)

            plt.show()

        slitfunction = None
        return spectrum, slitfunction, uncertainties

    def execute(self, img, **kwargs):
        self.nrow, self.ncol = img.shape
        orders, column_range, extraction_width = self._fix(img)

        if self.extraction_type == "optimal":
            # the "normal" case, except for wavelength calibration files
            return self.execute_optimal(
                img,
                orders,
                self.tilt,
                self.shear,
                extraction_width,
                column_range,
                **kwargs,
            )
        elif self.extraction_type == "normalize":
            # Prepare normalized flat field image if necessary
            # These will be passed and "returned" by reference
            # I dont like it, but it works for now
            return self.execute_normalize(
                img,
                orders,
                self.tilt,
                self.shear,
                extraction_width,
                column_range,
                **kwargs,
            )
        elif self.extraction_type == "arc":
            # Simpler extraction, just summing along the arc of the order
            return self.execute_arc(
                img, orders, extraction_width, column_range, **kwargs
            )
        else:
            raise ValueError(
                f"Parameter 'extraction_type' not understood. Expected 'optimal', 'normalize', or 'arc' bug got {self.extraction_type}."
            )
