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

from .cwrappers import slitfunc, slitfunc_curved
from .util import make_index

# TODO put the plotting somewhere else


def getflatimg(img, axis=0):
    """Flatten image and indices

    Parameters
    ----------
    img : array
        image to flatten
    axis : int, optional
        axis to flatten along (default: 0)

    Returns
    -------
    index: array
        flattened indices
    img
        flat image
    """

    idx = np.indices(img.shape)[axis]
    return idx.flatten(), img.flat


def getspecvar(img):
    """ get the spectrum """
    ny, nx = img.shape
    nimg = img / np.nansum(img, axis=1)[:, None]
    x = np.indices(img.shape)[1]
    return x.flatten(), nimg.flat


def getslitvar(img, xoff, osample=1):
    """ get the slit function """
    x = np.indices(img.shape)[0]
    x = x - xoff[None, :] + 1
    return x.flatten() * osample, img.flat


def plot_slitfunction(img, spec, slitf, model, ycen, onum, left, right, osample, mask):
    """ Plot (and update the plot of) the current swath with spectrum and slitfunction """
    ny, nx = img.shape
    ny_orig = ny
    size = img.size
    ny_os = len(slitf)
    os = (ny_os - 1) / (ny + 1)

    idx = np.indices(img.shape)

    di = img.data / np.sum(img)
    ds = spec.data / np.sum(spec)

    df = np.copy(slitf)
    dm = model / np.sum(model)

    max_bad_pixels = 100

    xbad = np.full(max_bad_pixels, np.nan)
    ybad = np.full(max_bad_pixels, np.nan)
    ibad = np.zeros(max_bad_pixels)
    jbad = np.zeros(max_bad_pixels)
    n = np.nonzero(mask)[0].size
    xbad[:n] = (di / np.nansum(di, axis=1)[:, None])[mask][:max_bad_pixels]
    ybad[:n] = (di[mask] * nx)[:max_bad_pixels]
    ibad[:n] = idx[0][mask][:max_bad_pixels]
    jbad[:n] = idx[1][mask][:max_bad_pixels]

    # on first execution of plot create a new figure
    if not hasattr(plot_slitfunction, "fig"):
        # If no figure exists, create a new one
        plt.ion()
        FIG = plt.figure(figsize=(12, 4))
        FIG.tight_layout(pad=0.05)

        AX1 = FIG.add_subplot(231)
        AX1.set_title("Swath")
        AX2 = FIG.add_subplot(132)
        AX2.set_title("Spectrum")
        AX3 = FIG.add_subplot(133)
        AX3.set_title("Slit")
        AX4 = FIG.add_subplot(234)
        AX4.set_title("Model")

        im1 = AX1.imshow(di, aspect="auto", origin="lower")
        line1, = AX1.plot(ny_orig / 2 + ycen, "-r")
        im4 = AX4.imshow(dm, aspect="auto", origin="lower")

        specvar, = AX2.plot(*getspecvar(di), ".r", ms=2, alpha=0.6)
        slitvar, = AX3.plot(*getslitvar(di * nx, ycen), ".r", ms=2, alpha=0.6)
        offset = 0.5 + ycen[0]
        slitfu, = AX3.plot(
            np.linspace(-(1 + offset), di.shape[0] + offset, len(df)), df, "-k", lw=3
        )  # TODO which limits for the x axis?

        masked, = AX3.plot(ibad, ybad, "+g")
        masked2, = AX2.plot(jbad, xbad, "+g")

        line2, = AX2.plot(ds, "-k")

        # Save plots to this function
        setattr(plot_slitfunction, "fig", FIG)
        setattr(plot_slitfunction, "ax1", AX1)
        setattr(plot_slitfunction, "ax2", AX2)
        setattr(plot_slitfunction, "ax3", AX3)
        setattr(plot_slitfunction, "ax4", AX4)
        setattr(plot_slitfunction, "ny", ny)
        setattr(plot_slitfunction, "nx", nx)

        setattr(plot_slitfunction, "im1", im1)
        setattr(plot_slitfunction, "line1", line1)
        setattr(plot_slitfunction, "line2", line2)
        setattr(plot_slitfunction, "masked", masked)
        setattr(plot_slitfunction, "masked2", masked2)
        setattr(plot_slitfunction, "specvar", specvar)
        setattr(plot_slitfunction, "slitvar", slitvar)
        setattr(plot_slitfunction, "slitfu", slitfu)
        setattr(plot_slitfunction, "im4", im4)
    else:
        FIG = plot_slitfunction.fig
        AX1 = plot_slitfunction.ax1
        AX2 = plot_slitfunction.ax2
        AX3 = plot_slitfunction.ax3
        AX4 = plot_slitfunction.ax4
        im1 = plot_slitfunction.im1
        line2 = plot_slitfunction.line2
        im4 = plot_slitfunction.im4
        line1 = plot_slitfunction.line1
        masked = plot_slitfunction.masked
        masked2 = plot_slitfunction.masked2
        specvar = plot_slitfunction.specvar
        slitvar = plot_slitfunction.slitvar
        slitfu = plot_slitfunction.slitfu

        ny = plot_slitfunction.ny
        nx = plot_slitfunction.nx

    # Fix size of array
    if di.shape[0] > ny:
        di = di[:ny, :]
        df = df[: ny + 2]
        dm = dm[:ny, :]
    elif di.shape[0] < ny:
        ypad = 0, ny - di.shape[0]
        di = np.pad(di, (ypad, (0, 0)), "constant", constant_values=np.nan)
        df = np.pad(df, (0, ny + 2 - df.shape[0]), "constant", constant_values=np.nan)
        dm = np.pad(dm, (ypad, (0, 0)), "constant", constant_values=np.nan)

    if di.shape[1] > nx:
        di = di[:, :nx]
        ds = ds[:nx]
        dm = dm[:, :nx]
        ycen = ycen[:nx]
    elif di.shape[1] < nx:
        xpad = 0, nx - di.shape[1]
        di = np.pad(di, ((0, 0), xpad), "constant", constant_values=np.nan)
        ds = np.pad(ds, xpad, "constant", constant_values=np.nan)
        dm = np.pad(dm, ((0, 0), xpad), "constant", constant_values=np.nan)
        ycen = np.pad(ycen, xpad, "constant", constant_values=np.nan)

    # Update data
    FIG.suptitle("Order %i, Columns %i - %i" % (onum, left, right))

    im1.set_norm(mcolors.Normalize(vmin=np.nanmin(di), vmax=np.nanmax(di)))
    im1.set_data(di)
    im4.set_norm(mcolors.Normalize(vmin=np.nanmin(dm), vmax=np.nanmax(dm)))
    im4.set_data(dm)

    line1.set_ydata(ny_orig / 2 + ycen)
    line2.set_ydata(ds)

    slitvar.set_data(*getslitvar(di * nx, ycen))
    slitfu.set_ydata(df)
    specvar.set_data(*getspecvar(di))

    masked.set_xdata(ibad)
    masked.set_ydata(ybad)

    masked2.set_xdata(jbad)
    masked2.set_ydata(xbad)

    # Set new limits
    AX1.set_xlim((0, img.shape[1]))
    AX1.set_ylim((0, img.shape[0]))
    AX4.set_xlim((0, img.shape[1]))
    AX4.set_ylim((0, img.shape[0]))

    AX2.set_xlim((0, len(spec)))
    AX3.set_xlim((0, img.shape[0]))
    AX2.set_ylim((0, np.nanmax(di / np.nansum(di, axis=1)[:, None]) * 1.1))
    AX3.set_ylim((0, np.nanmax(di) * nx * 1.1))

    FIG.canvas.draw()
    FIG.canvas.flush_events()
    # plt.show()


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
    bins_start = np.floor(bins[:-2]).astype(int)  # beginning of each bin
    bins_end = np.ceil(bins[2:]).astype(int)  # end of each bin

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
        sc[itel] = np.median(tel[itel])

    return sc


def calc_scatter_correction(scatter, ycen_low, height):
    """ Calculate scatter correction

    Parameters
    ----------
    scatter : array[4, ncol]
        background scattered light, (below, above, ybelow, yabove)
    ycen_low : array[ncol]
        y center of the order
    height : int
        height of the extraction window (?)

    Returns
    -------
    scatter_correction : array[ncol]
        correction for scattered light
    """

    # scatter = (below, above, ybelow, yabove)
    index_y = np.array([np.arange(k, height + k) for k in ycen_low])
    dy_scatter = (index_y.T - scatter[2][None, :]) / (
        scatter[3][None, :] - scatter[2][None, :]
    )
    scatter_correction = (
        scatter[1][None, :] - scatter[0][None, :]
    ) * dy_scatter + scatter[0][None, :]

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
    shear=None,
    plot=False,
    ord_num=0,
    im_norm=None,
    im_ordr=None,
    **kwargs
):
    """
    Extract the spectrum of a single order from an image

    The order is split into several swathes of roughly swath_width length, which overlap half-half
    For each swath a spectrum and slitfunction are extracted
    overlapping sections are combined using linear weights (centrum is strongest, falling off to the edges)

    Here is the layout for the bins:

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
    scatter : {array[4, ncol], None}, optional
        background scatter (below, above, ybelow, yabove) (default: None, no correction)
    normalize : bool, optional
        wether to create a normalized image. If true, im_norm and im_ordr are used as output (default: False)
    threshold : int, optional
        threshold for normalization (default: 0)
    shear : array[ncol], optional
        The shear (tilt) of the order, if given will use curved extraction instead of vertical extraction (default: None, i.e. no shear)
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

    spec = np.zeros(ncol)
    sunc = np.zeros(ncol)

    nbin, bins_start, bins_end = make_bins(swath_width, xlow, xhigh, ycen, ncol)
    slitf = np.zeros((2 * nbin, nslitf))

    swath_spec = [None for _ in range(2 * nbin - 1)]
    swath_unc = [None for _ in range(2 * nbin - 1)]
    if normalize:
        norm_img = [None for _ in range(2 * nbin - 1)]
        norm_model = [None for _ in range(2 * nbin - 1)]

    # Perform slit decomposition within each swath stepping through the order with
    # half swath width. Spectra for each decomposition are combined with linear weights.
    for ihalf, (ibeg, iend) in enumerate(zip(bins_start, bins_end)):
        width = iend - ibeg  # number of columns in swath
        logging.debug("Extracting Swath %i, Columns: %i - %i", ihalf, ibeg, iend)

        # Cut out swath from image
        index = make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
        swath_img = img[index]

        # swath_img = cutout_image(img, ycen_int - ylow, ycen_int + yhigh, ibeg, iend)

        # Corrections
        if telluric is not None:
            telluric_correction = calc_telluric_correction(telluric, swath_img)
        else:
            telluric_correction = 0

        if scatter is not None:
            scatter_correction = calc_scatter_correction(
                scatter[:, ibeg:iend], ycen_int[ibeg:iend] - ylow, height
            )
        else:
            scatter_correction = 0

        # Do Slitfunction extraction
        swath_img -= scatter_correction + telluric_correction
        swath_img = np.clip(swath_img, 0, None)

        # offset from the central line
        y_offset = ycen[ibeg:iend] - ycen_int[ibeg:iend]

        if shear is None:
            swath_spec[ihalf], slitf[ihalf], swath_model, swath_unc[
                ihalf
            ], mask = slitfunc(
                swath_img,
                y_offset,
                lambda_sp=lambda_sp,
                lambda_sf=lambda_sf,
                osample=osample,
            )
        else:
            swath_spec[ihalf], slitf[ihalf], swath_model, swath_unc[
                ihalf
            ], mask = slitfunc_curved(
                swath_img,
                y_offset,
                shear,
                lambda_sp=lambda_sp,
                lambda_sf=lambda_sf,
                osample=osample,
            )

        if normalize:
            # Save image and model for later
            norm_img[ihalf] = np.where(
                swath_model > threshold / gain, swath_img / swath_model, 1
            )
            norm_model[ihalf] = swath_model

        if plot:
            if not np.all(np.isnan(swath_img)):
                plot_slitfunction(
                    swath_img,
                    swath_spec[ihalf],
                    slitf[ihalf],
                    swath_model,
                    y_offset,
                    ord_num,
                    ibeg,
                    iend,
                    osample,
                    mask,
                )

    # Weight for combining overlapping regions
    weight = [np.ones(bins_end[i] - bins_start[i]) for i in range(nbin * 2 - 1)]
    for i, j in zip(range(0, nbin * 2 - 2), range(1, nbin * 2 - 1)):
        width = bins_end[i] - bins_start[i]

        overlap_start = bins_start[j] - bins_start[i]
        overlap = width - overlap_start

        weight[i][overlap_start:] = np.linspace(1, 0, overlap)
        weight[j][:overlap] = np.linspace(0, 1, overlap)

    # DEBUG: Check weigths
    # total_weight = np.zeros(ncol)
    # for i, (ib, ie) in enumerate(zip(bins_start, bins_end)):
    #     total_weight[ib:ie] += weight[i]
    # if not np.all(total_weight[xlow:xhigh] == 1):
    #     raise Exception("Weights are wrong")

    # Remove points at the border of the image, if order has shear
    # as those pixels have bad information
    if shear is not None:
        if np.isscalar(shear):
            shear = [shear, shear]
        y = yhigh if shear[0] < 0 else ylow
        shear_margin = int(np.ceil(shear[0] * y))
        weight[0][:shear_margin] = 0
        y = ylow if shear[-1] < 0 else yhigh
        shear_margin = int(np.ceil(shear[-1] * y))
        if shear_margin != 0:
            weight[-1][-shear_margin:] = 0

    # Apply weights
    for i, (ibeg, iend) in enumerate(zip(bins_start, bins_end)):
        spec[ibeg:iend] += swath_spec[i] * weight[i]
        sunc[ibeg:iend] += swath_unc[i] * weight[i]

        if normalize:
            index = make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
            im_norm[index] += norm_img[i] * weight[i]
            im_ordr[index] += norm_model[i] * weight[i]

    slitf = np.mean(slitf, axis=0)
    sunc = np.sqrt(sunc ** 2 + (readnoise / gain) ** 2)

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
    img, orders, extraction_width, column_range, scatter, shear, **kwargs
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

    spectrum = np.zeros((nord - 2, ncol))
    uncertainties = np.zeros((nord - 2, ncol))
    slitfunction = [None for _ in range(nord - 2)]

    # Add mask as defined by column ranges
    mask = np.full((nord - 2, ncol), True)
    for i, onum in enumerate(range(1, nord - 1)):
        mask[i, column_range[onum, 0] : column_range[onum, 1]] = False
    spectrum = np.ma.array(spectrum, mask=mask)
    uncertainties = np.ma.array(uncertainties, mask=mask)

    ix = np.arange(ncol)

    for i, onum in enumerate(range(1, nord - 1)):
        if nord < 10 or onum % 5 == 0:
            logging.info("Extracting relative order %i out of %i", onum, nord - 2)

        # Define a fixed height area containing one spectral order
        ycen = np.polyval(orders[onum], ix)
        yrange = get_y_scale(ycen, column_range[onum], extraction_width[onum], nrow)

        spectrum[i], slitfunction[i], _, uncertainties[i] = extract_spectrum(
            img,
            ycen,
            yrange,
            column_range[onum],
            scatter=scatter[i],
            shear=shear[i],
            ord_num=onum - 1,
            **kwargs
        )

    if kwargs.get("plot", False):
        plt.ioff()
        plt.close()

    slitfunction = np.array(slitfunction)
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
    **kwargs
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
        output = np.zeros((np.sum(extraction_width[1:-1]) + nord - 2, ncol))
        pos = [0]

    spectrum = np.zeros((nord - 2, ncol))
    uncertainties = np.zeros((nord - 2, ncol))

    # Add mask as defined by column ranges
    mask = np.full((nord - 2, ncol), True)
    for i, onum in enumerate(range(1, nord - 1)):
        mask[i, column_range[onum, 0] : column_range[onum, 1]] = False
    spectrum = np.ma.array(spectrum, mask=mask)
    uncertainties = np.ma.array(uncertainties, mask=mask)

    x = np.arange(ncol)

    for i, onum in enumerate(range(1, nord - 1)):  # loop thru orders
        x_left_lim = column_range[onum, 0]  # First column to extract
        x_right_lim = column_range[onum, 1]  # Last column to extract

        ycen = np.polyval(orders[onum], x).astype(int)
        yb, yt = ycen - extraction_width[onum, 0], ycen + extraction_width[onum, 1]
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
        plt.imshow(
            output,
            vmin=0,
            vmax=np.mean(output) + 5 * np.std(output),
            origin="lower",
            aspect="auto",
        )

        for i in range(nord - 2):
            tmp = spectrum[i] - np.min(
                spectrum[i, column_range[i + 1, 0] : column_range[i + 1, 1]]
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
        regions = np.where(np.diff(points_in_image) != 1)[0]
        regions = [(r, r + 1) for r in regions]
        regions = [points_in_image[0], *points_in_image[regions], points_in_image[-1]]
        regions = [(regions[i], regions[i + 1]) for i in range(0, len(regions), 2)]
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

        column_range[i, 1] += 1

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


def fix_extraction_width(extraction_width, orders, column_range, ncol):
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

    if np.all(extraction_width > 1.5):
        # already in pixel scale
        extraction_width = extraction_width.astype(int)
        return extraction_width

    x = np.arange(ncol)
    # if extraction width is in relative scale transform to pixel scale
    for i in range(1, len(extraction_width) - 1):
        left, right = column_range[i]
        current = np.polyval(orders[i], x[left:right])

        if extraction_width[i, 0] < 1.5:
            below = np.polyval(orders[i - 1], x[left:right])
            extraction_width[i, 0] *= np.mean(current - below)
        if extraction_width[i, 1] < 1.5:
            above = np.polyval(orders[i + 1], x[left:right])
            extraction_width[i, 1] *= np.mean(above - current)

    extraction_width[0] = extraction_width[1]
    extraction_width[-1] = extraction_width[-2]
    extraction_width = extraction_width.astype(int)

    return extraction_width


def extract(
    img,
    orders,
    column_range=None,
    order_range=None,
    extraction_width=0.5,
    extraction_type="optimal",
    polarization=False,
    shear=None,
    **kwargs
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
    if shear is None:
        shear = np.zeros((nord, ncol))
    if order_range is None:
        order_range = (0, nord - 1)
    if column_range is None:
        column_range = np.tile([0, ncol], (nord, 1))
    if np.isscalar(extraction_width):
        extraction_width = np.tile([extraction_width, extraction_width], (nord, 1))
    scatter = [None for _ in range(nord + 1)]
    xscatter, yscatter = kwargs.get("xscatter"), kwargs.get("yscatter")

    # Limit orders (and related properties) to orders in range
    nord = order_range[1] - order_range[0] + 1
    orders = orders[order_range[0] : order_range[1] + 1]
    column_range = column_range[order_range[0] : order_range[1] + 1]
    extraction_width = extraction_width[order_range[0] : order_range[1] + 1]
    scatter = scatter[order_range[0] : order_range[1] + 2]
    if xscatter is not None:
        xscatter = xscatter[order_range[0] : order_range[1] + 2]
    if yscatter is not None:
        yscatter = yscatter[order_range[0] : order_range[1] + 2]
    shear = shear[order_range[0] : order_range[1] + 1]

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

    if xscatter is not None and yscatter is not None:
        scatter = np.zeros((nord, 4, ncol))
        for onum in range(1, nord + 1):
            if polarization:
                # skip inter-polarization gaps
                oo = ((onum - 1) // 2) * 2 + 1
                scatter[onum - 1, 0] = xscatter[oo - 1]
                scatter[onum - 1, 1] = xscatter[oo + 1]
                scatter[onum - 1, 2] = yscatter[oo - 1]
                scatter[onum - 1, 3] = yscatter[oo + 1]
            else:  # below, above, ybelow, yabove
                scatter[onum - 1, 0] = xscatter[onum - 1]
                scatter[onum - 1, 1] = xscatter[onum]
                scatter[onum - 1, 2] = yscatter[onum - 1]
                scatter[onum - 1, 3] = yscatter[onum]

    if extraction_type == "optimal":
        # the "normal" case, except for wavelength calibration files
        spectrum, slitfunction, uncertainties = optimal_extraction(
            img,
            orders,
            extraction_width,
            column_range,
            scatter=scatter,
            shear=shear,
            **kwargs
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
            scatter=scatter,
            shear=shear,
            normalize=True,
            im_norm=im_norm,
            im_ordr=im_ordr,
            **kwargs
        )
        im_norm[im_norm == 0] = 1
        im_ordr[im_ordr == 0] = 1
        return im_norm, im_ordr, blaze
    elif extraction_type == "arc":
        # Simpler extraction, just summing along the arc of the order
        spectrum, uncertainties = arc_extraction(
            img, orders, extraction_width, column_range, **kwargs
        )

    return spectrum, uncertainties
