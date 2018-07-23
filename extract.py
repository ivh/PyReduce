import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d, median_filter
import astropy.io.fits as fits
import logging
import pickle

from make_scatter import make_scatter
from slitfunc import slitfunc

#from slitfunc_wrapper import slitfunc, slitfunc_curved
from util import make_index


def getflatimg(img, axis=0):
    idx = np.indices(img.shape)[axis]
    return idx.flatten(), img.flat


def getspecvar(img):
    ny, nx = img.shape
    nimg = img / np.nansum(img, axis=1)[:, None]
    x = np.indices(img.shape)[1]
    return x.flatten(), nimg.flat


def getslitvar(img, xoff, osample=1):
    x = np.indices(img.shape)[0]
    x = x - xoff[None, :] + 1
    return x.flatten() * osample, img.flat


def plot_slitfunction(img, spec, slitf, model, ycen, onum, left, right, osample):

    ny, nx = img.shape
    ny_orig = ny
    size = img.size
    ny_os = len(slitf)
    os = (ny_os - 1) / (ny + 1)

    mask = np.ma.getmaskarray(img)
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
    n = np.ma.count_masked(img)
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
            np.linspace(-(1 + offset), di.shape[0]+offset, len(df)), df, "-k", lw=3
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
    if swath_width is None:
        i = np.unique(ycen.astype(int))  # Points of row crossing
        ni = len(i)  # This is how many times this order crosses to the next row
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

def extract_spectrum(
    img,
    ycen,
    ylow,
    yhigh,
    xlow,
    xhigh,
    gain=1,
    readn=0,
    lambda_sf=0.1,
    lambda_sp=0,
    osample=1,
    swath_width=None,
    no_scatter=False,
    telluric=False,
    normalize=False,
    scatter_below=0,
    scatter_above=0,
    yscatter_below=0,
    yscatter_above=0,
    threshold=0,
    use_2d=False,
    plot=False,
    ord_num=0,
    im_norm=None,
    im_ordr=None,
    **kwargs
):

    if use_2d:
        raise NotImplementedError("Curved extraction not supported yet")

    nrow, ncol = img.shape
    nslitf = osample * (ylow + yhigh + 2) + 1
    nysf = yhigh + ylow + 1

    noise = readn / gain

    spec = np.zeros(ncol)
    sunc = np.zeros(ncol)

    no_scatter = no_scatter or (
        np.all(scatter_below == 0) and np.all(scatter_above == 0)
    )

    nbin, bins_start, bins_end = make_bins(swath_width, xlow, xhigh, ycen, ncol)
    slitf = np.zeros((2 * nbin, nslitf))

    # Here is the layout for the bins:
    # Bins overlap roughly half-half, i.e. the second half of bin1 is the same as the first half of bin2
    #
    #        1st swath    3rd swath    5th swath      ...
    #     /============|============|============|============|============|
    #
    #               2nd swath    4th swath    6th swath
    #            |------------|------------|------------|------------|
    #            |.....|
    #            overlap
    #
    #            +     ******* 1
    #             +   *
    #              + *
    #               *            weights (+) previous swath, (*) current swath
    #              * +
    #             *   +
    #            *     +++++++ 0

    # Perform slit decomposition within each swath stepping through the order with
    # half swath width. Spectra for each decomposition are combined with linear weights.
    for ihalf in range(0, 2 * nbin - 1):  # loop through swaths
        ib = bins_start[ihalf]  # left column
        ie = bins_end[ihalf]  # right column
        nc = ie - ib  # number of columns in swath
        logging.debug("Extracting Swath %i, Columns: %i - %i", ihalf, ib, ie)

        # Weight for combining overlapping regions
        weight = np.ones(nc)
        if ihalf != 0:
            weight[: nc // 2 + 1] = np.arange(nc // 2 + 1) / nc * 2

        # Cut out swath from image
        index = make_index(ycen.astype(int) - ylow, ycen.astype(int) + yhigh, ib, ie)
        swath_img = img[index]

        # Telluric
        if telluric:
            tel_lim = (
                telluric if telluric > 5 and telluric < nysf / 2 else min(5, nysf / 3)
            )
            tel = np.sum(swath_img, axis=0)
            itel = np.arange(nysf)
            itel = itel[np.abs(itel - nysf / 2) >= tel_lim]
            tel = swath_img[itel, :]
            sc = np.zeros(nc)

            for itel in range(nc):
                sc[itel] = np.median(tel[itel])

            tell = sc
        else:
            tell = 0

        if not no_scatter:
            # y indices
            index_y = np.array([np.arange(k, nysf + k) for k in yc[ib:ie] + ylow])
            dy_scatter = (index_y.T - yscatter_below[None, ib:ie]) / (
                yscatter_above[None, ib:ie] - yscatter_below[None, ib:ie]
            )
            scatter = (
                scatter_above[None, ib:ie] - scatter_below[None, ib:ie]
            ) * dy_scatter + scatter_below[None, ib:ie]
        else:
            scatter = 0

        # Do Slitfunction extraction
        swath_img -= scatter + tell
        swath_img = np.clip(swath_img, 0, None)

        # offset from the central line
        y_offset = ycen[ib:ie] - ycen.astype(int)[ib:ie]
        swath_spec, swath_slitf, swath_model, swath_unc = slitfunc(
            swath_img, y_offset, lambda_sp=lambda_sp, lambda_sf=lambda_sf, osample=osample
        )

        if normalize:
            # In case we do FF normalization replace the original image by the
            # ratio of sf/sfbin where number of counts is larger than threshold
            # and with 1 elsewhere
            scale = 1

            ii = np.where(swath_model > threshold / gain)
            sss = np.ones((nysf, nc))
            ddd = np.copy(swath_model)
            sss[ii] = swath_img[ii] / swath_model[ii]

            if ihalf != 0:
                overlap = bins_end[ihalf - 1] - bins_start[ihalf] + 1
                sss[ii] /= scale
                swath_spec *= scale
            else:
                nc_old = nc
                sss_old = np.zeros((nysf, nc))
                ddd_old = np.zeros((nysf, nc))
                overlap = bins_start[1] - bins_start[0]

            # Combine new and old sections
            ncc = overlap

            index = make_index(yc - ylow, yc + yhigh, ib, ib + ncc)
            im_norm[index] = (
                sss_old[:, -ncc:] * oweight[:ncc] + sss[:, :ncc] * weight[:ncc]
            )
            im_ordr[index] = (
                ddd_old[:, -ncc:] * oweight[:ncc] + ddd[:, :ncc] * weight[:ncc]
            )

            if ihalf == 2 * nbin - 2:
                # TODO check
                index = make_index(yc - ylow, yc + yhigh, ib + ncc, ib + nc)
                im_norm[index] = sss[:, ncc:nc]
                im_ordr[index] = ddd[:, ncc:nc]

            nc_old = nc
            sss_old = np.copy(sss)
            ddd_old = np.copy(ddd)

        # Combine overlaping regions
        spec[ib:ie] = spec[ib:ie] * (1-weight) + swath_spec * weight
        sunc[ib:ie] = sunc[ib:ie] * (1-weight) + swath_unc * weight
        slitf[ihalf] = swath_slitf

        if plot:
            plot_slitfunction(swath_img, swath_spec, swath_slitf, swath_model, y_offset, ord_num, ib, ie, osample)

    # TODO ????? is that really correct
    sunc = np.sqrt(sunc + spec)

    # TODO what to return ?
    slitf = np.mean(slitf, axis=0)
    swath_model = spec[None, :] * slitf[:, None]
    return spec, slitf, swath_model, sunc


def get_y_scale(order, order_below, order_above, ix, cole, extraction_width, nrow):
    ycen = np.polyval(order, ix)  # row at order center

    left, right = cole  # First column to extract

    ycenn = ycen[left:right]

    ymin = ycenn - extraction_width[0]
    ymin = np.floor(ymin)
    if min(ymin) < 0:
        ymin = ymin - min(ymin)  # help for orders at edge

    ymax = ycenn + extraction_width[1]
    ymax = np.ceil(ymax)
    if max(ymax) > nrow:
        ymax = ymax - max(ymax) + nrow - 1  # helps at edge

    # Define a fixed height area containing one spectral order
    y_lower_lim = int(np.min(ycen[left:right] - ymin))  # Pixels below center line
    y_upper_lim = int(np.min(ymax - ycen[left:right]))  # Pixels above center line

    return y_lower_lim, y_upper_lim


def optimal_extraction(
    img,
    orders,
    extraction_width,
    column_range,
    scatter=None,
    yscatter=None,
    polarization=False,
    **kwargs
):
    logging.info("Using optimal extraction to produce spectrum")

    nrow, ncol = img.shape
    nord = len(orders)

    spectrum = np.zeros((nord, ncol))
    slitfunction = [None for _ in range(nord)]
    uncertainties = np.zeros((nord, ncol))
    ix = np.arange(ncol)

    if scatter is None:
        scatter = np.zeros(nord)
        yscatter = np.zeros(nord)

    # TODO each order is independant so extract in parallel
    for i, onum in enumerate(range(1, nord - 1)):  # loop through orders
        # Background must be subtracted for slit function logic to work but kept
        # as part of the FF signal during normalization

        if polarization:
            # skip inter-polarization gaps
            oo = ((onum - 1) // 2) * 2 + 1
            scatter_below = scatter[oo - 1]
            yscatter_below = yscatter[oo - 1]
            scatter_above = scatter[oo + 1]
            yscatter_above = yscatter[oo + 1]
        else:
            scatter_below = scatter[onum - 1]
            yscatter_below = yscatter[onum - 1]
            scatter_above = scatter[onum]
            yscatter_above = yscatter[onum]

        if nord < 10 or onum % 5 == 0:
            logging.info("Extracting relative order %i out of %i" % (onum, nord - 2))

        # Define a fixed height area containing one spectral order
        x_left_lim, x_right_lim = column_range[onum]  # First and last column to extract

        ycen = np.polyval(orders[onum], ix)
        y_lower_lim, y_upper_lim = get_y_scale(
            orders[onum],
            orders[onum - 1],  # order below
            orders[onum + 1],  # order above
            ix,
            column_range[onum],
            extraction_width[onum],
            nrow,
        )

        spectrum[i], slitfunction[i], _, uncertainties[i] = extract_spectrum(
            img,
            ycen,
            y_lower_lim,
            y_upper_lim,
            x_left_lim,
            x_right_lim,
            scatter_below=scatter_below,
            scatter_above=scatter_above,
            yscatter_below=yscatter_below,
            yscatter_above=yscatter_above,
            ord_num=onum - 1,
            **kwargs
        )

    return spectrum, slitfunction, uncertainties


def arc_extraction(
    img,
    orders,
    extraction_width,
    column_range,
    gain=1,
    readn=0,
    dark=0,
    plot=False,
    **kwargs
):
    logging.info("Using arc extraction to produce spectrum.")
    _, ncol = img.shape
    nord, _ = orders.shape

    if plot:
        # Prepare output image
        output = np.zeros((np.sum(extraction_width[1:-1]) + nord - 2, ncol))
        pos = [0]

    spectrum = np.zeros((nord - 2, ncol))
    uncertainties = np.zeros((nord - 2, ncol))
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
            np.sqrt(np.abs(arc * gain + dark + readn ** 2)) / gain
        )  # estimate uncertainty

        if plot:
            output[pos[i] : pos[i] + index[0].shape[0], x_left_lim:x_right_lim] = img[
                index
            ]
            pos += [pos[i] + index[0].shape[0]]

    if plot:
        plt.imshow(
            output, vmin=0, vmax=np.mean(output) + 5 * np.std(output), origin="lower"
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

    return spectrum, 0, uncertainties


def fix_column_range(img, orders, extraction_width, column_range):
    nrow, ncol = img.shape
    ix = np.arange(ncol)
    # Fix column_range of each order
    for i in range(1, len(orders) - 1):
        order = orders[i]

        coeff_bot, coeff_top = np.copy(order), np.copy(order)
        coeff_bot[-1] -= extraction_width[i, 0]
        coeff_top[-1] += extraction_width[i, 1]

        ixx = ix[column_range[i, 0] : column_range[i, 1]]
        y_bot = np.polyval(coeff_bot, ixx)  # low edge of arc
        y_top = np.polyval(coeff_top, ixx)  # high edge of arc
        # shrink column range so that only valid columns are included, this assumes
        column_range[i] = np.clip(
            column_range[i, 0] + np.where((y_bot > 0) & (y_top < nrow - 1))[0][[0, -1]],
            None,
            column_range[i, 1],
        )

    return column_range


def extend_orders(orders, nrow):
    """ Extrapolate extra orders above and below the existing ones """
    nord, ncoef = orders.shape

    # TODO same as in extract
    if nord > 1:
        order_low = 2 * orders[0] - orders[1]
        order_high = 2 * orders[-1] - orders[-2]
    else:
        order_low = [0 for _ in range(ncoef)]
        order_high = [0 for _ in range(ncoef - 1)] + [nrow]

    orcend = np.array([order_low, *orders, order_high])
    return orcend


def fix_extraction_width(extraction_width, orders, column_range, ncol):
    """ convert fractional extraction width to pixel range """
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
    head,
    orders,
    column_range=None,
    extraction_width=0.5,
    order_range=None,
    thar=False,
    **kwargs
):
    # TODO which parameters should be passed here?

    # Extract relevant header keywords
    kwargs["gain"] = head["e_gain"]
    kwargs["dark"] = head["e_drk"]
    kwargs["readn"] = head["e_readn"]

    normalize = kwargs.get("normalize", False)

    # TODO use curved extraction
    shear = kwargs.get("tilt")
    kwargs["use_2D"] = shear is not None

    nrow, ncol = img.shape
    nord, opower = orders.shape
    if order_range is None:
        order_range = (0, nord - 1)
    if column_range is None:
        column_range = np.tile([0, ncol], (nord, 1))
    if np.isscalar(extraction_width):
        extraction_width = np.tile([extraction_width, extraction_width], (nord, 1))

    # Limit orders (and related properties) to orders in range
    nord = order_range[1] - order_range[0] + 1
    orders = orders[order_range[0] : order_range[1] + 1]
    column_range = column_range[order_range[0] : order_range[1] + 1]
    extraction_width = extraction_width[order_range[0] : order_range[1] + 1]

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

    # TODO
    # Prepare normalized flat field image if necessary
    # These will be passed and "returned" by reference
    # I dont like it, but it works for now
    if normalize:
        im_norm = np.ones_like(img)
        im_ordr = np.ones_like(img)
    else:
        im_norm = im_ordr = None

    if not thar:  # the "normal" case, except for wavelength calibration files
        spectrum, slitfunction, uncertainties = optimal_extraction(
            img,
            orders,
            extraction_width,
            column_range,
            im_norm=im_norm,
            im_ordr=im_ordr,
            **kwargs
        )
    else:
        spectrum, slitfunction, uncertainties = arc_extraction(
            img, orders, extraction_width, column_range, **kwargs
        )

    #TODO remove "extra" orders at boundary
    spectrum = spectrum[:-2]
    uncertainties = uncertainties[:-2]

    if normalize:
        return im_norm, im_ordr, spectrum  # spectrum = blaze

    return spectrum, uncertainties
