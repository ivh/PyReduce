import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d, median_filter

import pickle

from make_scatter import make_scatter
from slitfunc_wrapper import slitfunc, slitfunc_curved
from util import make_index

def getarc(img, orders, onum, awid, x_left_lim=0, x_right_lim=-1):
    """
    # This subroutine extracts a curved arc (arc) from an image array (im). The
    #   curvature of the arc is determined from polynomial fit coefficients (orc)
    #   which usually trace the curvature of the echelle orders. The particular
    #   arc to extract is specified by an order number (onum), which need not be
    #   integral. Positions of nonintegral orders are interpolated from surrounding
    #   orders.
    # im (input array (# columns , # rows)) image from which to extract arc.
    # orc (input array (# of coeff per fit , # of orders)) coefficients from PIT
    #   fit of column number versus row number (of echelle orders, usually). The
    #   polynomials trace arcs (orders, usually) indexed by order number, begining
    #   with zero closest to row zero and increasing as row number increases.
    #   **Note**  These are the extended order coefficients.
    # onum (input scalar) order number of arc to extract - need not be integral.
    # awid (input scalar) full width of arc to be extracted.
    #   Two specifications are possible:
    #     max(awid) <= 1, awid is fraction of the local distance between orders to mash.
    #     max(awid) >  1, awid is the specific number of pixels to mash.
    # arc (output vector (# columns)) counts PER PIXEL in arc extracted from image.
    # [pix (output vector (# columns)] returns the fractional number of pixels
    #   mashed in each column to make arc.
    # 29-Nov-91 GB translated from ANA
    # 22-Dec-91 GB made to return zeros if arc off image
    # 05-Jul-94 JAV, CMJ Moved endelse to extract full arc instead of half when
    #           fraction of an arc is specified
    """

    # ncol, nrow = img.shape
    i1, i2 = x_left_lim, x_right_lim
    # if(keyword_set(x_left_lim )) then i1 = x_left_lim  else i1 = 0
    # if(keyword_set(x_right_lim)) then i2 = x_right_lim else i2 = ncol-1

    # Define useful quantities
    # ncol = len(img[0, i1:i2])  # number of columns
    nrow = len(img[:, i1])  # number of rows
    ix = np.arange(i1, i2, 1, dtype=float)  # vector of column indicies
    arc = np.zeros_like(ix)  # dimension arc vector
    pix = 1.0  # define in case of trouble

    # Interpolate polynomial coefficients for surrounding orders to get polynomial
    # coefficients.  Note that this is mathematically equivalent to interpolating
    # the column indicies for surrounding orders, since the column indicies are
    # linear functions of the polynomial coefficients. However, interpolating
    # coefficients should be faster.

    if np.max(awid) < 1:  # awid is an order fraction
        ob = onum - awid / 2  # order # of bottom edge of arc
        obi = int(ob)  # next lowest integral order #A
        cb = orders[obi] + (ob - obi) * (orders[obi + 1] - orders[obi])
        yb = np.polyval(cb, ix)  # row # of bottom edge of swath

        ot = onum + awid / 2  # order # of top edge of arc
        oti = int(ot)  # next lowest integral order #
        ct = orders[oti] + (ot - oti) * (orders[oti + 1] - orders[oti])
        yt = np.polyval(ct, ix)  # row # of top edge of swath

    else:  # awid is number of pixels
        c = np.copy(orders[onum])
        yb = np.polyval(c, ix) - awid / 2  # row # of bottom edge of swath
        yt = np.polyval(c, ix) + awid / 2  # row # of top edge of swath

    if np.min(yb) < 0:  # check if arc is off bottom
        raise Exception(
            "FORDS: Warning - requested arc is below bottom of image. %i" % onum
        )
    if np.max(yt) > nrow:  # check if arc is off top of im
        raise Exception(
            "FORDS: Warning - requested arc is above top of image. %i" % onum
        )

    diff = np.round(np.mean(yt - yb) * 0.5)
    yb = np.clip(np.round((yb + yt) * 0.5 - diff), 0, None).astype(int)
    yt = np.clip(np.round((yb + yt) * 0.5 + diff), None, nrow).astype(int)

    # Define the indices for the pixels
    # in x: the rows between yb and yt
    # in y: the column, but n times to match the x index
    index = make_index(yb, yt, i1, i2, zero=True)
    # Sum over the prepared index
    arc = np.sum(img[index], axis=0)

    return arc, pix


def make_slitfunction(
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

    nrow, ncol = img.shape
    noise = readn / gain
    irow = np.arange(nrow)
    ycene = ycen[xlow:xhigh]
    nysf = yhigh + ylow + 1
    yslitf = -ylow, yhigh

    spec = np.zeros(ncol)
    sunc = np.zeros(ncol)

    no_scatter = no_scatter or (
        np.all(scatter_below == 0) and np.all(scatter_above == 0)
    )

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

    nslitf = osample * (ylow + yhigh + 2) + 1
    slitf = np.zeros((2 * nbin, nslitf))

    # Define order boundary
    yc = ycen.astype(int)
    ymin = ycen - ylow
    ymax = ycen + yhigh

    # Calculate boundaries of distinct slitf regions
    # Here is the layout to understand the lines below
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

    bins = np.linspace(xlow, xhigh, 2 * nbin + 1)  # boundaries of bins
    ibeg_half = np.ceil(bins[:-2]).astype(int)  # beginning of each bin
    iend_half = np.floor(bins[2:]).astype(int)  # end of each bin
    bincen = 0.5 * (ibeg_half + iend_half)  # center of each bin

    # Perform slit decomposition within each swath stepping through the order with
    # half swath width. Spectra for each decomposition are combined with linear weights.
    for ihalf in range(0, 2 * nbin - 1):  # loop through swaths
        ib = ibeg_half[ihalf]  # left column
        ie = iend_half[ihalf] + 1  # right column
        nc = ie - ib  # number of columns in swath

        # Weight for combining overlapping regions
        weight = np.ones(nc)
        if ihalf > 0:
            weight[: nc // 2 + 1] = np.arange(nc // 2 + 1) / nc * 2
        oweight = 1 - weight

        # Cut out swath from image
        index = make_index(yc - ylow, yc + yhigh, ib, ie)
        sf = img[index]

        # Telluric
        if telluric:
            tel_lim = (
                telluric if telluric > 5 and telluric < nysf / 2 else min(5, nysf / 3)
            )
            tel = np.sum(sf, axis=0)
            itel = np.arange(nysf)
            itel = itel[np.abs(itel - nysf / 2) >= tel_lim]
            tel = sf[itel, :]
            sc = np.zeros(nc)

            for itel in range(nc):
                sc[itel] = np.median(tel[itel])

            tell = sc
        else:
            tell = 0

        if not no_scatter:
            # y indices
            index_y = np.array([np.arange(k, nysf + k) for k in yc[ib:ie] - yslitf[0]])
            dy_scatter = (index_y.T - yscatter_below[None, ib:ie]) / (
                yscatter_above[None, ib:ie] - yscatter_below[None, ib:ie]
            )
            scatter = (
                scatter_above[None, ib:ie] - scatter_below[None, ib:ie]
            ) * dy_scatter + scatter_below[None, ib:ie]
        else:
            scatter = 0

        # Do Slitfunction extraction
        sf -= scatter + tell
        sf = np.clip(sf, 0, None)

        # offset from the central line
        y_offset = ycen[ib:ie] - yc[ib:ie]
        if use_2d:
            sp, sfsm, model, unc = slitfunc_curved(
                sf,
                y_offset,
                tilt,
                lambda_sp=lambda_sp,
                lambda_sl=lambda_sf,
                osample=osample,
            )
            delta_x = None  # TODO get this from slitfunc_curved
        else:
            sf = np.copy(sf)
            y_offset = np.copy(y_offset)
            sp, sfsm, model, unc = slitfunc(
                sf, y_offset, lambda_sp=lambda_sp, lambda_sf=lambda_sf, osample=osample
            )

        if normalize:
            # In case we do FF normalization replace the original image by the
            # ratio of sf/sfbin where number of counts is larger than threshold
            # and with 1 elsewhere
            scale = 1

            ii = np.where(model > threshold / gain)
            sss = np.ones((nysf, nc))
            ddd = np.copy(model)
            sss[ii] = sf[ii] / model[ii]

            if ihalf > 0:
                overlap = iend_half[ihalf - 1] - ibeg_half[ihalf] + 1
                sss[ii] /= scale
                sp *= scale
            else:
                nc_old = nc
                sss_old = np.zeros((nysf, nc))
                ddd_old = np.zeros((nysf, nc))
                overlap = ibeg_half[1] - ibeg_half[0]

            # Combine new and old sections
            ncc = overlap

            index = make_index(yc + yslitf[0], yc + yslitf[1], ib, ib + ncc)
            im_norm[index] = (
                sss_old[:, -ncc:] * oweight[:ncc] + sss[:, :ncc] * weight[:ncc]
            )
            im_ordr[index] = (
                ddd_old[:, -ncc:] * oweight[:ncc] + ddd[:, :ncc] * weight[:ncc]
            )

            if ihalf == 2 * nbin - 2:
                # TODO check
                index = make_index(yc + yslitf[0], yc + yslitf[1], ib + ncc, ib + nc)
                im_norm[index] = sss[:, ncc:nc]
                im_ordr[index] = ddd[:, ncc:nc]

            nc_old = nc
            sss_old = np.copy(sss)
            ddd_old = np.copy(ddd)

        # Combine overlaping regions
        if use_2d:
            if ihalf > 0 and ihalf < 2 * nbin - 2:
                spec[ib + delta_x : ie - delta_x] = (
                    spec[ib + delta_x : ie - delta_x] * oweight[delta_x : nc - delta_x]
                    + sp[delta_x:-delta_x] * weight[delta_x:-delta_x]
                )
            elif ihalf == 0:
                spec[ib : ie - delta_x] = sp[:-delta_x]
            elif ihalf == 2 * nbin - 2:
                spec[ib + delta_x : ie] = (
                    spec[ib + delta_x : ie] * oweight[delta_x:-1]
                    + sp[delta_x:-1] * weight[delta_x:-1]
                )
        else:
            spec[ib:ie] = spec[ib:ie] * oweight + sp * weight

        sunc[ib:ie] = sunc[ib:ie] * oweight + unc * weight

        if plot:
            # TODO make this nice
            plt.clf()
            scale = 1
            pscale = np.mean(sp)
            sfplot = gaussian_filter1d(sfsm, osample)
            sfflat = sfsm[:-2] * pscale
            model = np.mean(model, axis=1)
            if not no_scatter:
                poffset = np.mean(scatter_below[ib:ie] + scatter_above[ib:ie]) * 0.5
            else:
                poffset = 0

            # Plot 1: The observed slit
            plt.subplot(221)
            plt.title("Order %i, Columns %i through %i" % (ord_num, ib, ie))
            #plt.plot(sfflat)
            # plt.plot(ysfpnt[jbad], sfpnt[jbad] * pscale, "g+")
            plt.plot(sfflat, '+')
            plt.plot(model)

            # Plot 2: The recovered slit function
            plt.subplot(222)
            plt.title("Order %i, Columns %i through %i" % (ord_num, ib, ie))
            plt.plot(model)
            # plt.plot(ysfpnt[jbad], sfpnt[jbad] * pscale, "g+")
            #plt.plot(sfplot)

            # Plot 3: Difference between observed and recovered
            plt.subplot(223)
            plt.title("Data - Fit")
            plt.plot(sfflat - model)
            # plt.plot(ysfpnt[jbad], (sfpnt - sfsm2)[jbad] * pscale, "g+")
            plt.plot(np.sqrt((model + poffset + readn ** 2) / gain))
            plt.plot(-np.sqrt((model + poffset + readn ** 2) / gain))

            plt.subplot(224)
            plt.title("Data - Fit")
            plt.plot(sfflat - model)
            # plt.plot(ysfpnt[jbad], (sf.flat - sfsm.flat)[jbad] * pscale, "g+")
            plt.plot(np.sqrt((model + poffset + readn ** 2) / gain))
            plt.plot(-np.sqrt((model + poffset + readn ** 2) / gain))

            plt.draw()
            plt.pause(0.001)

    # TODO ????? is that really correct
    sunc = np.sqrt(sunc + spec)

    # TODO what to return ?
    slitf = np.mean(slitf, axis=0)
    model = spec[:, None] * slitf[None, :]
    return spec, slitf, model, sunc


def get_y_scale(order, order_below, order_above, ix, cole, xwd, nrow):
    ycen = np.polyval(order, ix)  # row at order center

    x_left_lim = cole[0]  # First column to extract
    x_right_lim = cole[1]  # Last column to extract

    ixx = ix[x_left_lim:x_right_lim]
    ycenn = ycen[x_left_lim:x_right_lim]
    if xwd[0] > 1.5:  # Extraction width in pixels
        ymin = ycenn - xwd[0]
    else:  # Fractional extraction width
        ymin = ycenn - xwd[0] * (ycenn - np.polyval(order_below, ixx))  # trough below

    ymin = np.floor(ymin)
    if min(ymin) < 0:
        ymin = ymin - min(ymin)  # help for orders at edge
    if xwd[1] > 1.5:  # Extraction width in pixels
        ymax = ycenn + xwd[1]
    else:  # Fractional extraction width
        ymax = ycenn + xwd[1] * (np.polyval(order_above, ixx) - ycenn)  # trough above

    ymax = np.ceil(ymax)
    if max(ymax) > nrow:
        ymax = ymax - max(ymax) + nrow - 1  # helps at edge

    # Define a fixed height area containing one spectral order
    y_lower_lim = int(min(ycen[cole[0] : cole[1]] - ymin))  # Pixels below center line
    y_upper_lim = int(min(ymax - ycen[cole[0] : cole[1]]))  # Pixels above center line

    return y_lower_lim, y_upper_lim


def optimal_extraction(
    img,
    orders,
    xwd,
    column_range,
    scatter=None,
    yscatter=None,
    polarization=False,
    **kwargs
):
    print("GETSPEC: Using optimal extraction to produce spectrum.")
    # xwd_boundaries = np.zeros((2, 1), dtype=int)

    nrow, ncol = img.shape
    nord = len(orders)

    spectrum = np.zeros((nord, ncol))
    slitfunction = [None for _ in range(nord)]
    uncertainties = np.zeros((nord, ncol))
    ix = np.arange(ncol)

    if scatter is None:
        scatter = np.zeros(nord)
        yscatter = np.zeros(nord)

    for i, onum in enumerate(range(1, nord - 1)):  # loop through orders
        # Background must be subtracted for slit function logic to work but kept
        # as part of the FF signal during normalization

        if polarization:
            # skip inter-polarization gaps
            oo = ((onum - 1) // 2) * 2 + 1
            scatter_below = scatter[oo - 1]
            yscatter_below = yscatter[oo - 1]
            scatter_above = scatter[oo]
            yscatter_above = yscatter[oo]
        else:
            scatter_below = scatter[onum - 1]
            yscatter_below = yscatter[onum - 1]
            scatter_above = scatter[onum]
            yscatter_above = yscatter[onum]

        if nord <= 10:
            print("GETSPEC: extracting relative order %i out of %i" % (onum, nord))
        else:
            if (onum - 1) % 5 == 0:
                print(
                    "GETSPEC: extracting relative orders %i-%i out of %i"
                    % (onum, np.clip(nord, None, onum + 4), nord)
                )

        # Define a fixed height area containing one spectral order
        x_left_lim, x_right_lim = column_range[onum]  # First and last column to extract

        ycen = np.polyval(orders[onum], ix)
        y_lower_lim, y_upper_lim = get_y_scale(
            orders[onum],
            orders[onum - 1],  # order below
            orders[onum + 1],  # order above
            ix,
            column_range[onum],
            xwd[onum],
            nrow,
        )

        spectrum[i], slitfunction[i], _, uncertainties[i] = make_slitfunction(
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
            **kwargs
        )

    return spectrum, slitfunction, uncertainties


def arc_extraction(img, orders, xwd, column_range, gain=1, readn=0, dark=0, **kwargs):
    print("Using arc extraction to produce spectrum.")
    _, ncol = img.shape
    nord = len(orders)

    spectrum = np.zeros((nord - 2, ncol))
    uncertainties = np.zeros((nord - 2, ncol))

    for i, onum in enumerate(range(1, nord - 1)):  # loop thru orders
        x_left_lim = column_range[onum, 0]  # First column to extract
        x_right_lim = column_range[onum, 1]  # Last column to extract
        awid = xwd[onum, 0] + xwd[onum, 1]

        arc, pix = getarc(
            img, orders, onum, awid, x_left_lim=x_left_lim, x_right_lim=x_right_lim
        )  # extract counts/pixel

        spectrum[i, x_left_lim:x_right_lim] = arc * pix  # store total counts
        uncertainties[i, x_left_lim:x_right_lim] = (
            np.sqrt(abs(arc * pix * gain + dark + pix * readn ** 2)) / gain
        )  # estimate uncertainty

    return spectrum, 0, uncertainties


def fix_column_range(img, orders, xwd, column_range):
    nrow, ncol = img.shape
    ix = np.arange(ncol)
    # Fix column_range of each order
    for i in range(1, len(orders) - 1):
        order = orders[i]
        if xwd[i, 0] > 1.5:  # Extraction width in pixels
            coeff_bot, coeff_top = np.copy(order), np.copy(order)
            coeff_bot[-1] -= xwd[i, 0]
            coeff_top[-1] += xwd[i, 1]
        else:  # Fraction extraction width
            coeff_bot = 0.5 * ((2 + xwd[i, 0]) * order - xwd[i, 0] * orders[i + 1])
            coeff_top = 0.5 * ((2 + xwd[i, 1]) * order - xwd[i, 1] * orders[i - 1])

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


def extract(
    img,
    head,
    orders,
    column_range=None,
    xwd=0.5,
    order_range=None,
    scatter=False,
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
        column_range = np.tile([0, ncol], (nord, 0))
    if np.isscalar(xwd):
        xwd = np.tile([xwd, xwd], (nord, 1))

    # Limit orders (and related properties) to orders in range
    nord = order_range[1] - order_range[0] + 1
    orders = orders[order_range[0] : order_range[1] + 1]
    column_range = column_range[order_range[0] : order_range[1] + 1]
    xwd = xwd[order_range[0] : order_range[1] + 1]

    # Extend orders and related properties
    orders = extend_orders(orders, nrow)
    xwd = np.array([xwd[0], *xwd, xwd[-1]])
    column_range = np.array([column_range[0], *column_range, column_range[-1]])

    # Fix column range
    column_range = fix_column_range(img, orders, xwd, column_range)

    # Calculate background scatter if necessary
    if scatter:
        # TODO DEBUG
        try:
            _f = open("background.dat", "rb")
            scatter, yscatter = pickle.load(_f)
        except FileNotFoundError:
            scatter, yscatter = make_scatter(
            img, orders, column_range=column_range, subtract=True, **kwargs
            )
            _f = open("background.dat", "wb")
            pickle.dump((bg, ybg), _f)
    else:
        scatter = yscatter = None

    # Prepare normalized flat field image if necessary
    if normalize:
        im_norm = np.ones_like(img)
        im_ordr = np.ones_like(img)
    else:
        im_norm = im_ordr = None

    if not thar:  # the "normal" case, except for wavelength calibration files
        spectrum, slitfunction, uncertainties = optimal_extraction(
            img,
            orders,
            xwd,
            column_range,
            scatter=scatter,
            yscatter=yscatter,
            im_norm=im_norm,
            im_ordr=im_ordr,
            **kwargs
        )
    else:
        spectrum, slitfunction, uncertainties = arc_extraction(
            img, orders, xwd, column_range, **kwargs
        )

    if normalize:
        return im_norm, im_ordr, spectrum  # spectrum = blaze

    return spectrum, uncertainties
