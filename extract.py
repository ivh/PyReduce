import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter, gaussian_filter1d
from scipy.interpolate import interp1d

# TODO DEBUG
import clib.build_extract

clib.build_extract.build()

from util import make_index

import clib._slitfunc_bd.lib as slitfunclib
import clib._slitfunc_2d.lib as slitfunc_2dlib
from clib._cluster import ffi

c_double = np.ctypeslib.ctypes.c_double
c_int = np.ctypeslib.ctypes.c_int

# plt.ion()  # TODO DEBUG


def slitfunc(img, ycen, lambda_sp=0, lambda_sl=0.1, osample=1):
    """Decompose image into spectrum and slitfunction

    This is for vertical(?) orders only, for curved orders use slitfunc_curved instead

    Parameters
    ----------
    img : array[n, m]
        image to decompose, should just contain a small part of the overall image
    ycen : array[n]
        traces the center of the order along the image, relative to the center of the image?
    lambda_sp : float, optional
        smoothing parameter of the spectrum (the default is 0, which no smoothing)
    lambda_sl : float, optional
        smoothing parameter of the slitfunction (the default is 0.1, which )
    osample : int, optional
        Subpixel ovsersampling factor (the default is 1, which no oversampling)

    Returns
    -------
    sp, sl, model, unc
        spectrum, slitfunction, model, spectrum uncertainties
    """

    if osample != 1:
        print("WARNING: Oversampling may be wrong !!!")

    # Get dimensions
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    # Inital guess for slit function and spectrum
    # Just sum up the image along one side and normalize
    sl = np.sum(img, axis=1)
    sl = sl / np.sum(sl)  # slit function

    sp = np.sum(img, axis=0)
    sp = sp / np.sum(sp) * np.sum(img)
    if lambda_sp != 0:
        sp = gaussian_filter1d(sp, lambda_sp)

    sp = np.ascontiguousarray(sp[::-1])  # TODO why?

    # Stretch sl by oversampling factor
    old_points = np.linspace(0, (nrows + 1) * osample, nrows, endpoint=True)
    sl = interp1d(old_points, sl, kind=2)(np.arange(ny))

    if hasattr(img, "mask"):
        mask = (~img.mask).astype(c_int).flatten()
        mask = np.ascontiguousarray(mask)
        cmask = ffi.cast("int *", mask.ctypes.data)
        # img = img.data
        # sp = sp.data
    else:
        mask = np.ones(nrows * ncols, dtype=c_int)
        cmask = ffi.cast("int *", mask.ctypes.data)

    sl = sl.astype(c_double)
    csl = ffi.cast("double *", sl.ctypes.data)

    sp = sp.astype(c_double)
    csp = ffi.cast("double *", sp.ctypes.data)

    img = img.flatten().astype(c_double)
    img = np.ascontiguousarray(img)
    cimg = ffi.cast("double *", img.ctypes.data)

    ycen = ycen.astype(c_double)
    ycen = np.ascontiguousarray(ycen)
    cycen = ffi.cast("double *", ycen.ctypes.data)

    model = np.zeros((nrows, ncols), dtype=c_double)
    cmodel = ffi.cast("double *", model.ctypes.data)

    unc = np.zeros(ncols, dtype=c_double)
    cunc = ffi.cast("double *", unc.ctypes.data)

    slitfunclib.slit_func_vert(
        ncols,
        nrows,
        cimg,
        cmask,
        cycen,
        osample,
        lambda_sp,
        lambda_sl,
        csp,
        csl,
        cmodel,
        cunc,
    )

    return sp, sl, model, unc


def slitfunc_curved(img, ycen, shear, osample=1, lambda_sp=0, lambda_sl=0.1):
    """Decompose an image into a spectrum and a slitfunction, image may be curved

    Parameters
    ----------
    img : array[n, m]
        input image
    ycen : array[n]
        traces the center of the order
    shear : array[n]
        tilt of the order along the image ???, set to 0 if order straight
    osample : int, optional
        Subpixel ovsersampling factor (the default is 1, which no oversampling)
    lambda_sp : float, optional
        smoothing factor spectrum (the default is 0, which no smoothing)
    lambda_sl : float, optional
        smoothing factor slitfunction (the default is 0.1, which small)

    Returns
    -------
    sp, sl, model, unc
        spectrum, slitfunction, model, spectrum uncertainties
    """

    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    y_lower_lim = int(min(ycen))  # TODO

    # Inital guess for slit function and spectrum
    # Just sum up the image along one side and normalize
    # TODO: rotate image before summation?
    sl = np.sum(img, axis=1)
    sl = sl / np.sum(sl)  # slit function

    sp = np.sum(img * sl[:, None], axis=0)
    sp = sp / np.sum(sp) * np.sum(img)

    # Stretch sl by oversampling factor
    old_points = np.linspace(0, (nrows + 1) * osample, nrows, endpoint=True)
    sl = interp1d(old_points, sl, kind=2)(np.arange(ny))

    sl = sl.astype(c_double)
    csl = ffi.cast("double *", sl.ctypes.data)

    sp = sp.astype(c_double)
    csp = ffi.cast("double *", sp.ctypes.data)

    if np.ma.is_masked(img):
        mask = (~img.mask).astype(c_int).flatten()
        mask = np.ascontiguousarray(mask)
        cmask = ffi.cast("int *", mask.ctypes.data)
    else:
        mask = np.ones(nrows * ncols, dtype=c_int)
        cmask = ffi.cast("int *", mask.ctypes.data)

    img = img.flatten().astype(c_double)
    img = np.ascontiguousarray(img)
    cimg = ffi.cast("double *", img.ctypes.data)

    ycen = ycen.astype(c_double)
    cycen = ffi.cast("double *", ycen.ctypes.data)

    ycen_offset = np.copy(ycen).astype(c_int)
    cycen_offset = ffi.cast("int *", ycen_offset.ctypes.data)

    shear = shear.astype(c_double)
    cshear = ffi.cast("double *", shear.ctypes.data)

    model = np.zeros((nrows, ncols), dtype=c_double)
    cmodel = ffi.cast("double *", model.ctypes.data)

    unc = np.zeros(ncols, dtype=c_double)
    cunc = ffi.cast("double *", unc.ctypes.data)

    slitfunc_2dlib.slit_func_curved(
        ncols,
        nrows,
        cimg,
        cmask,
        cycen,
        cycen_offset,
        cshear,
        y_lower_lim,
        osample,
        lambda_sp,
        lambda_sl,
        csp,
        csl,
        cmodel,
        cunc,
    )

    return sp, sl, model, unc


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
    index = make_index(yb, yt, 0, i2 - i1)
    index = (index[0], index[1] + i1)
    # Sum over the prepared index
    arc = np.sum(img[index], axis=0)

    # for col in range(nrow):  # sum image in requested arc
    #    scol = img[yb[col] : yt[col], col + i1]
    #    arc[col] = np.sum(scol)

    # Define vectors along edge of swath.
    # vb = img[ix + ncol * ybi]  # bottommost pixels in swath
    # vt = img[ix + ncol * yti]  # topmost pixels in swath

    return arc, pix


def mkslitf(
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
    **kwargs
):

    nrow, ncol = img.shape
    noise = readn / gain
    irow = np.arange(nrow)
    ycene = ycen[xlow:xhigh]
    nysf = yhigh + ylow + 1
    yslitf0, yslitf1 = -ylow, yhigh
    jbad = []

    spec = np.zeros(ncol)
    sunc = np.zeros(ncol)

    no_scatter = no_scatter or (
        np.all(scatter_below == 0) and np.all(scatter_above == 0)
    )

    if normalize:
        im_norm = np.zeros_like(img)
        im_ordr = np.zeros_like(img)

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
        nbin = np.clip(np.round((xhigh - xlow) / swath_width), 1, None)

    nslitf = osample * (ylow + yhigh + 2) + 1
    yslitf = yslitf0 + (np.arange(nslitf) - 0.5) / osample - 1.5
    slitf = np.zeros((2 * nbin, nslitf))

    # Define order boundary
    yc = ycen.astype(int)
    ymin = ycen - ylow
    ymax = ycen + yhigh

    # Calculate boundaries if distinct slitf regions
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

        # load data
        nsf = nc * nysf  # number of slitfunc points
        j0 = np.zeros(nc, dtype=int)
        j1 = np.zeros(nc, dtype=int)

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
            index_y = np.array([np.arange(k, nysf + k) for k in yc[ib:ie] - yslitf0])
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
        sfpnt = sf.flatten()  # ?
        ysfpnt = (irow[:, None] - ycen[None, :])[index].flatten()  # ?

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
            sp, sfsm, model, unc = slitfunc(
                sf, y_offset, lambda_sp=lambda_sp, lambda_sl=lambda_sf, osample=osample
            )

        # Combine overlapping regions
        weight = np.ones(nc)
        if ihalf > 0:
            weight[: nc // 2 + 1] = np.arange(nc // 2 + 1) / nc * 2
        oweight = 1 - weight

        # In case we do FF normalization replace the original image by the
        # ratio of sf/sfbin where number of counts is larger than threshold
        # and with 1 elsewhere

        scale = 1
        if normalize:
            ii = np.where(model > threshold / gain)
            sss = np.ones((nysf, nc))
            ddd = np.zeros((nysf, nc))
            sss[ii] = sf[ii] / model[ii]
            ddd = np.copy(model)

            if ihalf > 0:
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

                overlap = iend_half[ihalf - 1] - ibeg_half[ihalf] + 1
                sss[ii] /= scale
                sp *= scale
            else:
                nc_old = nc
                sss_old = np.zeros((nysf, nc))
                ddd_old = np.zeros((nysf, nc))
                overlap = nc_old + 1

            # This loop is complicated because swaths do overlap to ensure continuity of the spectrum.
            ncc = overlap if ihalf != 0 else ibeg_half[1] - ibeg_half[0]

            for j in range(ncc):
                # TODO same as above?
                icen = yc[ib + j]
                k0 = icen + yslitf0
                k1 = icen + yslitf1 + 1
                j0[j] = j * nysf
                j1[j] = j0[j] + k1 - k0
                jj = nc_old - ncc + j
                im_norm[k0:k1, ib + j] = (
                    sss_old[:, jj] * oweight[j] + sss[:, j] * weight[j]
                )
                im_ordr[k0:k1, ib + j] = (
                    ddd_old[:, jj] * oweight[j] + ddd[:, j] * weight[j]
                )

            if ihalf == 2 * nbin - 2:
                for j in range(ncc, nc):
                    # TODO same as above
                    icen = yc[ib + j]
                    k0 = icen + yslitf0
                    k1 = icen + yslitf1 + 1
                    j0[j] = j * nysf
                    j1[j] = j0[j] + k1 - k0
                    jj = nc_old - ncc + j

                    im_norm[k0:k1, ib + j] = sss[:, j]
                    im_ordr[k0:k1, ib + j] = ddd[:, j]

            nc_old = nc
            sss_old = np.copy(sss)
            ddd_old = np.copy(ddd)

        nbad = len(jbad)
        if nbad == 1 and jbad == 0:
            nbad = 0

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

        sfsm2 = model.T.flatten()
        j = np.argsort(ysfpnt)

        jbad = np.argsort(j)[jbad]
        ysfpnt = ysfpnt[j]
        sfpnt = sfpnt[j]
        sfsm2 = sfsm2[j]

        slitf[ihalf, :] = sfsm / np.sum(sfsm) * osample

        if plot:
            # TODO make this nice
            pscale = np.mean(sp)
            sfplot = gaussian_filter1d(sfsm * pscale, osample)
            if not no_scatter:
                poffset = np.mean(scatter_below[ib:ie] + scatter_above[ib:ie]) * 0.5
            else:
                poffset = 0

            plt.subplot(221)
            plt.title("Order %i, Columns %i through %i" % (ord_num, ib, ie))
            plt.plot(ysfpnt, sfpnt * pscale)
            plt.plot(ysfpnt[jbad], sfpnt[jbad] * pscale, "g+")
            plt.plot(yslitf, sfplot)

            plt.subplot(222)
            plt.title("Order %i, Columns %i through %i" % (ord_num, ib, ie))
            plt.plot(ysfpnt, sfpnt * pscale)
            plt.plot(ysfpnt[jbad], sfpnt[jbad] * pscale, "g+")
            plt.plot(yslitf, sfplot)

            plt.subplot(223)
            plt.title("Data - Fit")
            plt.plot(ysfpnt, (sfpnt - sfsm2) * pscale)
            plt.plot(ysfpnt[jbad], (sfpnt - sfsm2)[jbad] * pscale, "g+")
            plt.plot(
                yslitf, np.sqrt((sfsm * pscale / scale + poffset + readn ** 2) / gain)
            )
            plt.plot(
                yslitf, -np.sqrt((sfsm * pscale / scale + poffset + readn ** 2) / gain)
            )

            plt.subplot(224)
            plt.title("Data - Fit")
            plt.plot(ysfpnt, (sfpnt - sfsm2) * pscale)
            plt.plot(ysfpnt[jbad], (sfpnt - sfsm2)[jbad] * pscale, "g+")
            plt.plot(
                yslitf, np.sqrt((sfsm * pscale / scale + poffset + readn ** 2) / gain)
            )
            plt.plot(
                yslitf, -np.sqrt((sfsm * pscale / scale + poffset + readn ** 2) / gain)
            )

            plt.show()

    #TODO ????? is that really correct
    sunc = np.sqrt(sunc + spec)

    # TODO what to return ?
    slitf = np.mean(slitf, axis=0)
    model = spec[:, None] * slitf[None, :]
    return spec, slitf, model, sunc


def optimal_extraction(
    img,
    head,
    orders,
    xwd,
    column_range,
    order_range,
    scatter=None,
    yscatter=None,
    **kwargs
):
    print("GETSPEC: Using optimal extraction to produce spectrum.")
    ofirst, olast = order_range
    # xwd_boundaries = np.zeros((2, 1), dtype=int)

    nrow, ncol = img.shape
    n_ord = len(orders)

    spectrum = np.zeros((n_ord, ncol))
    slitfunction = [None for _ in range(n_ord)]
    uncertainties = np.zeros((n_ord, ncol))
    ix = np.arange(ncol)

    if scatter is None:
        scatter = np.zeros(n_ord)
        yscatter = np.zeros(n_ord)

    for i, onum in enumerate(range(ofirst, olast + 1)):  # loop thru orders
        # ncole = (
        #    column_range[onum, 1] - column_range[onum, 0] + 1
        # )  # number of columns to extract
        cole0 = column_range[onum, 0]  # first column to extract
        cole1 = column_range[onum, 1]  # last column to extract

        # Background must be subtracted for slit function logic to work but kept
        # as part of the FF signal during normalization

        scatter_below = scatter[onum - 1]
        yscatter_below = yscatter[onum - 1]
        scatter_above = scatter[onum]
        yscatter_above = yscatter[onum]

        if n_ord <= 10:
            print("GETSPEC: extracting relative order %i out of %i" % (onum, n_ord))
        else:
            if (onum - 1) % 5 == 0:
                print(
                    "GETSPEC: extracting relative orders %i-%i out of %i"
                    % (onum, np.clip(n_ord, None, onum + 4), n_ord)
                )

        ycen = np.polyval(orders[onum], ix)  # row at order center

        x_left_lim = cole0  # First column to extract
        x_right_lim = cole1  # Last column to extract

        ixx = ix[x_left_lim:x_right_lim]
        ycenn = ycen[x_left_lim:x_right_lim]
        if xwd[onum, 0] > 1.5:  # Extraction width in pixels
            ymin = ycenn - xwd[onum, 0]
        else:  # Fractional extraction width
            ymin = ycenn - xwd[onum, 0] * (
                ycenn - np.polyval(orders[onum - 1], ixx)
            )  # trough below

        ymin = np.floor(ymin)
        if min(ymin) < 0:
            ymin = ymin - min(ymin)  # help for orders at edge
        if xwd[onum, 1] > 1.5:  # Extraction width in pixels
            ymax = ycenn + xwd[onum, 1]
        else:  # Fractional extraction width
            ymax = ycenn + xwd[onum, 1] * (
                np.polyval(orders[onum - 1], ixx) - ycenn
            )  # trough above

        ymax = np.ceil(ymax)
        if max(ymax) > nrow:
            ymax = ymax - max(ymax) + nrow - 1  # helps at edge

        # Define a fixed height area containing one spectral order
        y_lower_lim = int(min(ycen[cole0:cole1] - ymin))  # Pixels below center line
        y_upper_lim = int(min(ymax - ycen[cole0:cole1]))  # Pixels above center line

        spectrum[i], slitfunction[i], _, uncertainties[i] = mkslitf(
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


def arc_extraction(img, head, orders, xwd, column_range, **kwargs):
    print("Using arc extraction to produce spectrum.")
    _, ncol = img.shape
    n_ord = len(orders)

    gain = head["e_gain"]
    dark = head["e_drk"]
    readn = head["e_readn"]

    spectrum = np.zeros((n_ord - 2, ncol))
    uncertainties = np.zeros((n_ord - 2, ncol))

    for i, onum in enumerate(range(1, n_ord - 1)):  # loop thru orders
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


def fix_column_range(img, orders, order_range, xwd, column_range):
    nrow, ncol = img.shape
    ix = np.arange(ncol)
    # Fix column_range of each order
    for order in range(order_range[0], order_range[1] + 1):
        if xwd[0, 0] > 1.5:  # Extraction width in pixels
            coeff_bot, coeff_top = np.copy(orders[order]), np.copy(orders[order])
            coeff_bot[-1] -= xwd[order, 0]
            coeff_top[-1] += xwd[order, 1]
        else:  # Fraction extraction width
            coeff_bot = 0.5 * (
                (2 + xwd[order, 0]) * orders[order] - xwd[order, 0] * orders[order + 1]
            )
            coeff_top = 0.5 * (
                (2 + xwd[order, 1]) * orders[order] - xwd[order, 1] * orders[order - 1]
            )

        ixx = ix[column_range[order][0] : column_range[order][1]]
        y_bot = np.polyval(coeff_bot, ixx)  # low edge of arc
        y_top = np.polyval(coeff_top, ixx)  # high edge of arc
        # shrink column range so that only valid columns are included, this assumes
        column_range[order] = np.clip(
            column_range[order][0] + np.where((y_bot > 0) & (y_top < nrow))[0][[0, -1]],
            None,
            column_range[order][1],
        )

    return column_range


def extract(img, head, orders, **kwargs):
    # TODO which parameters should be passed here?
    xwd = kwargs.get("xwd", 50)
    column_range = kwargs.get(
        "column_range", np.array([(0, img.shape[1]) for _ in orders])
    )

    # TODO use curved extraction
    shear = kwargs.get("tilt")

    nrow, ncol = img.shape
    n_ord = len(orders)
    order_range = kwargs.get("order_range", (0, n_ord - 1))
    n_ord = len(order_range)

    # Extrapolate extra orders above and below the existing ones
    if n_ord > 1:
        order_low = 2 * orders[order_range[0]] - orders[order_range[0] + 1]
        order_high = 2 * orders[order_range[1]] - orders[order_range[1] - 1]
    else:
        opower = len(orders[order_range[0]])
        order_low = [0 for _ in range(opower)]
        order_high = [0 for _ in range(opower - 1)] + [nrow]

    column_range = fix_column_range(img, orders, order_range, xwd, column_range)

    tmp_orders = [order_low]
    for i in range(order_range[0], order_range[1] + 1):
        tmp_orders += [orders[i]]
    tmp_orders += [order_high]
    orders = np.array(tmp_orders)

    tmp_range = [column_range[order_range[0]]]
    for i in range(order_range[0], order_range[1] + 1):
        tmp_range += [column_range[i]]
    tmp_range += [column_range[order_range[1]]]
    column_range = np.array(tmp_range)
    del kwargs["column_range"]  # TODO needs refactoring

    xwd = np.array([xwd[0], *xwd, xwd[-1]])
    del kwargs["xwd"]

    order_range = order_range[0] + 1, order_range[1] + 1
    del kwargs["order_range"]

    if not kwargs.get("thar", False):
        spectrum, slitfunction, uncertainties = optimal_extraction(
            img, head, orders, xwd, column_range, order_range, **kwargs
        )
    else:
        spectrum, slitfunction, uncertainties = arc_extraction(
            img, head, orders, xwd, column_range, **kwargs
        )

    return spectrum, uncertainties


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import readsav

    sav = readsav("./Test/test.dat")
    img = sav["im"]
    ycen = sav["ycen"]
    shear = np.zeros(img.shape[1])

    sp, sl, model, unc = slitfunc(img, ycen, osample=3)

    plt.subplot(211)
    plt.plot(sp)
    plt.title("Spectrum")

    plt.subplot(212)
    plt.plot(sl)
    plt.title("Slitfunction")
    plt.show()
