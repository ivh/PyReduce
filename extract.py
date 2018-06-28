import numpy as np
from scipy.ndimage.filters import median_filter, gaussian_filter1d
from scipy.interpolate import interp1d

# TODO DEBUG
import clib.build_extract

clib.build_extract.build()


import clib._slitfunc_bd.lib as slitfunclib
import clib._slitfunc_2d.lib as slitfunc_2dlib
from clib._cluster import ffi

c_double = np.ctypeslib.ctypes.c_double
c_int = np.ctypeslib.ctypes.c_int


def slitfunc(img, ycen, lambda_sp=0, lambda_sl=0.1, osample=1):
    """Decompose image into spectrum and slitfunction

    This is for vertical(?) orders only, for curved orders use slitfunc_curved instead

    Parameters
    ----------
    img : array[n, m]
        image to decompose, should just contain a small part of the overall image
    ycen : array[n]
        traces the center of the order along the image
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

    # Get dimensions
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    # Inital guess for slit function and spectrum
    # Just sum up the image along one side and normalize
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

    ncol, nrow = img.shape
    i1, i2 = x_left_lim, x_right_lim
    # if(keyword_set(x_left_lim )) then i1 = x_left_lim  else i1 = 0
    # if(keyword_set(x_right_lim)) then i2 = x_right_lim else i2 = ncol-1

    # Define useful quantities
    nrow = len(img[0, i1:i2])  # number of columns
    ncol = len(img[:, i1])  # number of rows
    maxo = len(orders)  # maximum order covered by orc
    ix = np.arange(i1, nrow + i1, 1, dtype=float)  # vector of column indicies
    arc = np.zeros_like(ix)  # dimension arc vector
    pix = 1.0  # define in case of trouble

    # Interpolate polynomial coefficients for surrounding orders to get polynomial
    # coefficients.  Note that this is mathematically equivalent to interpolating
    # the column indicies for surrounding orders, since the column indicies are
    # linear functions of the polynomial coefficients. However, interpolating
    # coefficients should be faster.
    # The +/- 10000 is to force argument of LONG to be positive before truncation.

    if np.max(awid) < 1:  # awid is an order fraction
        if onum < awid or onum > maxo - awid:  # onum must be covered by orc
            raise Exception(
                "Requested order not covered by order location coefficients. %i" % onum
            )

        ob = onum - awid / 2  # order # of bottom edge of arc
        obi = int(ob)  # next lowest integral order #A
        cb = orders[obi] + (ob - obi) * (orders[obi + 1] - orders[obi])
        yb = np.polyval(cb, ix)  # row # of bottom edge of swath

        ot = onum + awid / 2  # order # of top edge of arc
        oti = int(ot)  # next lowest integral order #
        ct = orders[oti] + (ot - oti) * (orders[oti + 1] - orders[oti])
        yt = np.polyval(ct, ix)  # row # of top edge of swath

    else:  # awid is number of pixels
        if onum < 0 or onum > maxo:  # onum must be covered by orc
            raise Exception(
                "Requested order not covered by order location coefficients. %i" % onum
            )
        c = orders[onum]
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
    for col in range(nrow):  # sum image in requested arc
        scol = img[yb[col] : yt[col], col + i1]
        arc[col] = np.sum(scol)

    # Define vectors along edge of swath.
    # vb = img[ix + ncol * ybi]  # bottommost pixels in swath
    # vt = img[ix + ncol * yti]  # topmost pixels in swath

    return arc, pix


def optimal_extraction(
    img, head, orders, swath, column_range, order_range, *args, **kwargs
):
    print("GETSPEC: Using optimal extraction to produce spectrum.")
    ofirst, olast = order_range
    swath_boundaries = np.zeros((2, 1), dtype=int)

    sl_smooth = kwargs.get("sf_smooth", 6)
    sp_smooth = kwargs.get("sp_smooth", 1)
    osample = kwargs.get("osample", 1)

    n_row, n_col = img.shape
    n_ord = len(orders)

    spectrum = np.zeros(n_ord, n_col)
    slitfunction = np.zeros(n_ord, 50)  # ?
    uncertainties = [None for _ in orders]
    ix = np.arange(n_col)

    for i, onum in enumerate(range(ofirst, olast)):  # loop thru orders
        ncole = (
            column_range[onum, 1] - column_range[onum, 0] + 1
        )  # number of columns to extract
        cole0 = column_range[onum, 0]  # first column to extract
        cole1 = column_range[onum, 1]  # last column to extract

        # Background must be subtracted for slit function logic to work but kept
        # as part of the FF signal during normalization

        scatter_below = 0
        yscatter_below = 0
        scatter_above = 0
        yscatter_above = 0

        if n_ord <= 10:
            print("GETSPEC: extracting relative order %i out of %i" % (onum, n_ord))
        else:
            if (onum - 1) % 5 == 0:
                print(
                    "GETSPEC: extracting relative orders %i-%i out of %i"
                    % (onum, np.clip(n_ord, None, onum + 4), n_ord)
                )

        ycen = np.polyval(orders(onum), ix)  # row at order center

        x_left_lim = cole0  # First column to extract
        x_right_lim = cole1  # Last column to extract
        ixx = ix[x_left_lim:x_right_lim]
        ycenn = ycen[x_left_lim:x_right_lim]
        if swath[onum, 0] > 1.5:  # Extraction width in pixels
            ymin = ycenn - swath[onum, 0]
        else:  # Fractional extraction width
            ymin = ycenn - swath[onum, 0] * (
                ycenn - np.polyval(orders[onum - 1], ixx)
            )  # trough below

        ymin = np.floor(ymin)
        if min(ymin) < 0:
            ymin = ymin - min(ymin)  # help for orders at edge
        if swath[onum, 1] > 1.5:  # Extraction width in pixels
            ymax = ycenn + swath[onum, 1]
        else:  # Fractional extraction width
            ymax = ycenn + swath[onum, 1] * (
                np.polyval(orders[onum - 1], ixx) - ycenn
            )  # trough above

        ymax = np.ceil(ymax)
        if max(ymax) > n_row:
            ymax = ymax - max(ymax) + n_row - 1  # helps at edge

        # Define a fixed height area containing one spectral order
        y_lower_lim = int(min(ycen[cole0:cole1] - ymin))  # Pixels below center line
        y_upper_lim = int(min(ymax - ycen[cole0:cole1]))  # Pixels above center line

        spectrum[i, :], slitfunction[i], _, uncertainties[i, :] = slitfunc(
            img, ycen, lambda_sp=sp_smooth, lambda_sl=sl_smooth, osample=osample
        )

    return spectrum, slitfunction, uncertainties


def arc_extraction(img, head, orders, swath, column_range, **kwargs):
    print("Using arc extraction to produce spectrum.")
    n_row, n_col = img.shape
    n_ord = len(orders)

    gain = head["e_gain"]
    dark = head["e_drk"]
    readn = head["e_readn"]

    spectrum = np.zeros((n_ord-2, n_col))
    uncertainties = np.zeros((n_ord-2, n_col))

    for i, onum in enumerate(range(1, n_ord-1)):  # loop thru orders
        x_left_lim = column_range[onum, 0]  # First column to extract
        x_right_lim = column_range[onum, 1]  # Last column to extract
        awid = swath[onum, 0] + swath[onum, 1]

        arc, pix = getarc(
            img, orders, onum, awid, x_left_lim=x_left_lim, x_right_lim=x_right_lim
        )  # extract counts/pixel

        spectrum[i, x_left_lim:x_right_lim] = arc * pix  # store total counts
        uncertainties[i, x_left_lim:x_right_lim] = (
            np.sqrt(abs(arc * pix * gain + dark + pix * readn ** 2)) / gain
        )  # estimate uncertainty

    return spectrum, 0, uncertainties


def extract(img, head, orders, **kwargs):
    # TODO which parameters should be passed here?
    swath = kwargs.get("xwd", 50)
    column_range = kwargs.get(
        "column_range", np.array([(0, img.shape[1]) for _ in orders])
    )
    if "tilt" in kwargs.keys():
        shear = kwargs["tilt"]
        # TODO use curved extraction

    n_row, n_col = img.shape
    n_ord = len(orders)
    order_range = kwargs.get("order_range", (1, n_ord))
    ix = np.arange(n_col)

    # Extrapolate extra orders above and below the existing ones

    if n_ord > 1:
        ixx = ix[column_range[0, 0] : column_range[0, 1]]
        order_low = 2 * orders[0] - orders[1]  # extrapolate orc
        if swath[0, 0] > 1.5:  # Extraction width in pixels
            coeff = orders[0]
            coeff[-1] -= swath[0, 0]
        else:  # Fraction extraction width
            coeff = 0.5 * ((2 + swath[0, 0]) * orders[0] - swath[0, 0] * orders[1])

        y = np.polyval(coeff, ixx)  # low edge of arc
        noff = np.min(y) < 0
    else:
        noff = False
        order_low = [0, *np.zeros_like(orders[0])]
    if noff:  # check if on image
        #   GETARC will reference im(j) where j<0. These array elements do not exist.
        raise Exception("Top order off image.")

    # Extend orc on the high end. Check that requested swath lies on image.
    if n_ord > 1:
        ix1 = column_range[-1, 0]
        ix2 = column_range[-1, 1]
        ixx = ix[ix1:ix2]
        order_high = 2 * orders[-1] - orders[-2]  # extrapolate orc
        if swath[-1, 1] > 1.5:  # Extraction width in pixels
            coeff = orders[-1]
            coeff[-1] += swath[-1, 1]
        else:  # Fraction extraction width
            coeff = 0.5 * ((2 + swath[-1, 1]) * orders[-1] - swath[-1, 1] * orders[-2])
        y = np.polyval(coeff, ixx)  # high edge of arc
        noff = np.max(y) > n_row
    else:
        noff = False
        order_high = [n_row, *np.zeros_like(orders[0])]
    if noff:
        raise Exception("Bottom order off image in columns.")

    orders = [order_low, *orders, order_high]
    column_range = np.array([column_range[0], *column_range, column_range[-1]])
    swath = np.array([swath[0], *swath, swath[-1]])

    if not kwargs.get("thar", False):
        spectrum, slitfunction, uncertainties = optimal_extraction(
            img, head, orders, swath, column_range, order_range, **kwargs
        )
    else:
        spectrum, slitfunction, uncertainties = arc_extraction(
            img, head, orders, swath, column_range, **kwargs
        )

    return spectrum, slitfunction, uncertainties


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import readsav

    sav = readsav("./Test/test.dat")
    img = sav["im"]
    ycen = sav["ycen"]
    shear = np.zeros(img.shape[1])

    sp, sl, model, unc = slitfunc_curved(img, ycen, shear, osample=1)

    plt.subplot(211)
    plt.plot(sp)
    plt.title("Spectrum")

    plt.subplot(212)
    plt.plot(sl)
    plt.title("Slitfunction")
    plt.show()
