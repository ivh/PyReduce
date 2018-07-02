import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solve_banded, solve
from scipy.ndimage.filters import median_filter
from extract import slitfunc
from util import make_index, top, middle, bottom


def make_scatter(im, orders, **kwargs):
    """
    This subroutine is still far from perfect. NP wrote it in 2002 and since it
    was a pain in the back. The problem is that that it works very smoothlu with
    a good data set but once we get to the low S/N, overlapping orders and bad
    cosmetics things starting to fail. Typical failures are connected to the
    the identification of the walls of adjucent orders and the bottom.
    
    History of modifications:
    2006-09-26 (NP) the condition for local maxima ID is set to
    (positive left der and non-positive right der) or
    (non-negative left der and negative right der)
    """

    # Get image size.
    nrow, ncol = im.shape
    nord = len(orders)  # number of orders

    # Get kwargs data, TODO change names
    ccd_gain = kwargs.get("gain", 1)
    ccd_rdnoise = kwargs.get("readn", 0) / ccd_gain
    osamp = kwargs.get("osample", 10)
    lambda_sf = kwargs.get("lam_sf", 1)
    lambda_sp = kwargs.get("lam_sp", 0)
    extra_offset = kwargs.get("order_extra_width", 1)
    width = kwargs.get("swath_width")
    column_range = kwargs.get("colrange", np.array([(0, ncol) for _ in orders]))

    debug = kwargs.get("debug", False)
    pol = kwargs.get("pol", False)
    subtract = kwargs.get("subtract", False)

    # Initialize arrays.
    xcol = np.arange(ncol)  # indices of all columns
    back = np.zeros((ncol, nord + 1))  # fitted scattered light model
    back_data = np.zeros((ncol, nord + 1))  # scattered light data
    yback = np.zeros((ncol, nord + 1))  # scattered light coordinates
    ycen1 = np.polyval(orders[0], xcol)  # the shape of the starting order

    # TODO: Loop over neighbouring orders, i.e. (0, 1), (1, 2), ...
    for i, order in orders.items():
        if i == 0:
            continue  # Skip order 0
        ycen0 = np.copy(ycen1)
        ycen1 = np.polyval(order, xcol)  # the shape of the next order

        cr0 = column_range[i - 1]
        cr1 = column_range[i]

        ibeg = max(cr0[0], cr1[0])  # left boundary to consider
        iend = min(cr0[1], cr1[1])  # right boundary to consider
        width = np.clip(
            (iend - ibeg) * 0.1, width, None
        )  # width to consider (swath < width < iend-ibeg+1)

        width = int(width // 2)  # half-width to avoid rounding errors
        height = np.mean(ycen1[ibeg:iend] - ycen0[ibeg:iend])  # mean height
        height = int(np.round(height * 0.5 * extra_offset))
        height = np.clip(height, 3, None)  # half-height to avoid rounding errors

        icen = (ibeg + iend) // 2  # central column
        ibeg = np.clip(
            icen - width, ibeg, None
        )  # starting column of the region of interest
        iend = np.clip(
            icen + width, None, iend
        )  # ending column of the region of interest
        ycen = 0.5 * (ycen1 + ycen0)  # central line of the troff
        ymin = ycen.astype(int) - height  # bottom boundary of the region of interest
        ymax = ycen.astype(int) + height  # top boundary of the region of interest

        nc = 2 * width  # dimensions of the region of interest
        nr = 2 * height

        # Copy the region of interest to sf, j=0..nc
        index = make_index(ymin, ymax, ibeg, iend)
        sf = im[index]

        sf = sf - np.min(sf)
        tmp = ycen[ibeg:iend] - ycen[ibeg:iend].astype(int)
        # TODO why is this so slow?
        # TODO use passed parameters
        # alternative: sfsm = np.sum(sf, axis=1)
        sp, sfsm, model, unc = slitfunc(sf, tmp, lambda_sp=2, lambda_sl=2, osample=1)

        nslitf = len(sfsm)
        yslitf = (
            np.arange(0.5, nslitf + 0.5, 1) / osamp - 1.5 - height
        )  # final subpixel scale

        dev = (
            ccd_rdnoise / ccd_gain / np.sqrt(np.sum(sp))
        )  # typical pixel scatter normalized to counts

        var = np.std(sfsm)

        #
        # This is an improved version which is near the final form
        #
        jback = bottom(
            np.clip(sfsm, None, np.median(sfsm)), 1, eps=dev, poly=True
        )  # fit the bottom with a smooth curve
        jback = np.clip(sfsm - jback, 0, None)  # find all positive spikes

        # ===========================================================================
        k = np.where(sfsm > np.median(sfsm) + 0.5 * var)
        nback = k[0].shape[0]
        if nback > 0:
            m1 = np.where(
                yslitf[k] < 0
            )  # part of the gap which is below the central line
            m2 = np.where(
                yslitf[k] > 0
            )  # part of the gap which is above the central line
            n1 = m1[0].shape[0]
            n2 = m2[0].shape[0]
        else:
            n1 = 0
            n2 = 0
        if n1 == 0 or n2 == 0:  # this happens e.g. if the two adjacent
            # orders have dramatically different level
            # of signal
            k = np.where(sfsm > middle(sfsm, 1, eps=dev, poly=True))
            m1 = np.where(yslitf[k] < 0)
            m2 = np.where(yslitf[k] > 0)
            nback = k[0].shape[0]
            n1 = m1[0].shape[0]
            n2 = m2[0].shape[0]
        if n1 == 0 or n2 == 0:  # this happens e.g. if the two adjacent
            # orders have dramatically different level
            # of signal
            k = np.where(sfsm > middle(sfsm, 1e3, eps=dev))
            m1 = np.where(yslitf[k] < 0)
            m2 = np.where(yslitf[k] > 0)
            nback = k[0].shape[0]
            n1 = m1[0].shape[0]
            n2 = m2[0].shape[0]
        ss = np.array([0, *jback, 0])
        kk = np.where(
            (ss[k + 1] >= ss[k] and ss[k + 1] > ss[k + 2])
            or (ss[k + 1] > ss[k] and ss[k + 1] >= ss[k + 2])
        )
        k = k[kk]

        m1 = np.where(yslitf[k] < 0)  # part of the gap which is below the central line
        m2 = np.where(yslitf[k] > 0)  # part of the gap which is above the central line
        n1 = m1[0].shape[0]
        n2 = m2[0].shape[0]
        # ===========================================================================

        if n1 == 0 or n2 == 0:  # ultimate method
            print(
                "mkscatter: failed finding interorder boundaries. "
                + "using 5 pixels around central line."
            )
            k = np.sort(abs(yslitf))[0] + [-3, 3]
            nback = len(k)
            debug = True

        if nback < 2 or (
            nback >= 2
            and (np.min(yslitf(k)) > height * 0.5 or np.max(yslitf(k)) < -height * 0.5)
        ):
            n1 = np.min(np.abs(yslitf + height * 0.5))  # in this case just select
            n2 = np.min(np.abs(yslitf - height * 0.5))  # the central half
            k1 = n1.shape[0]
            k2 = n2.shape[0]
            k = [k1, k2]
            nback = 2

        if jback(0) >= jback(1):
            k = (0, k)
            nback = nback + 1
        if jback(nslitf - 1) >= jback(nslitf - 2):
            k = (k, nslitf - 1)
            nback = nback + 1

        if debug:
            plt.plot(yslitf, sfsm)
            plt.plot(yslitf, middle(sfsm, 1, eps=dev, poly=True))
            plt.plot(yslitf, middle(sfsm, 1e3, eps=dev))
            plt.hline(np.median(sfsm) + 0.5 * var + (0, 0), line=2)
            plt.plot(yslitf[k], sfsm[k], psym=2)
            plt.show()

        jback = np.where(yslitf[k] < 0)
        n1 = jback[0].shape[0]
        imax1 = np.argmax(yslitf[k[jback]])
        m1 = yslitf[k[jback]][imax1]
        imax1 = k[jback[imax1]]

        jback = np.where(yslitf[k] > 0)
        n2 = jback[0].shape[0]
        imax2 = np.argmin(yslitf[k[jback]])
        imax2 = k[jback[imax2]]
        m2 = yslitf[k[jback]][imax2]

        k = (imax1, imax2)
        if nback == 0:
            print(
                "mkscatter: failed to find the boundaries of the inter-order troff using default order: %i x-range: (%i,%i)"
                % (order, ibeg, iend)
            )
            k = [0, len(nslitf)]

        # if(dbg eq 1) then stop
        jback = bottom(sfsm[k[0]], 1)  # fit bottom with a straight line
        iback = np.where(sfsm[k[0]] <= jback + ccd_rdnoise)  # find all the points below
        nback = iback[0].shape[0]
        if nback <= 5:
            plt.plot(yslitf, sfsm, xs=1)
            plt.plot(
                yslitf,
                bottom(np.clip(sfsm, None, np.median(sfsm)), 1, eps=dev, poly=True),
            )
            plt.plot(yslitf[k], sfsm[k], psym=2)
            raise Exception(
                "mkscatter: major error in the order format: could not detect inter-order troff"
            )
        iback = iback + k[0]
        imax1 = np.min(iback)
        imax2 = np.max(iback)
        sf1 = sfsm[iback]
        iback = 0
        jback = 0

        step = (max(sf1) - min(sf1)) / np.clip(len(sf1) / 10, 10, 200)
        h = plt.hist(sf1, bin=step)
        nh = len(h)
        hmax = max(h)
        ihmax = len(hmax)

        i0 = np.where(h[0 : ihmax + 1] < 0.1 * hmax)
        n1 = i0[0].shape[0]
        if n1 > 0:
            i0 = max(i0)
        else:
            i0 = 0
        i1 = np.where(h[ihmax : nh - 1 + 1] < 0.1 * hmax)
        n2 = i1[0].shape[0]
        if n2 > 0:
            i1 = min(i1) + ihmax
        else:
            i1 = nh - 1
        ii = (
            np.where(sf1 - min(sf1) >= step * i0 and sf1 - min(sf1) < step * i1) + imax1
        )
        nii1 = ii[0].shape[0]
        if nii1 <= 0:
            print(
                "mkscatter: could not detect background points between orders %i and %i"
                % (ord - 1, ord)
            )
        else:
            y1 = np.ceil(np.min(yslitf[ii]))  # bottom boundary of the troff
            y2 = np.floor(np.max(yslitf[ii]))  # top boundary of the troff

        if debug:
            plt.axvline(y1)
            plt.axvline(y2)
            plt.show()

        if nii1 <= 0:
            raise Exception("mkscatter: could not detect any background points")

        yback[ord, :] = ycen  # scattered light coordinates

        for j in range(ncol):  # scattered light in col j
            yy1 = np.clip(np.round(ycen[j] + y1), 0, None)  # first row
            yy2 = np.clip(np.round(ycen[j] + y2), None, nrow - 1)  # last row

            if yy2 - yy1 > (0.3 * (y2 - y1) + 1):  # check if inside the frame
                scatter = im[yy1 : yy2 + 1, j]  # scattered light profile
                # Interpolate bad pixels in mask, TODO
                scatter = interpol(np.arange(0, yy2 - yy1), k, scatter[k])
                back_data[ord, j] = np.median(scatter)  # take median
            else:
                back_data[ord, j] = -10000  # bad point

            if ord == 1:  # for the first order try
                yy = ycen0[j] - (ycen[j] - ycen0[j])  # to find background below
                yy1 = np.clip(np.round(yy + y1), 0, None)  # first row
                yy2 = np.clip(np.round(yy + y2), None, nrow - 1)  # last row
                yback[ord - 1, j] = yy  # scattered light coordinates

                if yy2 - yy1 > (0.3 * (y2 - y1) + 1):  # check if inside the frame
                    scatter = im[yy1 : yy2 + 1, j]  # scattered light profile
                    # Interpolate bad pixels in mask, TODO
                    scatter = interpol(np.arange(0, yy2 - yy1), k, scatter[k])
                    back_data[ord - 1, j] = np.median(scatter)  # take median
                else:
                    back_data[ord - 1, j] = -10000
            elif ord == nord - 1:  #             ;for the last order try
                yy = ycen1[j] + (ycen1[j] - ycen[j])  # to find background above
                yback[ord + 1, j] = yy  # scattered light coordinates
                yy1 = np.clip(np.round(yy + y1), 0, None)  # first row
                yy2 = np.clip(np.round(yy + y2), None, nrow - 1)  # last row

                if yy2 - yy1 > (0.3 * (y2 - y1) + 1):  # check if inside the frame
                    scatter = im[yy1 : yy2 + 1, j]  # scattered light profile
                    scatter = interpol(np.arange(0, yy2 - yy1), k, scatter[k])
                    back_data[ord + 1, j] = np.median(scatter)  # take median
                else:
                    back_data[ord + 1, j] = -10000

        if ((ord + 1) % 10) == 5 or ((ord + 1) % 10) == 0 and ord != nord - 1:
            print("mkscatter: order %i of total  %i was processed" % (ord + 1, nord))
    print("mkscatter: order %i was processed" % nord)

    # Interpolate missing data from the adjacent troffs
    for i in range(ncol):
        y = yback[:, i]
        k = np.where(back_data[:, i] > -1000)
        back_data[:, i] = interpol(yback[:, i], yback[k, i], back_data[k, i])

    # Filter out noise: 0th background troff
    crange = (column_range[0, 0], column_range[0, 1])
    x = np.arange(crange[1] - crange[0]) + crange[0]
    b = middle(back_data[0, x], 20., eps=dev)
    if pol:
        back[0, x] = middle(b, 11, poly=True, eps=dev, min=0)
    else:
        back[0, x] = middle(b, lambda_sp, eps=dev, min=0)

    # All intermediate background troffs
    if nord > 1:
        for order in orders.keys():
            crange = (
                column_range[order - 1][0] < column_range[order][0],
                column_range[order - 1][1] > column_range[order][1],
            )
            x = np.arange(crange[1] - crange[0]) + crange[0]
            b = middle(back_data[ord, x], 20., eps=dev)
            if pol:
                back[ord, x] = middle(b, 11, poly=True, eps=dev, min=0, double=True)
            else:
                back[ord, x] = middle(b, lambda_sp, eps=dev, min=0)

    # The last background troff
    crange = (column_range[nord - 1, 0], column_range[nord - 1, 1])
    x = np.arange(crange[1] - crange[0]) + crange[0]
    b = middle(back_data[nord, x], 20., eps=dev)
    if pol:
        back[nord, x] = bottom(b, 11, poly=True, eps=dev, min=0, double=True)
    else:
        back[nord, x] = bottom(b, lambda_sp, eps=dev, min=0)

    if subtract:
        # Type conversion to avoid problems with UINT arrays when subtracting

        im = im.astype(np.float32)
        ycen = np.arange(nrow)
        if pol:
            ii = np.arange(nord / 2) * 2  # polarization: orders come in pairs
            ii = (ii, np.max(ii) + 2)  # do not use interpolarization space
        else:
            ii = np.arange(nord + 1)  # no polarization, count all orders
        for j in range(ncol):
            b = back[:, j]
            b = b[ii]
            y = yback[:, j]
            y = y[ii]
            im[:, j] -= interpol(ycen, y, b)

    return back, yback
