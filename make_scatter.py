import matplotlib.pyplot as plt
import numpy as np
import logging

from slitfunc_wrapper import slitfunc
from util import bottom, make_index, middle, interpolate_masked


def interpolate_bad_pixels(yy, y1, y2, im, j, nrow):
    yy1 = np.clip(int(np.round(yy + y1)), 0, None)  # first row
    yy2 = np.clip(int(np.round(yy + y2)), None, nrow)  # last row

    if yy2 - yy1 > 0.3 * (y2 - y1) + 1:  # check if inside the frame
        scatter = im[yy1:yy2, j]  # scattered light profile
        scatter = interpolate_masked(scatter)
        return np.median(scatter)  # take median
    else:
        return -1000


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
    first = 0  # first order
    last = len(orders) + 1  # last order
    nord = len(orders)

    # Get kwargs data, TODO change names
    ccd_gain = kwargs.get("gain", 1)
    ccd_rdnoise = kwargs.get("readn", 0) / ccd_gain
    osamp = kwargs.get("osample", 10)
    lambda_sf = kwargs.get("lambda_sf", 1)
    lambda_sp = kwargs.get("lambda_sp", 0)
    extra_offset = kwargs.get("order_extra_width", 1)
    width = kwargs.get("swath_width", 400)
    column_range = kwargs.get("column_range", np.tile([0, ncol], (nord, 0)))

    debug = kwargs.get("debug", False)
    pol = kwargs.get("pol", False)
    subtract = kwargs.get("subtract", False)

    # Initialize arrays.
    xcol = np.arange(ncol)  # indices of all columns
    back = np.zeros((nord + 1, ncol))  # fitted scattered light model
    back_data = np.zeros((nord + 1, ncol))  # scattered light data
    yback = np.zeros((nord + 1, ncol))  # scattered light coordinates
    ycen1 = np.polyval(orders[0], xcol)  # the shape of the starting order

    # TODO DEBUG
    # plt.ion()

    # Loop over neighbouring orders, (0, 1), (1, 2), (2, 3), ...
    for order0, order1 in zip(range(first, last), range(first + 1, last + 1)):
        logging.debug("orders: %i, %i" % (order0, order1))
        # Calculate shapes
        ycen0 = np.polyval(orders[order0], xcol)
        ycen1 = np.polyval(orders[order1], xcol)

        cr0 = column_range[order0]
        cr1 = column_range[order1]

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

        yback[order1, :] = ycen  # scattered light coordinates

        # Copy the region of interest to sf, j=0..nc
        index = make_index(ymin, ymax, ibeg, iend)
        sf = im[index]

        sf = sf - np.min(sf)
        tmp = ycen[ibeg:iend] - ycen[ibeg:iend].astype(int)

        # TODO debug
        # plt.imshow(sf)
        # plt.plot(sf.shape[0]/2 + tmp)
        # plt.show()

        # TODO why is this so slow?
        # TODO use passed parameters
        osamp = 1
        # TODO: Copying these ensures that nothing bad happens during slitfunc, which sometimes happens otherwise
        sf = np.copy(sf)
        tmp = np.copy(tmp)
        sp, sfsm, model, unc = slitfunc(sf, tmp, lambda_sp=2, lambda_sf=2, osample=1)
        nslitf = len(sfsm)
        yslitf = (
            np.arange(-0.5, nslitf - 0.5, 1) / osamp - 1.5 - height
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
        k = np.where(sfsm > np.median(sfsm) + 0.5 * var)[0]
        nback = k.shape[0]
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
            k = np.where(sfsm > middle(sfsm, 1, eps=dev, poly=True))[0]
            m1 = np.where(yslitf[k] < 0)
            m2 = np.where(yslitf[k] > 0)
            nback = k.shape[0]
            n1 = m1[0].shape[0]
            n2 = m2[0].shape[0]
        if n1 == 0 or n2 == 0:  # this happens e.g. if the two adjacent
            # orders have dramatically different level
            # of signal
            k = np.where(sfsm > middle(sfsm, 1e3, eps=dev))[0]
            m1 = np.where(yslitf[k] < 0)
            m2 = np.where(yslitf[k] > 0)
            nback = k.shape[0]
            n1 = m1[0].shape[0]
            n2 = m2[0].shape[0]
        ss = np.array([0, *jback, 0])
        kk = np.where(
            ((ss[k + 1] >= ss[k]) & (ss[k + 1] > ss[k + 2]))
            | ((ss[k + 1] > ss[k]) & (ss[k + 1] >= ss[k + 2]))
        )
        k = k[kk]

        m1 = np.where(yslitf[k] < 0)  # part of the gap which is below the central line
        m2 = np.where(yslitf[k] > 0)  # part of the gap which is above the central line
        n1 = m1[0].shape[0]
        n2 = m2[0].shape[0]
        # ===========================================================================

        if n1 == 0 or n2 == 0:  # ultimate method
            logging.info(
                "mkscatter: failed finding interorder boundaries. Using 5 pixels around central line."
            )
            k = np.sort(abs(yslitf))[0] + [-3, 3]
            nback = len(k)
            debug = True

        if nback < 2 or (
            nback >= 2
            and (np.min(yslitf[k]) > height * 0.5 or np.max(yslitf[k]) < -height * 0.5)
        ):
            n1 = np.min(np.abs(yslitf + height * 0.5))  # in this case just select
            n2 = np.min(np.abs(yslitf - height * 0.5))  # the central half
            k1 = n1.shape[0]
            k2 = n2.shape[0]
            k = [k1, k2]
            nback = 2

        if jback[0] >= jback[1]:
            k = np.array([0, *k])
            nback = nback + 1
        if jback[nslitf - 1] >= jback[nslitf - 2]:
            k = np.array([*k, nslitf - 1])
            nback = nback + 1

        if debug:
            plt.plot(yslitf, sfsm)
            plt.plot(yslitf, middle(sfsm, 1, eps=dev, poly=True))
            plt.plot(yslitf, middle(sfsm, 1e3, eps=dev, poly=True))
            plt.hline(np.median(sfsm) + 0.5 * var)
            plt.plot(yslitf[k], sfsm[k])
            plt.show()

        jback = np.where(yslitf[k] < 0)[0]
        n1 = jback.shape[0]
        imax1 = np.argmax(yslitf[k[jback]])
        m1 = yslitf[k[jback]][imax1]
        imax1 = k[jback[imax1]]

        jback = np.where(yslitf[k] > 0)[0]
        n2 = jback.shape[0]
        imax2 = np.argmin(yslitf[k[jback]])
        m2 = yslitf[k[jback]][imax2]
        imax2 = k[jback[imax2]]

        k = [imax1, imax2]
        if nback == 0:
            print(
                "mkscatter: failed to find the boundaries of the inter-order troff using default orders: %i-%i x-range: (%i,%i)"
                % (order0, order1, ibeg, iend)
            )
            k = [0, len(nslitf)]

        tmp = np.clip(sfsm[k[0] : k[1]], None, np.median(sfsm))
        jback = bottom(
            sfsm[k[0] : k[1]], 1, eps=dev, poly=True
        )  # fit bottom with a straight line
        iback = np.where(sfsm[k[0] : k[1]] <= jback + ccd_rdnoise)[
            0
        ]  # find all the points below
        nback = iback.shape[0]
        if nback <= 5:
            plt.plot(yslitf, sfsm)
            plt.plot(
                yslitf,
                bottom(np.clip(sfsm, None, np.median(sfsm)), 1, eps=dev, poly=True),
            )
            plt.plot(yslitf[k], sfsm[k], "*")
            raise Exception(
                "mkscatter: major error in the order format: could not detect inter-order troff"
            )
        iback += k[0]
        imax1 = np.min(iback)
        imax2 = np.max(iback)
        sf1 = sfsm[iback]

        step = (max(sf1) - min(sf1)) / np.clip(len(sf1) / 10, 10, 200)
        nh = int(1 / step)
        h, _ = np.histogram(sf1, bins=nh)
        ihmax = np.argmax(h)
        hmax = h[ihmax]

        i0 = np.where(h[:ihmax] < 0.1 * hmax)
        n1 = i0[0].shape[0]
        i0 = np.max(i0) if n1 > 0 else 0

        i1 = np.where(h[ihmax:] < 0.1 * hmax)[0]
        n2 = i1.shape[0]
        i1 = np.min(i1) + ihmax if n2 > 0 else nh - 1

        ii = (
            np.where((sf1 - min(sf1) >= step * i0) & (sf1 - min(sf1) < step * i1))[0]
            + imax1
        )
        nii1 = ii.shape[0]
        if nii1 <= 0:
            raise Exception(
                "mkscatter: could not detect background points between orders %i and %i"
                % (order0, order1)
            )

        y1 = np.ceil(np.min(yslitf[ii]))  # bottom boundary of the troff
        y2 = np.floor(np.max(yslitf[ii]))  # top boundary of the troff

        for j in range(ncol):  # scattered light in col j
            back_data[order1, j] = interpolate_bad_pixels(ycen[j], y1, y2, im, j, nrow)

            if order0 == first:  # for the first order try
                yy = ycen0[j] - (ycen[j] - ycen0[j])  # to find background below
                yback[0, j] = yy  # scattered light coordinates
                back_data[0, j] = interpolate_bad_pixels(yy, y1, y2, im, j, nrow)
            elif order1 == last:  # for the last order try
                yy = ycen1[j] + (ycen1[j] - ycen[j])  # to find background above
                yback[-1, j] = yy  # scattered light coordinates
                back_data[-1, j] = interpolate_bad_pixels(yy, y1, y2, im, j, nrow)

        if (order1 % 10) == 5 or (order1 % 10) == 0 or nord < 10:
            logging.info("mkscatter: order %i of total  %i was processed", order1, last)

    # bad pixels are masked out
    back_data = np.ma.masked_array(back_data, mask=back_data == -1000)

    # Interpolate missing data from the adjacent troffs
    for i in range(ncol):
        back_data[:, i] = np.interp(yback[:, i], yback[:, i], back_data[:, i])

    # Filter out noise: 0th background troff
    crange = column_range[0]
    x = np.arange(crange[0], crange[1])
    b = middle(back_data[0, x], 20., eps=dev)
    if pol:
        back[0, x] = middle(b, 11, poly=True, eps=dev, min=0)
    else:
        back[0, x] = middle(b, lambda_sp, eps=dev, min=0)

    # All intermediate background troffs
    if nord > 1:
        for order0, order1 in zip(range(first, last), range(first + 1, last + 1)):
            cr0 = column_range[order0]
            cr1 = column_range[order1]

            crange = (
                max(cr0[0], cr1[0]),
                min(cr0[1], cr1[1]),
            )  # right boundary to consider

            x = np.arange(crange[0], crange[1])
            b = middle(back_data[order1, x], 20., eps=dev)
            if pol:
                back[order1, x] = middle(b, 11, poly=True, eps=dev, min=0, double=True)
            else:
                back[order1, x] = middle(b, lambda_sp, eps=dev, min=0)

    # The last background troff
    crange = column_range[last]
    x = np.arange(crange[0], crange[1])
    b = middle(back_data[last, x], 20., eps=dev)
    if pol:
        back[-1, x] = bottom(b, 11, poly=True, eps=dev, min=0)
    else:
        back[-1, x] = bottom(b, lambda_sp, eps=dev, min=0)

    if subtract:
        # Type conversion to avoid problems with UINT arrays when subtracting

        im = im.astype(np.float32)
        ycen = np.arange(nrow)
        if pol:
            ii = np.arange(0, nord, 2)  # polarization: orders come in pairs
            ii = np.array([*ii, np.max(ii) + 2])  # do not use interpolarization space
        else:
            ii = np.arange(nord)  # no polarization, count all orders

        for j in range(ncol):
            b = back[:, j]
            b = b[ii]
            y = yback[:, j]
            y = y[ii]
            im[:, j] -= np.interp(ycen, y, b)

    return back, yback
