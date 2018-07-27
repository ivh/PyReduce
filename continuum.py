import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
from itertools import chain

from util import top, middle, bezier_interp


#

def splice_orders(
    spec, wave, cont, sigm, orders=None, column_range=None, scaling=False, debug=True
):
    nord, ncol = spec.shape  # Number of sp. orders, Order length in pixels
    if column_range is None:
        column_range = np.tile([0, ncol], (nord, 1))
    if orders is None:
        orders = np.arange(nord)

    nord = len(orders)
    order_scales = np.ones(nord)



    # Reorganize input arrays, to make everything simpler
    # Memory addresses stay the same throughout the function, so any changes will also be in data
    # data = np.rec.fromarrays([spec, wave, cont, sigma], names="spec,wave,cont,sigma")
    # data = data[orders]
    spec = spec[orders]
    wave = wave[orders]
    cont = cont[orders]
    sigm = sigm[orders]
    column_range = column_range[orders]

    mask = np.full(spec.shape, True)
    for iord in range(len(column_range)):
        mask[iord, column_range[iord, 0] : column_range[iord, 1]] = False

    spec = np.ma.masked_array(spec, mask=mask)
    wave = np.ma.masked_array(wave, mask=mask)
    cont = np.ma.masked_array(cont, mask=mask)
    sigm = np.ma.masked_array(sigm, mask=mask)


    # Scale everything to roughly the same size, around spec/blaze = 1
    if scaling:
        scale = np.ma.median(spec, axis=1) / np.ma.median(cont, axis=1)
    else:
        scale = np.array([1])
    cont = median_filter(cont, 5) * scale[:, None]
    cont.mask = mask #median filter changes the mask, but we want it to stay the same still

    if debug:
        plt.subplot(211)
        plt.title("Before")
        for i in range(spec.shape[0]):
            plt.plot(wave[i], spec[i] / cont[i])


    # Order with largest signal, everything is scaled relative to this order
    iord0 = np.argmax(np.ma.median(spec / cont, axis=1))

    # Loop from iord0 outwards, first to the top, then to the bottom
    tmp0 = chain(range(iord0, 0, -1), range(iord0, nord - 1))
    tmp1 = chain(range(iord0 - 1, -1, -1), range(iord0 + 1, nord))

    for iord0, iord1 in zip(tmp0, tmp1):
        # get data for current order
        # Note that those are just references to parts of the original data
        # any changes will also affect spec, wave, cont, and sigm
        s0, s1 = spec[iord0], spec[iord1]
        w0, w1 = wave[iord0], wave[iord1]
        c0, c1 = cont[iord0], cont[iord1]
        u0, u1 = sigm[iord0], sigm[iord1]

        # Calculate Overlap
        i0 = np.where((w0 >= np.min(w1)) & (w0 <= np.max(w1)))
        i1 = np.where((w1 >= np.min(w0)) & (w1 <= np.max(w0)))

        # Orders overlap
        if i0[0].size > 0 and i1[0].size > 0:
            # Interpolate the overlapping region onto the wavelength grid of the other order
            tmpS0 = bezier_interp(w1, s1, w0[i0])
            tmpB0 = bezier_interp(w1, c1, w0[i0])
            tmpU0 = bezier_interp(w1, u1, w0[i0])

            tmpS1 = bezier_interp(w0, s0, w1[i1])
            tmpB1 = bezier_interp(w0, c0, w1[i1])
            tmpU1 = bezier_interp(w0, u0, w1[i1])

            # Weights depend on the direction of the orders
            if iord0 > iord1:
                wgt0 = np.linspace(1, 0, i0[0].size)
                wgt1 = np.linspace(0, 1, i1[0].size)
            else:
                wgt0 = np.linspace(0, 1, i0[0].size)
                wgt1 = np.linspace(1, 0, i1[0].size)

            # Combine the two orders linearly to get get spliced spectrum
            s0[i0] = s0[i0] * wgt0 + tmpS0 * (1 - wgt0)
            c0[i0] = c0[i0] * wgt0 + tmpB0 * (1 - wgt0)
            u0[i0] = np.sqrt(u0[i0] ** 2 * wgt0 + tmpU0 ** 2 * (1 - wgt0))

            s1[i1] = s1[i1] * wgt1 + tmpS1 * (1 - wgt1)
            c1[i1] = c1[i1] * wgt1 + tmpB1 * (1 - wgt1)
            u1[i1] = np.sqrt(u1[i1] ** 2 * wgt1 + tmpU1 ** 2 * (1 - wgt1))

        else:  # Orders dont overlap
            raise NotImplementedError("Orders don't overlap, please test")
            scale0 = top(s0 / c0, 1, poly=True)
            c0 *= scale0
            scale0 = top(s0 / c0, 1, poly=True)
            scale0 = np.polyfit(w0, scale0, 1)

            scale1 = top(s1 / c1, 1, poly=True)
            scale1 = np.polyfit(w1, scale1, 1)

            xx = np.linspace(np.min(w0), np.max(w1), 100)

            # TODO try this
            # scale = np.sum(scale0[0] * scale1[0] * xx * xx + scale0[0] * scale1[1] * xx + scale1[0] * scale0[1] * xx + scale1[1] * scale0[1])
            scale = scale0[::-1, None] * scale1[None, ::-1]
            scale = np.sum(np.polynomial.polynomial.polyval2d(xx, xx, scale)) / np.sum(
                np.polyval(scale1, xx) ** 2
            )
            s1 *= scale
            order_scales[iord1] = scale

    # TODO: flatten data into one large spectrum
    # Problem: orders overlap

    # data = np.sort(data.flatten(), order="wave")

    if debug:
        plt.subplot(212)
        plt.title("After")
        for i in range(nord):
            plt.plot(wave[i], spec[i] / cont[i], label="order=%i" % i)

        plt.legend(loc="best")
        plt.show()

    return wave, spec, cont, sigm


if __name__ == "__main__":
    pass
