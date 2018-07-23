import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
from itertools import chain

import awlib.bezier as bezier
from util import top


def splice_orders(
    spec, wave, blaze, sigma, orders=None, column_range=None, scaling=False, debug=True
):
    nord, ncol = spec.shape  # Number of sp. orders, Order length in pixels
    if column_range is None:
        column_range = np.tile([0, ncol], (nord, 1))
    if orders is None:
        orders = np.arange(nord)

    # TODO: remove
    spec = spec[:-2]
    blaze = blaze[:-2]
    sigma = sigma[:-2]
    orders = orders[:-2]

    nord = len(orders)
    order_scales = np.ones(nord)

    # Reorganize input arrays, to make everything simpler
    # Memory addresses stay the same throughout the function, so any changes will also be in data
    data = np.rec.fromarrays([spec, wave, blaze, sigma], names="spec,wave,blaze,sigma")
    data = data[orders]

    if debug:
        plt.subplot(211)
        plt.title("Before")
        plt.plot(data.wave.flat, (data.spec / data.blaze).flat)

    blaze = np.clip(blaze, 1, None)

    for iord in range(nord):
        i0, i1 = column_range[iord]
        if scaling:
            scale = np.median(spec[iord, i0:i1]) / np.median(
                median_filter(blaze[iord, i0:i1], 5)
            )
        else:
            scale = 1
        blaze[iord, i0:i1] = median_filter(blaze[iord, i0:i1], 5) * scale

    # Order with largest signal, everything is scaled relative to this order
    iord0 = np.argmax(
        np.median(spec / blaze, axis=1)
    )

    # Loop from iord0 outwards, first to the top, then to the bottom
    tmp0 = chain(range(iord0, nord - 1), range(iord0, 0, -1))
    tmp1 = chain(range(iord0 + 1, nord), range(iord0 - 1, -1, -1))

    for iord0, iord in zip(tmp0, tmp1):
        beg0, end0 = column_range[iord0]
        beg1, end1 = column_range[iord]
        d0 = data[iord0, beg0:end0]
        d1 = data[iord, beg1:end1]

        # Calculate Overlap
        i0 = np.where((d0.wave >= np.min(d1.wave)) & (d0.wave <= np.max(d1.wave)))
        i1 = np.where((d1.wave >= np.min(d0.wave)) & (d1.wave <= np.max(d0.wave)))

        # Orders overlap
        if i0[0].size > 0 and i1[0].size > 0:
            tmpS0 = bezier.interpolate(d1.wave, d1.spec, d0.wave[i0])
            tmpB0 = bezier.interpolate(d1.wave, d1.blaze, d0.wave[i0])
            tmpU0 = bezier.interpolate(d1.wave, d1.sigma, d0.wave[i0])

            tmpS1 = bezier.interpolate(d0.wave, d0.spec, d1.wave[i1])
            tmpB1 = bezier.interpolate(d0.wave, d0.blaze, d1.wave[i1])
            tmpU1 = bezier.interpolate(d0.wave, d0.sigma, d1.wave[i1])

            if scaling:
                scl0 = np.sum(d0.spec[i0] / d0.blaze[i0]) / np.sum(tmpS0 / tmpB0)
                scl1 = np.sum(d1.spec[i1] / d1.blaze[i1]) / np.sum(tmpS1 / tmpB1)
                scale = np.sqrt(scl0 / scl1)
                d1.spec *= scale
                tmpS0 *= scale
                order_scales[iord] = scale

            wgt0 = np.linspace(0, 1, i0[0].size)
            wgt1 = 1 - wgt0

            d0.spec[i0] = d0.spec[i0] * wgt0 + tmpS0 * wgt1
            d0.blaze[i0] = d0.blaze[i0] * wgt0 + tmpB0 * wgt1
            d0.sigma[i0] = d0.sigma[i0] * wgt0 + tmpU0 * wgt1

            wgt0 = np.linspace(0, 1, i1[0].size)
            wgt1 = 1 - wgt0

            d1.spec[i1] = d1.spec[i1] * wgt0 + tmpS1 * wgt1
            d1.blaze[i1] = d1.blaze[i1] * wgt0 + tmpB1 * wgt1
            d1.sigma[i1] = d1.sigma[i1] * wgt0 + tmpU1 * wgt1
        else:  # Orders dont overlap
            scale0 = top(d0.spec / d0.blaze, 1, poly=True)
            d0.blaze *= scale0
            scale0 = top(d0.spec / d0.blaze, 1, poly=True)
            scale0 = np.polyfit(d0.wave, scale0, 1)

            scale1 = top(d1.spec / d1.blaze, 1, poly=True)
            scale1 = np.polyfit(d1.wave, scale1, 1)

            xx = np.linspace(np.min(d0.wave), np.max(d1.wave), 100)

            # TODO try this
            # scale = np.sum(scale0[0] * scale1[0] * xx * xx + scale0[0] * scale1[1] * xx + scale1[0] * scale0[1] * xx + scale1[1] * scale0[1])
            scale = scale0[::-1, None] * scale1[None, ::-1]
            scale = np.sum(np.polynomial.polynomial.polyval2d(xx, xx, scale)) / np.sum(
                np.polyval(scale1, xx) ** 2
            )
            d1.spec *= scale
            order_scales[iord] = scale

    # TODO: flatten data into one large spectrum
    # Problem: orders overlap

    if debug:
        plt.subplot(212)
        plt.title("After")
        plt.plot(data.wave.flat, (data.spec / data.blaze).flat)
        plt.show()

    return data.wave, data.spec, data.blaze, data.sigma


if __name__ == "__main__":
    pass
