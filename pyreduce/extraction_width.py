# -*- coding: utf-8 -*-
import logging

import matplotlib.pyplot as plt
import numpy as np

from .util import gaussfit4 as gaussfit
from .util import gaussval2 as gaussval
from .util import make_index

logger = logging.getLogger(__name__)


def estimate_extraction_width(
    img, orders, column_range, plot=False
):  # pragma: no cover
    raise NotImplemented
    nrow, ncol = img.shape
    nord, _ = orders.shape
    extraction_width = np.zeros((nord, 2), dtype=int)

    for i in range(nord):
        # first guess, half way to the next order
        # To order above
        if i < nord - 1:
            beg = max(column_range[[i, i + 1], 0])
            end = min(column_range[[i, i + 1], 1])
            x = np.arange(beg, end)
            y = np.polyval(orders[i], x)
            y_above = np.polyval(orders[i + 1], x)
            width_above = int(np.mean(y_above - y) // 2)

        # To order below
        if i > 0:
            beg = max(column_range[[i - 1, i], 0])
            end = min(column_range[[i - 1, i], 1])
            x = np.arange(beg, end)
            y = np.polyval(orders[i], x)
            y_below = np.polyval(orders[i - 1], x)
            width_below = int(np.mean(y - y_below) // 2)
        else:
            width_below = width_above

        if i == nord - 1:
            width_above = width_below

        beg, end = column_range[i]
        x = np.arange(beg, end)
        y = np.polyval(orders[i], x)

        y_int = y.astype(int)
        y_above = y_int + width_above
        y_below = y_int - width_below

        if np.ma.any(y_above >= nrow) or np.ma.any(y_below < 0):
            beg, end = np.where((y_above < nrow) & (y_below >= 0))[0][[0, -1]]

        index = make_index(y_int - width_below, y_int + width_above, beg, end, True)

        slitf = np.ma.sum(img[index], axis=1)

        # TODO fit rectangular profile

        # width = int(4 * coef[2])

        # plt.plot(p, slitf)
        # plt.plot(p, gaussval(p, *coef))
        # plt.vlines([coef[1] - width, coef[1] + width], slitf.min(), slitf.max())
        # plt.show()

        # width_below = min(width_below, width)
        # width_above = min(width_above, width)

        # index = make_index(y_int - width_below, y_int + width_above, beg, end, True)

        # plt.imshow(np.log(img[index]), aspect="auto", origin="lower")
        # plt.show()
        width = 0.5
        extraction_width[i] = [width, width]

    return extraction_width
