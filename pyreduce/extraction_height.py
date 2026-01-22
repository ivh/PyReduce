import logging

import numpy as np

from .util import make_index

logger = logging.getLogger(__name__)


def estimate_extraction_height(
    img, traces, column_range, plot=False
):  # pragma: no cover
    raise NotImplementedError
    nrow, ncol = img.shape
    ntrace, _ = traces.shape
    extraction_height = np.zeros((ntrace, 2), dtype=int)

    for i in range(ntrace):
        # first guess, half way to the next trace
        # To trace above
        if i < ntrace - 1:
            beg = max(column_range[[i, i + 1], 0])
            end = min(column_range[[i, i + 1], 1])
            x = np.arange(beg, end)
            y = np.polyval(traces[i], x)
            y_above = np.polyval(traces[i + 1], x)
            width_above = int(np.mean(y_above - y) // 2)

        # To trace below
        if i > 0:
            beg = max(column_range[[i - 1, i], 0])
            end = min(column_range[[i - 1, i], 1])
            x = np.arange(beg, end)
            y = np.polyval(traces[i], x)
            y_below = np.polyval(traces[i - 1], x)
            width_below = int(np.mean(y - y_below) // 2)
        else:
            width_below = width_above

        if i == ntrace - 1:
            width_above = width_below

        beg, end = column_range[i]
        x = np.arange(beg, end)
        y = np.polyval(traces[i], x)

        y_int = y.astype(int)
        y_above = y_int + width_above
        y_below = y_int - width_below

        if np.ma.any(y_above >= nrow) or np.ma.any(y_below < 0):
            beg, end = np.where((y_above < nrow) & (y_below >= 0))[0][[0, -1]]

        index = make_index(y_int - width_below, y_int + width_above, beg, end, True)

        np.ma.sum(img[index], axis=1)

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
        extraction_height[i] = [width, width]

    return extraction_height
