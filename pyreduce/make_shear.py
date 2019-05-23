"""
Calculate the tilt based on a reference spectrum with high SNR, e.g. Wavelength calibration image

Authors
-------
Nikolai Piskunov
Ansgar Wehrhahn

Version
--------
0.9 - NP - IDL Version
1.0 - AW - Python Version

License
-------
....
"""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from numpy.polynomial.polynomial import polyval2d
from skimage.filters import threshold_otsu

from .extract import fix_extraction_width, extend_orders, fix_column_range
from .util import make_index, gaussfit4 as gaussfit, polyfit2d


class ProgressPlot:
    def __init__(self):
        plt.ion()

    def update(self):
        pass

    def close(self):
        plt.close()
        plt.ioff()


class Curvature:
    def __init__(
        self,
        orders,
        extraction_width=0.5,
        column_range=None,
        order_range=None,
        width=9,
        threshold=10,
        fit_degree=2,
        sigma_cutoff=3,
        max_iter=None,
        mode="1D",
        plot=False,
    ):
        self.orders = orders
        self.extraction_width = extraction_width
        self.column_range = column_range
        if order_range is None:
            order_range = (0, self.nord)
        self.order_range = order_range
        self.width = width
        self.threshold = threshold
        self.fit_degree = fit_degree
        self.sigma_cutoff = sigma_cutoff
        if max_iter is None:
            max_iter = np.inf
        self.max_iter = max_iter
        self.mode = mode
        self.plot = plot

    @property
    def nord(self):
        return self.orders.shape[0]

    @property
    def n(self):
        return self.order_range[1] - self.order_range[0]

    def _fix_inputs(self, original):
        nrow, ncol = original.shape

        orders = self.orders
        extraction_width = self.extraction_width
        column_range = self.column_range

        if np.isscalar(extraction_width):
            extraction_width = np.tile([extraction_width], [self.nord, 2])
        if column_range is None:
            column_range = np.tile([0, ncol], [self.nord, 1])

        orders = extend_orders(orders, nrow)
        extraction_width = np.array(
            [extraction_width[0], *extraction_width, extraction_width[-1]]
        )
        column_range = np.array([column_range[0], *column_range, column_range[-1]])

        # Fix column range, so that all extractions are fully within the image
        extraction_width = fix_extraction_width(
            extraction_width, orders, column_range, ncol
        )
        column_range = fix_column_range(
            original, orders, extraction_width, column_range
        )

        extraction_width = extraction_width[1:-1]
        column_range = column_range[1:-1]
        orders = orders[1:-1]

        self.column_range = column_range[self.order_range[0] : self.order_range[1]]
        self.extraction_width = extraction_width[
            self.order_range[0] : self.order_range[1]
        ]
        self.orders = orders[self.order_range[0] : self.order_range[1]]
        self.order_range = (0, self.n)

    def _find_peaks(self, vec, cr):
        # This should probably be the same as in the wavelength calibration
        vec -= np.ma.min(vec)
        vec = np.ma.filled(vec, 0)
        height = np.quantile(vec, 0.1) * self.threshold
        peaks, _ = signal.find_peaks(vec, prominence=height)

        # Remove peaks at the edge
        peaks = peaks[(peaks >= self.width + 1) & (peaks < len(vec) - self.width - 1)]
        # Remove the offset, due to vec being a subset of extracted
        peaks += cr[0]
        return vec, peaks

    def _determine_curvature_single_line(self, original, peak, ycen, xwd):
        nrow, ncol = original.shape
        height = np.sum(xwd) + 1

        # look at +- width pixels around the line
        #:array of shape (2*width + 1,): indices of the pixels to the left and right of the line peak
        index_x = np.arange(-self.width, self.width + 1)
        #:array of shape (height,): stores the peak positions of the fits to each row
        xcen = np.zeros(height)
        #:array of shape (height,): indices of the rows in the order, with 0 being the central row
        xind = np.arange(-xwd[0], xwd[1] + 1)
        #:array of shape (height,): Scatter of the values within the row, to seperate in order and out of order rows
        deviation = np.zeros(height)

        segments = []

        # Extract short horizontal strip for each row in extraction width
        # Then fit a gaussian to each row, to find the center of the line
        x = peak + index_x
        x = x[(x >= 0) & (x < ncol)]
        for i, irow in enumerate(xind):
            # Trying to access values outside the image
            assert not np.any((ycen + irow)[x[0] : x[-1] + 1] >= nrow)

            # Just cutout this one row
            idx = make_index(ycen + irow, ycen + irow, x[0], x[-1] + 1)
            segment = original[idx][0]
            segments += [segment]

            if np.all(np.ma.getmask(segment)):
                # If this row is masked, this will happen
                # It will be ignored by the thresholding anyway
                xcen[i] = np.mean(x)
                deviation[i] = 0
            else:
                try:
                    coef = gaussfit(x, segment)
                    # Store line center
                    xcen[i] = coef[1]
                    # Store the variation within the row
                    deviation[i] = np.ma.std(segment)
                except RuntimeError:
                    xcen[i] = np.mean(x)
                    deviation[i] = 0

        # Seperate in order pixels from out of order pixels
        # TODO: actually we want to weight them by the slitfunction?
        idx = deviation > threshold_otsu(deviation)

        # Linear fit to slit image
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coef = np.polyfit(xind[idx], xcen[idx], 2)
        except:
            logging.warning("Could not fit curvature to line, using 0 instead.")
            coef = (0, 0)
        tilt, shear = coef[1], coef[0]

        # plot = False
        # if plot:
        #     plt.figure()
        #     for i, s in enumerate(segments):
        #         s = s - s.min()
        #         s = s / s.max() * 5
        #         plt.plot(s + i)
        #     plt.show()

        return tilt, shear

    def _fit_curvature_single_order(self, peaks, tilt, shear):

        # Make them masked arrays to avoid copying the data all the time
        # Updating the mask updates all of them (as it is not copied)
        mask = np.full(peaks.shape, False)
        peaks = np.ma.masked_array(peaks, mask=mask)
        tilt = np.ma.masked_array(tilt, mask=mask)
        shear = np.ma.masked_array(shear, mask=mask)

        # Fit a 2nd order polynomial through all individual lines
        # And discard obvious outliers
        iteration = 0
        while iteration < self.max_iter:
            iteration += 1
            coef_tilt = np.ma.polyfit(peaks, tilt, self.fit_degree)
            coef_shear = np.ma.polyfit(peaks, shear, self.fit_degree)

            diff = np.polyval(coef_tilt, peaks) - tilt
            idx1 = np.ma.abs(diff) >= np.ma.std(diff) * self.sigma_cutoff
            mask |= idx1

            diff = np.polyval(coef_shear, peaks) - shear
            idx2 = np.ma.abs(diff) >= np.ma.std(diff) * self.sigma_cutoff
            mask |= idx2

            # if no maximum iteration is given, go on forever
            if np.ma.all(~idx1) and np.ma.all(~idx2):
                break
            if np.all(mask):
                raise ValueError("Could not fit polynomial to the data")

        coef_tilt = np.ma.polyfit(peaks, tilt, self.fit_degree)
        coef_shear = np.ma.polyfit(peaks, shear, self.fit_degree)

        return coef_tilt, coef_shear, peaks

    def _determine_curvature_all_lines(self, original, extracted):
        ncol = original.shape[1]
        # Store data from all orders
        all_peaks = []
        all_tilt = []
        all_shear = []
        plot_vec = []

        for j in range(self.n):
            if self.n < 10 or j % 5 == 0:
                logging.info("Calculating tilt of order %i out of %i", j + 1, self.n)
            else:
                logging.debug("Calculating tilt of order %i out of %i", j + 1, self.n)

            cr = self.column_range[j]
            xwd = self.extraction_width[j]
            ycen = np.polyval(self.orders[j], np.arange(ncol)).astype(int)

            # Find peaks
            vec = extracted[j, cr[0] : cr[1]]
            vec, peaks = self._find_peaks(vec, cr)
            npeaks = len(peaks)
            if npeaks < self.fit_degree + 1:
                raise ValueError(
                    f"Not enough peaks found to fit a polynomial of degree {self.fit_degree}"
                )

            # Determine curvature for each line seperately
            tilt = np.zeros(npeaks)
            shear = np.zeros(npeaks)
            for ipeak, peak in enumerate(peaks):
                # TODO progress plot
                # plt.figure()
                # plt.plot(np.arange(len(vec)) + cr[0], vec)
                # plt.plot(peaks[ipeak], vec[peaks[ipeak] - cr[0]], "d")
                tilt[ipeak], shear[ipeak] = self._determine_curvature_single_line(
                    original, peak, ycen, xwd
                )

            # Store results
            all_peaks += [peaks]
            all_tilt += [tilt]
            all_shear += [shear]
            plot_vec += [vec]
        return all_peaks, all_tilt, all_shear, plot_vec

    def fit(self, peaks, tilt, shear):
        if self.mode == "1D":
            coef_tilt = np.zeros((self.n, self.fit_degree + 1))
            coef_shear = np.zeros((self.n, self.fit_degree + 1))
            for j in range(self.n):
                coef_tilt[j], coef_shear[j], _ = self._fit_curvature_single_order(
                    peaks[j], tilt[j], shear[j]
                )
        elif self.mode == "2D":
            x = np.concatenate(peaks)
            y = [np.full(len(p), i) for i, p in enumerate(peaks)]
            y = np.concatenate(y)
            z = np.concatenate(tilt)
            coef_tilt = polyfit2d(x, y, z, degree=self.fit_degree)
            z = np.concatenate(shear)
            coef_shear = polyfit2d(x, y, z, degree=self.fit_degree)
        else:
            raise ValueError(
                f"Value for 'mode' not understood. Expected one of ['1D', '2D'] but got {self.mode}"
            )

        return coef_tilt, coef_shear

    def eval(self, peaks, order, coef_tilt, coef_shear):
        if self.mode == "1D":
            tilt = np.zeros(peaks.shape)
            shear = np.zeros(peaks.shape)
            for i in np.unique(order):
                idx = order == i
                tilt[idx] = np.polyval(coef_tilt[i], peaks[idx])
                shear[idx] = np.polyval(coef_shear[i], peaks[idx])
        elif self.mode == "2D":
            tilt = polyval2d(peaks, order, coef_tilt)
            shear = polyval2d(peaks, order, coef_shear)
        else:
            raise ValueError(
                f"Value for 'mode' not understood. Expected one of ['1D', '2D'] but got {self.mode}"
            )
        return tilt, shear

    def plot_results(
        self, ncol, plot_peaks, plot_vec, plot_tilt, plot_shear, tilt_x, shear_x
    ):
        fig, axes = plt.subplots(nrows=self.n // 2, ncols=2, squeeze=False)
        fig.suptitle("Peaks")
        fig1, axes1 = plt.subplots(nrows=self.n // 2, ncols=2, squeeze=False)
        fig1.suptitle("1st Order Curvature")
        fig2, axes2 = plt.subplots(nrows=self.n // 2, ncols=2, squeeze=False)
        fig2.suptitle("2nd Order Curvature")
        plt.subplots_adjust(hspace=0)

        for j in range(self.n):
            cr = self.column_range[j]
            peaks = plot_peaks[j]
            vec = np.clip(plot_vec[j], 0, None)
            tilt = plot_tilt[j]
            shear = plot_shear[j]
            x = np.arange(cr[0], cr[1])

            order = np.full(len(x), j)
            t, s = self.eval(x, order, tilt_x, shear_x)

            # Figure Peaks found (and used)
            axes[j // 2, j % 2].plot(vec)
            axes[j // 2, j % 2].plot(peaks - cr[0], vec[peaks - cr[0]], "+")
            axes[j // 2, j % 2].set_xlim([0, ncol])
            axes[j // 2, j % 2].set_yscale("log")
            if j not in (self.n - 1, self.n - 2):
                axes[j // 2, j % 2].get_xaxis().set_ticks([])

            # Figure 1st order
            axes1[j // 2, j % 2].plot(peaks, tilt, "rx")
            axes1[j // 2, j % 2].plot(x, t)
            axes1[j // 2, j % 2].set_xlim(0, ncol)

            lower = t.min() * (0.5 if t.min() > 0 else 1.5)
            upper = t.max() * (1.5 if t.max() > 0 else 0.5)
            axes1[j // 2, j % 2].set_ylim(lower, upper)
            if j not in (self.n - 1, self.n - 2):
                axes1[j // 2, j % 2].get_xaxis().set_ticks([])

            # Figure 2nd order
            axes2[j // 2, j % 2].plot(peaks, shear, "rx")
            axes2[j // 2, j % 2].plot(x, s)
            axes2[j // 2, j % 2].set_xlim(0, ncol)

            lower = s.min() * (0.5 if s.min() > 0 else 1.5)
            upper = s.max() * (1.5 if s.max() > 0 else 0.5)
            axes2[j // 2, j % 2].set_ylim(lower, upper)
            if j not in (self.n - 1, self.n - 2):
                axes2[j // 2, j % 2].get_xaxis().set_ticks([])

        plt.show()

    def execute(self, extracted, original):
        _, ncol = original.shape
        self._fix_inputs(original)

        peaks, tilt, shear, vec = self._determine_curvature_all_lines(
            original, extracted
        )
        coef_tilt, coef_shear = self.fit(peaks, tilt, shear)

        if self.plot:
            self.plot_results(ncol, peaks, vec, tilt, shear, coef_tilt, coef_shear)

        order, peaks = np.indices(extracted.shape)
        tilt, shear = self.eval(peaks, order, coef_tilt, coef_shear)
        return tilt, shear
