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
from tqdm import tqdm

from numpy.polynomial.polynomial import polyval2d
from scipy.optimize import least_squares
from skimage.filters import threshold_otsu

from .extract import fix_parameters
from .util import make_index, gaussfit4 as gaussfit, polyfit2d

logger = logging.getLogger(__name__)


class ProgressPlot:  # pragma: no cover
    def __init__(self, ncol, width, height):
        plt.ion()

        fig, (ax1, ax2) = plt.subplots(ncols=2)

        fig.suptitle("Curvature in each order")

        line1, = ax1.plot(np.arange(ncol) + 1)
        line2, = ax1.plot(0, 0, "d")
        ax1.set_yscale("log")

        lines = [None] * height
        for i in range(height):
            lines[i], = ax2.plot(
                np.arange(-width, width + 1), np.arange(-width, width + 1)
            )

        line3, = ax2.plot(np.arange(height), "r--")
        line4, = ax2.plot(np.arange(height), "rx")
        ax2.set_xlim((-width, width))
        ax2.set_ylim((0, height + 5))

        self.ncol = ncol
        self.width = width * 2 + 1
        self.height = height

        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.line1 = line1
        self.line2 = line2
        self.line3 = line3
        self.line4 = line4
        self.lines = lines

    def update_plot1(self, vector, peak, offset=0):
        data = np.ones(self.ncol)
        data[offset : len(vector) + offset] = np.clip(vector, 1, None)
        self.line1.set_ydata(data)
        self.line2.set_xdata(peak)
        self.line2.set_ydata(data[peak])
        self.ax1.set_ylim((data.min(), data.max()))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_plot2(self, segments, tilt, shear, positions, values):
        l4_y = np.full(self.height, np.nan)
        for i, (s, v) in enumerate(zip(segments, values)):
            s, v = s - s.min(), v - s.min()
            s, v = s / s.max() * 5, v / s.max() * 5
            l4_y[i] = v + i
            self.lines[i].set_ydata(s + i)
        for i in range(len(segments), self.height):
            self.lines[i].set_ydata(np.full(self.width, np.nan))

        y = np.arange(0, self.height) - self.height // 2
        x = np.polyval((shear, tilt, 0), y)
        y += np.arange(self.height)
        y += self.height // 2
        self.line3.set_xdata(x)
        self.line3.set_ydata(y)

        if positions.size > self.height:
            positions = positions[: self.height]
        elif positions.size < self.height:
            positions = np.concatenate(
                (positions, np.zeros(self.height - positions.size))
            )

        self.line4.set_xdata(positions)
        self.line4.set_ydata(l4_y)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

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
        window_width=9,
        peak_threshold=10,
        fit_degree=2,
        sigma_cutoff=3,
        mode="1D",
        plot=False,
        curv_degree=2,
    ):
        self.orders = orders
        self.extraction_width = extraction_width
        self.column_range = column_range
        if order_range is None:
            order_range = (0, self.nord)
        self.order_range = order_range
        self.window_width = window_width
        self.threshold = peak_threshold
        self.fit_degree = fit_degree
        self.sigma_cutoff = sigma_cutoff
        self.mode = mode
        self.plot = plot
        self.curv_degree = curv_degree

    @property
    def nord(self):
        return self.orders.shape[0]

    @property
    def n(self):
        return self.order_range[1] - self.order_range[0]

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in ["1D", "2D"]:
            raise ValueError(
                f"Value for 'mode' not understood. Expected one of ['1D', '2D'] but got {value}"
            )
        self._mode = value

    def _fix_inputs(self, original):
        orders = self.orders
        extraction_width = self.extraction_width
        column_range = self.column_range

        nrow, ncol = original.shape
        nord = len(orders)

        extraction_width, column_range, orders = fix_parameters(
            extraction_width, column_range, orders, nrow, ncol, nord
        )

        self.column_range = column_range[self.order_range[0] : self.order_range[1]]
        self.extraction_width = extraction_width[
            self.order_range[0] : self.order_range[1]
        ]
        self.orders = orders[self.order_range[0] : self.order_range[1]]
        self.order_range = (0, self.n)

    def _find_peaks(self, vec, cr):
        # This should probably be the same as in the wavelength calibration
        vec -= np.ma.median(vec)
        vec = np.ma.filled(vec, 0)
        height = np.percentile(vec, 68) * self.threshold
        peaks, _ = signal.find_peaks(vec, prominence=height)

        # Remove peaks at the edge
        peaks = peaks[(peaks >= self.window_width + 1) & (peaks < len(vec) - self.window_width - 1)]
        # Remove the offset, due to vec being a subset of extracted
        peaks += cr[0]
        return vec, peaks

    def _determine_curvature_single_line(self, original, peak, ycen, xwd):
        nrow, ncol = original.shape
        height = np.sum(xwd) + 1

        # look at +- width pixels around the line
        #:array of shape (2*width + 1,): indices of the pixels to the left and right of the line peak
        index_x = np.arange(-self.window_width, self.window_width + 1)
        #:array of shape (height,): stores the peak positions of the fits to each row
        xcen = np.zeros(height)
        vcen = np.zeros(height)
        wcen = np.zeros(height)
        #:array of shape (height,): indices of the rows in the order, with 0 being the central row
        xind = np.arange(-xwd[0], xwd[1] + 1)
        #:array of shape (height,): Scatter of the values within the row, to seperate in order and out of order rows
        deviation = np.zeros(height)

        segments = []

        # Extract short horizontal strip for each row in extraction width
        # Then fit a gaussian to each row, to find the center of the line
        x = peak + index_x
        x = x[(x >= 0) & (x < ncol)]
        xmin, xmax = x[0], x[-1] + 1
        x = np.ma.masked_array(x)
        for i, irow in enumerate(xind):
            # Trying to access values outside the image
            # assert not np.any((ycen + irow)[xmin:xmax] >= nrow)

            # Just cutout this one row
            idx = make_index(ycen + irow, ycen + irow, xmin, xmax)
            segment = original[idx][0]
            segment -= segment.min()
            segments += [segment]

            try:
                x.mask = np.ma.getmaskarray(segment)
                coef = gaussfit(x, segment)
                # Store line center
                xcen[i] = coef[1]
                wcen[i] = coef[2]
                vcen[i] = coef[0] + coef[3]
                # Store the variation within the row
                deviation[i] = np.ma.std(segment)
            except Exception as e:
                xcen[i] = peak + self.window_width
                deviation[i] = 0

        # Seperate in order pixels from out of order pixels
        # TODO: actually we want to weight them by the slitfunction?
        # If any of this fails, we will just ignore this line
        try:
            idx = deviation > threshold_otsu(deviation)
        except ValueError:
            raise RuntimeError

        # Linear fit to slit image
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = np.sqrt(1 / wcen[idx])
            try:
                coef = np.polyfit(xind[idx], xcen[idx], self.curv_degree, w=w)
            except ValueError:
                # Polyfit failed for some reason
                raise RuntimeError


        if self.curv_degree == 1:
            tilt, shear = coef[0], 0
        elif self.curv_degree == 2:
            tilt, shear = coef[1], coef[0]
        else:
            raise ValueError("Only curvature degrees 1 and 2 are supported")

        if self.plot >= 2:  # pragma: no cover
            self.progress.update_plot2(segments, tilt, shear, xcen - peak, vcen)
        return tilt, shear

    def _fit_curvature_single_order(self, peaks, tilt, shear):
        try:
            middle = np.median(tilt)
            sigma = np.percentile(tilt, (32, 68))
            sigma = middle - sigma[0], sigma[1] - middle
            mask = (tilt >= middle - 5 * sigma[0]) & (tilt <= middle + 5 * sigma[1])
            peaks, tilt, shear = peaks[mask], tilt[mask], shear[mask]

            coef_tilt = np.polyfit(peaks, tilt, self.fit_degree)
            coef_shear = np.polyfit(peaks, shear, self.fit_degree)
        except:
            logger.error(
                "Could not fit the curvature of this order. Using no curvature instead"
            )
            coef_tilt = np.zeros(self.fit_degree + 1)
            coef_shear = np.zeros(self.fit_degree + 1)

        return coef_tilt, coef_shear, peaks

    def _determine_curvature_all_lines(self, original, extracted):
        ncol = original.shape[1]
        # Store data from all orders
        all_peaks = []
        all_tilt = []
        all_shear = []
        plot_vec = []

        for j in tqdm(range(self.n), desc="Order"):
            logger.debug("Calculating tilt of order %i out of %i", j + 1, self.n)

            cr = self.column_range[j]
            xwd = self.extraction_width[j]
            ycen = np.polyval(self.orders[j], np.arange(ncol)).astype(int)

            # Find peaks
            vec = extracted[j, cr[0] : cr[1]]
            vec, peaks = self._find_peaks(vec, cr)
            
            npeaks = len(peaks)

            # Determine curvature for each line seperately
            tilt = np.zeros(npeaks)
            shear = np.zeros(npeaks)
            mask = np.full(npeaks, True)
            for ipeak, peak in tqdm(enumerate(peaks), total=len(peaks), desc="Peak", leave=False):
                if self.plot >= 2:  # pragma: no cover
                    self.progress.update_plot1(vec, peak, cr[0])
                try:
                    tilt[ipeak], shear[ipeak] = self._determine_curvature_single_line(
                        original, peak, ycen, xwd
                    )
                except RuntimeError:  # pragma: no cover
                    mask[ipeak] = False

            # Store results
            all_peaks += [peaks[mask]]
            all_tilt += [tilt[mask]]
            all_shear += [shear[mask]]
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
        return tilt, shear

    def plot_results(
        self, ncol, plot_peaks, plot_vec, plot_tilt, plot_shear, tilt_x, shear_x
    ):  # pragma: no cover
        fig, axes = plt.subplots(nrows=self.n // 2 + self.n % 2, ncols=2, squeeze=False)
        fig.suptitle("Peaks")
        fig1, axes1 = plt.subplots(
            nrows=self.n // 2 + self.n % 2, ncols=2, squeeze=False
        )
        fig1.suptitle("1st Order Curvature")
        fig2, axes2 = plt.subplots(
            nrows=self.n // 2 + self.n % 2, ncols=2, squeeze=False
        )
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
            axes[j // 2, j % 2].plot(np.arange(cr[0], cr[1]), vec)
            axes[j // 2, j % 2].plot(peaks, vec[peaks - cr[0]], "X")
            axes[j // 2, j % 2].set_xlim([0, ncol])
            axes[j // 2, j % 2].set_yscale("log")
            if j not in (self.n - 1, self.n - 2):
                axes[j // 2, j % 2].get_xaxis().set_ticks([])

            # Figure 1st order
            axes1[j // 2, j % 2].plot(peaks, tilt, "rX")
            axes1[j // 2, j % 2].plot(x, t)
            axes1[j // 2, j % 2].set_xlim(0, ncol)

            lower = t.min() * (0.5 if t.min() > 0 else 1.5)
            upper = t.max() * (1.5 if t.max() > 0 else 0.5)
            axes1[j // 2, j % 2].set_ylim(lower, upper)
            if j not in (self.n - 1, self.n - 2):
                axes1[j // 2, j % 2].get_xaxis().set_ticks([])

            # Figure 2nd order
            axes2[j // 2, j % 2].plot(peaks, shear, "rX")
            axes2[j // 2, j % 2].plot(x, s)
            axes2[j // 2, j % 2].set_xlim(0, ncol)

            lower = s.min() * (0.5 if s.min() > 0 else 1.5)
            upper = s.max() * (1.5 if s.max() > 0 else 0.5)
            axes2[j // 2, j % 2].set_ylim(lower, upper)
            if j not in (self.n - 1, self.n - 2):
                axes2[j // 2, j % 2].get_xaxis().set_ticks([])

        plt.show()

    def plot_comparison(self, original, tilt, shear, peaks):  # pragma: no cover
        _, ncol = original.shape
        output = np.zeros((np.sum(self.extraction_width) + self.nord, ncol))
        pos = [0]
        x = np.arange(ncol)
        for i in range(self.nord):
            ycen = np.polyval(self.orders[i], x)
            yb = ycen - self.extraction_width[i, 0]
            yt = ycen + self.extraction_width[i, 1]
            xl, xr = self.column_range[i]
            index = make_index(yb, yt, xl, xr)
            yl = pos[i]
            yr = pos[i] + index[0].shape[0]
            output[yl:yr, xl:xr] = original[index]
            pos += [yr]

        vmin, vmax = np.percentile(output[output != 0], (5, 95))
        plt.imshow(output, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")

        for i in range(self.nord):
            for p in peaks[i]:
                ew = self.extraction_width[i]
                x = np.zeros(ew[0] + ew[1] + 1)
                y = np.arange(-ew[0], ew[1] + 1)
                for j, yt in enumerate(y):
                    x[j] = p + yt * tilt[i, p] + yt ** 2 * shear[i, p]
                y += pos[i] + ew[0]
                plt.plot(x, y, "r")

        locs = np.sum(self.extraction_width, axis=1) + 1
        locs = np.array([0, *np.cumsum(locs)[:-1]])
        locs[:-1] += (np.diff(locs) * 0.5).astype(int)
        locs[-1] += ((output.shape[0] - locs[-1]) * 0.5).astype(int)

        plt.yticks(locs, range(len(locs)))
        plt.xlabel("x [pixel]")
        plt.ylabel("order")
        plt.show()

    def execute(self, extracted, original):
        logger.info("Determining the Slit Curvature")

        _, ncol = original.shape

        self._fix_inputs(original)

        if self.plot >= 2:  # pragma: no cover
            height = np.sum(self.extraction_width, axis=1).max() + 1
            self.progress = ProgressPlot(ncol, self.window_width, height)

        peaks, tilt, shear, vec = self._determine_curvature_all_lines(
            original, extracted
        )

        coef_tilt, coef_shear = self.fit(peaks, tilt, shear)

        if self.plot >= 2: # pragma: no cover
            self.progress.close()

        if self.plot:  # pragma: no cover
            self.plot_results(ncol, peaks, vec, tilt, shear, coef_tilt, coef_shear)

        iorder, ipeaks = np.indices(extracted.shape)
        tilt, shear = self.eval(ipeaks, iorder, coef_tilt, coef_shear)

        if self.plot:  # pragma: no cover
            self.plot_comparison(original, tilt, shear, peaks)

        return tilt, shear
