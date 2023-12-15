# -*- coding: utf-8 -*-
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

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval2d
from scipy import signal
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import least_squares
from tqdm import tqdm

from .extract import fix_parameters
from .util import make_index
from .util import polyfit2d_2 as polyfit2d

logger = logging.getLogger(__name__)


class ProgressPlot:  # pragma: no cover
    def __init__(self, ncol, width, title=None):
        plt.ion()

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

        plot_title = "Curvature in each order"
        if title is not None:
            plot_title = f"{title}\n{plot_title}"
        fig.suptitle(plot_title)

        (line1,) = ax1.plot(np.arange(ncol) + 1)
        (line2,) = ax1.plot(0, 0, "d")
        ax1.set_yscale("log")

        self.ncol = ncol
        self.width = width * 2 + 1

        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.line1 = line1
        self.line2 = line2

    def update_plot1(self, vector, peak, offset=0):
        data = np.ones(self.ncol)
        data[offset : len(vector) + offset] = np.clip(vector, 1, None)
        self.line1.set_ydata(data)
        self.line2.set_xdata(peak)
        self.line2.set_ydata(data[peak])
        self.ax1.set_ylim((data.min(), data.max()))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_plot2(self, img, model, tilt, shear, peak):
        self.ax2.clear()
        self.ax3.clear()

        self.ax2.imshow(img)
        self.ax3.imshow(model)

        nrows, _ = img.shape
        middle = nrows // 2
        y = np.arange(-middle, -middle + nrows)
        x = peak + (tilt + shear * y) * y
        y += middle

        self.ax2.plot(x, y, "r")
        self.ax3.plot(x, y, "r")

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
        peak_width=1,
        fit_degree=2,
        sigma_cutoff=3,
        mode="1D",
        plot=False,
        plot_title=None,
        peak_function="gaussian",
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
        self.peak_width = peak_width
        self.fit_degree = fit_degree
        self.sigma_cutoff = sigma_cutoff
        self.mode = mode
        self.plot = plot
        self.plot_title = plot_title
        self.curv_degree = curv_degree
        self.peak_function = peak_function

        if self.mode == "1D":
            # fit degree is an integer
            if not np.isscalar(self.fit_degree):
                self.fit_degree = self.fit_degree[0]
        elif self.mode == "2D":
            # fit degree is a 2 tuple
            if np.isscalar(self.fit_degree):
                self.fit_degree = (self.fit_degree, self.fit_degree)

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
        peaks, _ = signal.find_peaks(
            vec, prominence=height, width=self.peak_width, distance=self.window_width
        )

        # Remove peaks at the edge
        peaks = peaks[
            (peaks >= self.window_width + 1)
            & (peaks < len(vec) - self.window_width - 1)
        ]
        # Remove the offset, due to vec being a subset of extracted
        peaks += cr[0]
        return vec, peaks

    def _determine_curvature_single_line(self, original, peak, ycen, ycen_int, xwd):
        """
        Fit the curvature of a single peak in the spectrum

        This is achieved by fitting a model, that consists of gaussians
        in spectrum direction, that are shifted by the curvature in each row.

        Parameters
        ----------
        original : array of shape (nrows, ncols)
            whole input image
        peak : int
            column position of the peak
        ycen : array of shape (ncols,)
            row center of the order of the peak
        xwd : 2 tuple
            extraction width above and below the order center to use

        Returns
        -------
        tilt : float
            first order curvature
        shear : float
            second order curvature
        """
        _, ncol = original.shape

        # look at +- width pixels around the line
        # Extract short horizontal strip for each row in extraction width
        # Then fit a gaussian to each row, to find the center of the line
        x = peak + np.arange(-self.window_width, self.window_width + 1)
        x = x[(x >= 0) & (x < ncol)]
        xmin, xmax = x[0], x[-1] + 1

        # Look above and below the line center
        y = np.arange(-xwd[0], xwd[1] + 1)[:, None] - ycen[xmin:xmax][None, :]

        x = x[None, :]
        idx = make_index(ycen_int - xwd[0], ycen_int + xwd[1], xmin, xmax)
        img = original[idx]
        img_compressed = np.ma.compressed(img)

        img -= np.percentile(img_compressed, 1)
        img /= np.percentile(img_compressed, 99)
        img = np.ma.clip(img, 0, 1)

        sl = np.ma.mean(img, axis=1)
        sl = sl[:, None]

        peak_func = {"gaussian": gaussian, "lorentzian": lorentzian}
        peak_func = peak_func[self.peak_function]

        def model(coef):
            A, middle, sig, *curv = coef
            mu = middle + shift(curv)
            mod = peak_func(x, A, mu, sig)
            mod *= sl
            return (mod - img).ravel()

        def model_compressed(coef):
            return np.ma.compressed(model(coef))

        A = np.nanpercentile(img_compressed, 95)
        sig = (xmax - xmin) / 4  # TODO
        if self.curv_degree == 1:
            shift = lambda curv: curv[0] * y
        elif self.curv_degree == 2:
            shift = lambda curv: (curv[0] + curv[1] * y) * y
        else:
            raise ValueError("Only curvature degrees 1 and 2 are supported")
        # res = least_squares(model, x0=[A, middle, sig, 0], loss="soft_l1", bounds=([0, xmin, 1, -10],[np.inf, xmax, xmax, 10]))
        x0 = [A, peak, sig] + [0] * self.curv_degree
        res = least_squares(
            model_compressed, x0=x0, method="trf", loss="soft_l1", f_scale=0.1
        )

        if self.curv_degree == 1:
            tilt, shear = res.x[3], 0
        elif self.curv_degree == 2:
            tilt, shear = res.x[3], res.x[4]
        else:
            tilt, shear = 0, 0

        # model = model(res.x).reshape(img.shape) + img
        # vmin = 0
        # vmax = np.max(model)

        # y = y.ravel()
        # x = res.x[1] - xmin + (tilt + shear * y) * y
        # y += xwd[0]

        # plt.subplot(121)
        # plt.imshow(img, vmin=vmin, vmax=vmax, origin="lower")
        # plt.plot(xwd[0] + ycen[xmin:xmax], "r")
        # plt.title("Input Image")
        # plt.xlabel("x [pixel]")
        # plt.ylabel("y [pixel]")

        # plt.subplot(122)
        # plt.imshow(model, vmin=vmin, vmax=vmax, origin="lower")
        # plt.plot(x, y, "r", label="curvature")
        # plt.ylim((-0.5, model.shape[0] - 0.5))
        # plt.title("Model")
        # plt.xlabel("x [pixel]")
        # plt.ylabel("y [pixel]")

        # plt.show()

        if self.plot >= 2:
            model = res.fun.reshape(img.shape) + img
            self.progress.update_plot2(img, model, tilt, shear, res.x[1] - xmin)

        return tilt, shear

    def _fit_curvature_single_order(self, peaks, tilt, shear):
        try:
            middle = np.median(tilt)
            sigma = np.percentile(tilt, (32, 68))
            sigma = middle - sigma[0], sigma[1] - middle
            mask = (tilt >= middle - 5 * sigma[0]) & (tilt <= middle + 5 * sigma[1])
            peaks, tilt, shear = peaks[mask], tilt[mask], shear[mask]

            coef_tilt = np.zeros(self.fit_degree + 1)
            res = least_squares(
                lambda coef: np.polyval(coef, peaks) - tilt,
                x0=coef_tilt,
                loss="arctan",
            )
            coef_tilt = res.x

            coef_shear = np.zeros(self.fit_degree + 1)
            res = least_squares(
                lambda coef: np.polyval(coef, peaks) - shear,
                x0=coef_shear,
                loss="arctan",
            )
            coef_shear = res.x

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
            ycen = np.polyval(self.orders[j], np.arange(ncol))
            ycen_int = ycen.astype(int)
            ycen -= ycen_int

            # Find peaks
            vec = extracted[j, cr[0] : cr[1]]
            vec, peaks = self._find_peaks(vec, cr)

            npeaks = len(peaks)

            # Determine curvature for each line seperately
            tilt = np.zeros(npeaks)
            shear = np.zeros(npeaks)
            mask = np.full(npeaks, True)
            for ipeak, peak in tqdm(
                enumerate(peaks), total=len(peaks), desc="Peak", leave=False
            ):
                if self.plot >= 2:  # pragma: no cover
                    self.progress.update_plot1(vec, peak, cr[0])
                try:
                    tilt[ipeak], shear[ipeak] = self._determine_curvature_single_line(
                        original, peak, ycen, ycen_int, xwd
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
            coef_tilt = polyfit2d(x, y, z, degree=self.fit_degree, loss="arctan")

            z = np.concatenate(shear)
            coef_shear = polyfit2d(x, y, z, degree=self.fit_degree, loss="arctan")

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

        title = "Peaks"
        if self.plot_title is not None:
            title = f"{self.plot_title}\n{title}"
        fig.suptitle(title)
        fig1, axes1 = plt.subplots(
            nrows=self.n // 2 + self.n % 2, ncols=2, squeeze=False
        )

        title = "1st Order Curvature"
        if self.plot_title is not None:
            title = f"{self.plot_title}\n{title}"
        fig1.suptitle(title)
        fig2, axes2 = plt.subplots(
            nrows=self.n // 2 + self.n % 2, ncols=2, squeeze=False
        )

        title = "2nd Order Curvature"
        if self.plot_title is not None:
            title = f"{self.plot_title}\n{title}"
        fig2.suptitle(title)
        plt.subplots_adjust(hspace=0)

        def trim_axs(axs, N):
            """little helper to massage the axs list to have correct length..."""
            axs = axs.flat
            for ax in axs[N:]:
                ax.remove()
            return axs[:N]

        t, s = [None for _ in range(self.n)], [None for _ in range(self.n)]
        for j in range(self.n):
            cr = self.column_range[j]
            x = np.arange(cr[0], cr[1])
            order = np.full(len(x), j)
            t[j], s[j] = self.eval(x, order, tilt_x, shear_x)

        t_lower = min(t.min() * (0.5 if t.min() > 0 else 1.5) for t in t)
        t_upper = max(t.max() * (1.5 if t.max() > 0 else 0.5) for t in t)

        s_lower = min(s.min() * (0.5 if s.min() > 0 else 1.5) for s in s)
        s_upper = max(s.max() * (1.5 if s.max() > 0 else 0.5) for s in s)

        for j in range(self.n):
            cr = self.column_range[j]
            peaks = plot_peaks[j]
            vec = np.clip(plot_vec[j], 0, None)
            tilt = plot_tilt[j]
            shear = plot_shear[j]
            x = np.arange(cr[0], cr[1])
            # Figure Peaks found (and used)
            axes[j // 2, j % 2].plot(np.arange(cr[0], cr[1]), vec)
            axes[j // 2, j % 2].plot(peaks, vec[peaks - cr[0]], "X")
            axes[j // 2, j % 2].set_xlim([0, ncol])
            # axes[j // 2, j % 2].set_yscale("log")
            if j not in (self.n - 1, self.n - 2):
                axes[j // 2, j % 2].get_xaxis().set_ticks([])

            # Figure 1st order
            axes1[j // 2, j % 2].plot(peaks, tilt, "rX")
            axes1[j // 2, j % 2].plot(x, t[j])
            axes1[j // 2, j % 2].set_xlim(0, ncol)

            axes1[j // 2, j % 2].set_ylim(t_lower, t_upper)
            if j not in (self.n - 1, self.n - 2):
                axes1[j // 2, j % 2].get_xaxis().set_ticks([])
            else:
                axes1[j // 2, j % 2].set_xlabel("x [pixel]")
            if j == self.n // 2 + 1:
                axes1[j // 2, j % 2].set_ylabel("tilt [pixel/pixel]")

            # Figure 2nd order
            axes2[j // 2, j % 2].plot(peaks, shear, "rX")
            axes2[j // 2, j % 2].plot(x, s[j])
            axes2[j // 2, j % 2].set_xlim(0, ncol)

            axes2[j // 2, j % 2].set_ylim(s_lower, s_upper)
            if j not in (self.n - 1, self.n - 2):
                axes2[j // 2, j % 2].get_xaxis().set_ticks([])
            else:
                axes2[j // 2, j % 2].set_xlabel("x [pixel]")
            if j == self.n // 2 + 1:
                axes2[j // 2, j % 2].set_ylabel("shear [pixel/pixel**2]")

        axes1 = trim_axs(axes1, self.n)
        axes2 = trim_axs(axes2, self.n)

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
        if self.plot_title is not None:
            plt.title(self.plot_title)
        plt.xlabel("x [pixel]")
        plt.ylabel("order")
        plt.show()

    def execute(self, extracted, original):
        logger.info("Determining the Slit Curvature")

        _, ncol = original.shape

        self._fix_inputs(original)

        if self.plot >= 2:  # pragma: no cover
            self.progress = ProgressPlot(ncol, self.window_width, title=self.plot_title)

        peaks, tilt, shear, vec = self._determine_curvature_all_lines(
            original, extracted
        )

        coef_tilt, coef_shear = self.fit(peaks, tilt, shear)

        if self.plot >= 2:  # pragma: no cover
            self.progress.close()

        if self.plot:  # pragma: no cover
            self.plot_results(ncol, peaks, vec, tilt, shear, coef_tilt, coef_shear)

        iorder, ipeaks = np.indices(extracted.shape)
        tilt, shear = self.eval(ipeaks, iorder, coef_tilt, coef_shear)

        if self.plot:  # pragma: no cover
            self.plot_comparison(original, tilt, shear, peaks)

        return tilt, shear


# TODO allow other line shapes
def gaussian(x, A, mu, sig):
    """
    A: height
    mu: offset from central line
    sig: standard deviation
    """
    return A * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def lorentzian(x, A, x0, mu):
    """
    A: height
    x0: offset from central line
    mu: width of lorentzian
    """
    return A * mu / ((x - x0) ** 2 + 0.25 * mu ** 2)
