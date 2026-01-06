"""
Calculate slit curvature based on a reference spectrum with high SNR, e.g. Wavelength calibration image

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
from scipy.optimize import least_squares
from tqdm import tqdm

from . import util
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

    def update_plot2(self, img, model, p1, p2, peak):
        self.ax2.clear()
        self.ax3.clear()

        self.ax2.imshow(img)
        self.ax3.imshow(model)

        nrows, _ = img.shape
        middle = nrows // 2
        y = np.arange(-middle, -middle + nrows)
        x = peak + (p1 + p2 * y) * y
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
        curve_height=0.5,
        extraction_height=0.2,
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
        curve_degree=2,
    ):
        self.orders = orders
        self.curve_height = curve_height
        self.extraction_height = extraction_height
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
        self.curve_degree = curve_degree
        self.peak_function = peak_function

        if self.curve_degree not in (1, 2):
            raise ValueError("Only curvature degrees 1 and 2 are supported")

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
        curve_height = self.curve_height
        extraction_height = self.extraction_height
        column_range = self.column_range

        nrow, ncol = original.shape
        nord = len(orders)

        curve_height, column_range, orders = fix_parameters(
            curve_height, column_range, orders, nrow, ncol, nord
        )
        extraction_height, _, _ = fix_parameters(
            extraction_height, column_range, orders, nrow, ncol, nord
        )

        self.column_range = column_range[self.order_range[0] : self.order_range[1]]
        self.curve_height = curve_height[self.order_range[0] : self.order_range[1]]
        self.extraction_height = extraction_height[
            self.order_range[0] : self.order_range[1]
        ]
        self.orders = orders[self.order_range[0] : self.order_range[1]]
        self.order_range = (0, self.nord)

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

    def _extract_offset_spectra(self, original, order_idx):
        """Extract N spectra at different y-offsets from the trace.

        Parameters
        ----------
        original : array of shape (nrow, ncol)
            Original image
        order_idx : int
            Index of the order in self.orders

        Returns
        -------
        spectra : array of shape (n_offsets, ncol)
            Extracted spectra at each offset
        offsets : array of shape (n_offsets,)
            Y-offset of each spectrum center relative to trace
        """
        nrow, ncol = original.shape
        cr = self.column_range[order_idx]
        xwd = self.extraction_height[order_idx]
        curve_xwd = self.curve_height[order_idx]

        # Number of spectra to extract
        total_extraction = xwd[0] + xwd[1]
        total_curve = curve_xwd[0] + curve_xwd[1]
        n_offsets = max(1, int(total_curve // total_extraction))

        # Compute offset centers, symmetric around trace
        offsets = (np.arange(n_offsets) - (n_offsets - 1) / 2) * total_extraction

        # Get trace position
        x = np.arange(ncol)
        ycen = np.polyval(self.orders[order_idx], x)
        ycen_int = ycen.astype(int)

        spectra = np.ma.zeros((n_offsets, ncol))
        spectra[:, :] = np.ma.masked

        for i, offset in enumerate(offsets):
            # Y bounds for this offset spectrum
            yb = ycen_int + int(offset) - xwd[0]
            yt = ycen_int + int(offset) + xwd[1]

            # Check bounds
            if np.any(yb < 0) or np.any(yt >= nrow):
                continue

            # Extract and collapse
            idx = make_index(yb, yt, cr[0], cr[1])
            img_slice = original[idx]
            spectra[i, cr[0] : cr[1]] = np.ma.median(img_slice, axis=0)

        return spectra, offsets

    def _fit_subpixel_peak(self, vec, peak_col):
        """Fit a peak function to find sub-pixel peak position.

        Parameters
        ----------
        vec : array
            1D spectrum
        peak_col : int
            Approximate column of the peak

        Returns
        -------
        position : float
            Sub-pixel peak position, or NaN if fit fails
        """
        hw = self.window_width
        xmin = max(0, peak_col - hw)
        xmax = min(len(vec), peak_col + hw + 1)

        x = np.arange(xmin, xmax)
        y = np.ma.filled(vec[xmin:xmax], 0)

        if np.sum(y > 0) < 3:
            return np.nan

        peak_func = {"gaussian": gaussian, "lorentzian": lorentzian}
        func = peak_func[self.peak_function]

        try:
            A = np.max(y)
            mu = peak_col
            sig = hw / 2

            def residual(coef):
                return func(x, coef[0], coef[1], coef[2]) - y

            res = least_squares(residual, x0=[A, mu, sig], method="lm")
            return res.x[1]
        except Exception:
            return np.nan

    def _find_peaks_in_spectra(self, spectra, offsets, cr):
        """Find peaks in each spectrum and track across offsets.

        Parameters
        ----------
        spectra : array of shape (n_offsets, ncol)
            Extracted spectra at different offsets
        offsets : array of shape (n_offsets,)
            Y-offset of each spectrum
        cr : array of shape (2,)
            Column range for this order

        Returns
        -------
        peaks : array
            Peak columns (from middle spectrum)
        positions : array of shape (n_peaks, n_offsets)
            Sub-pixel x-position of each peak at each offset
        """
        n_offsets = len(offsets)
        mid_idx = n_offsets // 2

        # Find peaks in the middle spectrum (closest to trace, highest S/N)
        mid_spectrum = spectra[mid_idx, cr[0] : cr[1]]
        _, peaks = self._find_peaks(mid_spectrum, cr)

        if len(peaks) == 0:
            return np.array([]), np.array([]).reshape(0, n_offsets)

        # Track each peak across all spectra
        positions = np.full((len(peaks), n_offsets), np.nan)

        for i_offset in range(n_offsets):
            spec = spectra[i_offset]
            if np.ma.count(spec[cr[0] : cr[1]]) == 0:
                continue

            for i_peak, peak in enumerate(peaks):
                # Search for the peak in a window around expected position
                search_min = max(cr[0], peak - self.window_width)
                search_max = min(cr[1], peak + self.window_width + 1)

                window = spec[search_min:search_max]
                if np.ma.count(window) == 0:
                    continue

                # Find local maximum
                local_max = search_min + np.ma.argmax(window)

                # Fit sub-pixel position
                positions[i_peak, i_offset] = self._fit_subpixel_peak(spec, local_max)

        return peaks, positions

    def _fit_curvature_from_positions(self, peaks, positions, offsets):
        """Fit p1 and p2 from peak position vs y-offset.

        For each peak: x(y) = x0 + p1*y + p2*y^2

        Parameters
        ----------
        peaks : array
            Peak columns
        positions : array of shape (n_peaks, n_offsets)
            X-position of each peak at each offset
        offsets : array of shape (n_offsets,)
            Y-offset of each spectrum

        Returns
        -------
        p1 : array of shape (n_peaks,)
            Linear curvature coefficient for each peak
        p2 : array of shape (n_peaks,)
            Quadratic curvature coefficient for each peak
        """
        n_peaks = len(peaks)
        p1 = np.zeros(n_peaks)
        p2 = np.zeros(n_peaks)

        for i in range(n_peaks):
            pos = positions[i]
            valid = ~np.isnan(pos)

            if np.sum(valid) < 2:
                continue

            y = offsets[valid]
            x = pos[valid]

            # Subtract mean x to get relative shift
            x0 = np.mean(x)
            dx = x - x0

            if self.curve_degree == 1:
                # Linear fit: dx = p1 * y
                if np.sum(valid) >= 2:
                    try:
                        coef = np.polyfit(y, dx, 1)
                        p1[i] = coef[0]
                    except Exception:
                        pass
            elif self.curve_degree == 2:
                # Quadratic fit: dx = p2 * y^2 + p1 * y
                if np.sum(valid) >= 3:
                    try:
                        coef = np.polyfit(y, dx, 2)
                        p2[i] = coef[0]
                        p1[i] = coef[1]
                    except Exception:
                        pass

        return p1, p2

    def _fit_curvature_single_order(self, peaks, p1, p2):
        try:
            middle = np.median(p1)
            sigma = np.percentile(p1, (32, 68))
            sigma = middle - sigma[0], sigma[1] - middle
            mask = (p1 >= middle - 5 * sigma[0]) & (p1 <= middle + 5 * sigma[1])
            peaks, p1, p2 = peaks[mask], p1[mask], p2[mask]

            coef_p1 = np.zeros(self.fit_degree + 1)
            res = least_squares(
                lambda coef: np.polyval(coef, peaks) - p1,
                x0=coef_p1,
                loss="arctan",
            )
            coef_p1 = res.x

            coef_p2 = np.zeros(self.fit_degree + 1)
            res = least_squares(
                lambda coef: np.polyval(coef, peaks) - p2,
                x0=coef_p2,
                loss="arctan",
            )
            coef_p2 = res.x

        except:
            logger.error(
                "Could not fit the curvature of this order. Using no curvature instead"
            )
            coef_p1 = np.zeros(self.fit_degree + 1)
            coef_p2 = np.zeros(self.fit_degree + 1)

        return coef_p1, coef_p2, peaks

    def _determine_curvature_all_lines(self, original):
        """Determine curvature for all lines using row-tracking method.

        Extracts N spectra at different y-offsets, finds peaks in each,
        and fits curvature from how peak positions shift with y-offset.

        Parameters
        ----------
        original : array of shape (nrow, ncol)
            Original image

        Returns
        -------
        all_peaks : list of arrays
            Peak columns for each order
        all_p1 : list of arrays
            Tilt values for each peak in each order
        all_p2 : list of arrays
            Shear values for each peak in each order
        plot_vec : list of arrays
            Middle spectrum for each order (for plotting)
        """
        all_peaks = []
        all_p1 = []
        all_p2 = []
        plot_vec = []

        for j in tqdm(range(self.n), desc="Order"):
            logger.debug("Calculating curvature of order %i out of %i", j + 1, self.n)

            cr = self.column_range[j]

            # Extract spectra at different y-offsets
            spectra, offsets = self._extract_offset_spectra(original, j)

            # Find peaks and track across offsets
            peaks, positions = self._find_peaks_in_spectra(spectra, offsets, cr)

            # For plotting, use middle spectrum
            mid_idx = len(offsets) // 2
            vec = spectra[mid_idx, cr[0] : cr[1]]
            vec = np.ma.filled(vec - np.ma.median(vec), 0)
            vec = np.clip(vec, 0, None)

            if len(peaks) == 0:
                all_peaks.append(np.array([]))
                all_p1.append(np.array([]))
                all_p2.append(np.array([]))
                plot_vec.append(vec)
                continue

            # Fit curvature from peak positions
            p1, p2 = self._fit_curvature_from_positions(peaks, positions, offsets)

            if self.plot >= 2:  # pragma: no cover
                for peak in peaks:
                    self.progress.update_plot1(vec, peak, cr[0])

            all_peaks.append(peaks)
            all_p1.append(p1)
            all_p2.append(p2)
            plot_vec.append(vec)

        return all_peaks, all_p1, all_p2, plot_vec

    def fit(self, peaks, p1, p2):
        if self.mode == "1D":
            coef_p1 = np.zeros((self.n, self.fit_degree + 1))
            coef_p2 = np.zeros((self.n, self.fit_degree + 1))
            for j in range(self.n):
                coef_p1[j], coef_p2[j], _ = self._fit_curvature_single_order(
                    peaks[j], p1[j], p2[j]
                )
        elif self.mode == "2D":
            x = np.concatenate(peaks)
            y = [np.full(len(p), i) for i, p in enumerate(peaks)]
            y = np.concatenate(y)
            z = np.concatenate(p1)
            coef_p1 = polyfit2d(x, y, z, degree=self.fit_degree, loss="arctan")

            z = np.concatenate(p2)
            coef_p2 = polyfit2d(x, y, z, degree=self.fit_degree, loss="arctan")

        return coef_p1, coef_p2

    def eval(self, peaks, order, coef_p1, coef_p2):
        if self.mode == "1D":
            p1 = np.zeros(peaks.shape)
            p2 = np.zeros(peaks.shape)
            for i in np.unique(order):
                idx = order == i
                p1[idx] = np.polyval(coef_p1[i], peaks[idx])
                p2[idx] = np.polyval(coef_p2[i], peaks[idx])
        elif self.mode == "2D":
            p1 = polyval2d(peaks, order, coef_p1)
            p2 = polyval2d(peaks, order, coef_p2)
        return p1, p2

    def plot_results(
        self, ncol, plot_peaks, plot_vec, plot_p1, plot_p2, p1_x, p2_x
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
            t[j], s[j] = self.eval(x, order, p1_x, p2_x)

        t_lower = min(t.min() * (0.5 if t.min() > 0 else 1.5) for t in t)
        t_upper = max(t.max() * (1.5 if t.max() > 0 else 0.5) for t in t)

        s_lower = min(s.min() * (0.5 if s.min() > 0 else 1.5) for s in s)
        s_upper = max(s.max() * (1.5 if s.max() > 0 else 0.5) for s in s)

        for j in range(self.n):
            cr = self.column_range[j]
            peaks = plot_peaks[j]
            vec = np.clip(plot_vec[j], 0, None)
            p1 = plot_p1[j]
            p2 = plot_p2[j]
            x = np.arange(cr[0], cr[1])
            # Figure Peaks found (and used)
            axes[j // 2, j % 2].plot(np.arange(cr[0], cr[1]), vec)
            axes[j // 2, j % 2].plot(peaks, vec[peaks - cr[0]], "X")
            axes[j // 2, j % 2].set_xlim([0, ncol])
            # axes[j // 2, j % 2].set_yscale("log")
            if j not in (self.n - 1, self.n - 2):
                axes[j // 2, j % 2].get_xaxis().set_ticks([])

            # Figure 1st order
            axes1[j // 2, j % 2].plot(peaks, p1, "rX")
            axes1[j // 2, j % 2].plot(x, t[j])
            axes1[j // 2, j % 2].set_xlim(0, ncol)

            axes1[j // 2, j % 2].set_ylim(t_lower, t_upper)
            if j not in (self.n - 1, self.n - 2):
                axes1[j // 2, j % 2].get_xaxis().set_ticks([])
            else:
                axes1[j // 2, j % 2].set_xlabel("x [pixel]")
            if j == self.n // 2 + 1:
                axes1[j // 2, j % 2].set_ylabel("p1 [pixel/pixel]")

            # Figure 2nd order
            axes2[j // 2, j % 2].plot(peaks, p2, "rX")
            axes2[j // 2, j % 2].plot(x, s[j])
            axes2[j // 2, j % 2].set_xlim(0, ncol)

            axes2[j // 2, j % 2].set_ylim(s_lower, s_upper)
            if j not in (self.n - 1, self.n - 2):
                axes2[j // 2, j % 2].get_xaxis().set_ticks([])
            else:
                axes2[j // 2, j % 2].set_xlabel("x [pixel]")
            if j == self.n // 2 + 1:
                axes2[j // 2, j % 2].set_ylabel("p2 [pixel/pixel**2]")

        axes1 = trim_axs(axes1, self.n)
        axes2 = trim_axs(axes2, self.n)

        util.show_or_save("curvature_fit")

    def plot_comparison(self, original, p1, p2, peaks):  # pragma: no cover
        _, ncol = original.shape
        output = np.zeros((np.sum(self.curve_height) + self.nord, ncol))
        pos = [0]
        x = np.arange(ncol)
        for i in range(self.nord):
            ycen = np.polyval(self.orders[i], x)
            yb = ycen - self.curve_height[i, 0]
            yt = ycen + self.curve_height[i, 1]
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
                ew = self.curve_height[i]
                x = np.zeros(ew[0] + ew[1] + 1)
                y = np.arange(-ew[0], ew[1] + 1)
                for j, yt in enumerate(y):
                    x[j] = p + yt * p1[i, p] + yt**2 * p2[i, p]
                y += pos[i] + ew[0]
                plt.plot(x, y, "r")

        locs = np.sum(self.curve_height, axis=1) + 1
        locs = np.array([0, *np.cumsum(locs)[:-1]])
        locs[:-1] += (np.diff(locs) * 0.5).astype(int)
        locs[-1] += ((output.shape[0] - locs[-1]) * 0.5).astype(int)

        plt.yticks(locs, range(len(locs)))
        if self.plot_title is not None:
            plt.title(self.plot_title)
        plt.xlabel("x [pixel]")
        plt.ylabel("order")
        util.show_or_save("curvature_comparison")

    def execute(self, original):
        """Execute curvature determination using row-tracking method.

        Parameters
        ----------
        original : array of shape (nrow, ncol)
            Original image

        Returns
        -------
        p1 : array of shape (nord, ncol)
            First order slit curvature at each point
        p2 : array of shape (nord, ncol)
            Second order slit curvature at each point
        """
        logger.info("Determining the Slit Curvature")

        _, ncol = original.shape

        self._fix_inputs(original)

        if self.plot >= 2:  # pragma: no cover
            self.progress = ProgressPlot(ncol, self.window_width, title=self.plot_title)

        peaks, p1, p2, vec = self._determine_curvature_all_lines(original)

        coef_p1, coef_p2 = self.fit(peaks, p1, p2)

        if self.plot >= 2:  # pragma: no cover
            self.progress.close()

        if self.plot:  # pragma: no cover
            self.plot_results(ncol, peaks, vec, p1, p2, coef_p1, coef_p2)

        # Create output arrays (nord, ncol)
        iorder, ipeaks = np.indices((self.n, ncol))
        p1, p2 = self.eval(ipeaks, iorder, coef_p1, coef_p2)

        if self.plot:  # pragma: no cover
            self.plot_comparison(original, p1, p2, peaks)

        return p1, p2


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
    return A * mu / ((x - x0) ** 2 + 0.25 * mu**2)
