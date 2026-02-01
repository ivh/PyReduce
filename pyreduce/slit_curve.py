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
from .curvature_model import SlitCurvature
from .extract import fix_parameters
from .util import make_index
from .util import polyfit2d_2 as polyfit2d

logger = logging.getLogger(__name__)


class Curvature:
    def __init__(
        self,
        traces,
        curve_height=0.5,
        extraction_height=0.2,
        column_range=None,
        trace_range=None,
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
        self.traces = traces
        self.curve_height = curve_height
        self.extraction_height = extraction_height
        self.column_range = column_range
        if trace_range is None:
            trace_range = (0, self.ntrace)
        self.trace_range = trace_range
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

        if self.curve_degree < 1 or self.curve_degree > 5:
            raise ValueError(f"Curvature degree must be 1-5, got {self.curve_degree}")

        if self.mode == "1D":
            # fit degree is an integer
            if not np.isscalar(self.fit_degree):
                self.fit_degree = self.fit_degree[0]
        elif self.mode == "2D":
            # fit degree is a 2 tuple
            if np.isscalar(self.fit_degree):
                self.fit_degree = (self.fit_degree, self.fit_degree)

    @property
    def ntrace(self):
        return self.traces.shape[0]

    @property
    def n(self):
        return self.trace_range[1] - self.trace_range[0]

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
        traces = self.traces
        curve_height = self.curve_height
        extraction_height = self.extraction_height
        column_range = self.column_range

        nrow, ncol = original.shape
        ntrace = len(traces)

        curve_height, column_range, traces = fix_parameters(
            curve_height, column_range, traces, nrow, ncol, ntrace
        )

        # For curvature, extraction_height is always literal pixels (no fractional conversion)
        if np.isscalar(extraction_height):
            extraction_height = np.full(ntrace, int(extraction_height))
        else:
            extraction_height = np.asarray(extraction_height, dtype=int)

        self.column_range = column_range[self.trace_range[0] : self.trace_range[1]]
        self.curve_height = curve_height[self.trace_range[0] : self.trace_range[1]]
        self.extraction_height = extraction_height[
            self.trace_range[0] : self.trace_range[1]
        ]
        self.traces = traces[self.trace_range[0] : self.trace_range[1]]
        self.trace_range = (0, self.ntrace)

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
            Index of the trace in self.traces

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

        # Get trace position
        x = np.arange(ncol)
        ycen = np.polyval(self.traces[order_idx], x)
        ycen_int = ycen.astype(int)

        # Special case: extraction_height=1 means row-by-row without extraction
        if xwd == 1:
            half = curve_xwd // 2
            offsets = np.arange(-half, curve_xwd - half)
            n_offsets = len(offsets)

            spectra = np.ma.zeros((n_offsets, ncol))
            spectra[:, :] = np.ma.masked

            for i, offset in enumerate(offsets):
                # Row position follows trace + offset
                row = ycen_int + offset
                # Check bounds
                if np.any(row < 0) or np.any(row >= nrow):
                    continue
                # Direct indexing along the curved trace
                spectra[i, cr[0] : cr[1]] = original[
                    row[cr[0] : cr[1]], x[cr[0] : cr[1]]
                ]

            return spectra, offsets.astype(float)

        # General case: extract and collapse multiple rows per offset
        n_offsets = max(1, int(curve_xwd // xwd))
        offsets = (np.arange(n_offsets) - (n_offsets - 1) / 2) * xwd

        spectra = np.ma.zeros((n_offsets, ncol))
        spectra[:, :] = np.ma.masked

        half = xwd // 2
        for i, offset in enumerate(offsets):
            # Y bounds for this offset spectrum
            yb = ycen_int + int(offset) - half
            yt = yb + xwd - 1

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
        """Fit polynomial curvature from peak position vs y-offset.

        For each peak: x(y) = x0 + c1*y + c2*y^2 + ... + cn*y^n

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
        coeffs : array of shape (n_peaks, curve_degree)
            Polynomial coefficients (excluding c0) for each peak.
            coeffs[i, 0] is c1 (linear), coeffs[i, 1] is c2 (quadratic), etc.
        residuals : array of shape (n_peaks, n_offsets)
            Residuals (measured - fitted) at each offset for each peak.
            NaN where no valid measurement.
        """
        n_peaks = len(peaks)
        n_offsets = len(offsets)
        coeffs = np.zeros((n_peaks, self.curve_degree))
        residuals = np.full((n_peaks, n_offsets), np.nan)
        min_points = self.curve_degree + 1

        for i in range(n_peaks):
            pos = positions[i]
            valid = ~np.isnan(pos)

            if np.sum(valid) < min_points:
                continue

            y = offsets[valid]
            x = pos[valid]

            # Subtract mean x to get relative shift
            x0 = np.mean(x)
            dx = x - x0

            try:
                # Fit polynomial of degree curve_degree
                # polyfit returns coefficients in descending order: [c_n, ..., c_1, c_0]
                poly_coef = np.polyfit(y, dx, self.curve_degree)
                # We want ascending order without c0: [c_1, c_2, ..., c_n]
                # poly_coef has length curve_degree+1
                # poly_coef[-2] is c_1, poly_coef[-3] is c_2, etc.
                for j in range(self.curve_degree):
                    coeffs[i, j] = poly_coef[-(j + 2)]

                # Compute residuals: measured - fitted
                dx_fitted = np.polyval(poly_coef, offsets)
                residuals[i, :] = positions[i] - x0 - dx_fitted
            except Exception:
                pass

        return coeffs, residuals

    def _fit_curvature_single_order(self, peaks, coeffs):
        """Fit smooth polynomial to curvature coefficients across an order.

        Parameters
        ----------
        peaks : array
            Peak columns
        coeffs : array of shape (n_peaks, curve_degree)
            Polynomial coefficients for each peak

        Returns
        -------
        fitted_coeffs : array of shape (curve_degree, fit_degree + 1)
            Fitted polynomial coefficients for each curvature term.
            fitted_coeffs[i] gives polyval coefficients for the i-th curvature term.
        peaks : array
            Filtered peak columns
        """
        try:
            # Use c1 (linear term) for outlier rejection
            c1 = coeffs[:, 0] if coeffs.shape[1] > 0 else np.zeros(len(peaks))
            middle = np.median(c1)
            sigma = np.percentile(c1, (32, 68))
            sigma = middle - sigma[0], sigma[1] - middle
            mask = (c1 >= middle - 5 * sigma[0]) & (c1 <= middle + 5 * sigma[1])
            peaks = peaks[mask]
            coeffs = coeffs[mask]

            fitted_coeffs = np.zeros((self.curve_degree, self.fit_degree + 1))
            for i in range(self.curve_degree):
                coef_init = np.zeros(self.fit_degree + 1)
                res = least_squares(
                    lambda coef, vals=coeffs[:, i]: np.polyval(coef, peaks) - vals,
                    x0=coef_init,
                    loss="arctan",
                )
                fitted_coeffs[i] = res.x

        except Exception:
            logger.error(
                "Could not fit the curvature of this order. Using no curvature instead"
            )
            fitted_coeffs = np.zeros((self.curve_degree, self.fit_degree + 1))

        return fitted_coeffs, peaks

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
        all_coeffs : list of arrays
            Curvature coefficients for each peak in each order.
            Each entry has shape (n_peaks, curve_degree).
        all_offsets : list of arrays
            Y-offsets used for each order
        all_residuals : list of arrays
            Fit residuals for each peak at each offset.
            Each entry has shape (n_peaks, n_offsets).
        plot_vec : list of arrays
            Middle spectrum for each order (for plotting)
        """
        all_peaks = []
        all_coeffs = []
        all_offsets = []
        all_residuals = []
        plot_vec = []

        for j in tqdm(range(self.n), desc="Trace"):
            logger.debug("Calculating curvature of trace %i out of %i", j + 1, self.n)

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
                all_coeffs.append(np.zeros((0, self.curve_degree)))
                all_offsets.append(offsets)
                all_residuals.append(np.zeros((0, len(offsets))))
                plot_vec.append(vec)
                continue

            # Fit curvature from peak positions
            coeffs, residuals = self._fit_curvature_from_positions(
                peaks, positions, offsets
            )

            all_peaks.append(peaks)
            all_coeffs.append(coeffs)
            all_offsets.append(offsets)
            all_residuals.append(residuals)
            plot_vec.append(vec)

        return all_peaks, all_coeffs, all_offsets, all_residuals, plot_vec

    def fit(self, peaks, all_coeffs):
        """Fit smooth polynomial to curvature coefficients.

        Parameters
        ----------
        peaks : list of arrays
            Peak columns for each order
        all_coeffs : list of arrays
            Curvature coefficients for each order.
            Each entry has shape (n_peaks, curve_degree).

        Returns
        -------
        fitted_coeffs : array
            For 1D mode: shape (n_orders, curve_degree, fit_degree + 1)
            For 2D mode: shape (curve_degree, ...) with polyfit2d coefficients
        """
        if self.mode == "1D":
            fitted_coeffs = np.zeros((self.n, self.curve_degree, self.fit_degree + 1))
            for j in range(self.n):
                fitted_coeffs[j], _ = self._fit_curvature_single_order(
                    peaks[j], all_coeffs[j]
                )
        elif self.mode == "2D":
            x = np.concatenate(peaks)
            y = [np.full(len(p), i) for i, p in enumerate(peaks)]
            y = np.concatenate(y)

            # Fit each curvature term separately
            fitted_coeffs = []
            for i in range(self.curve_degree):
                z = np.concatenate([c[:, i] for c in all_coeffs])
                coef = polyfit2d(x, y, z, degree=self.fit_degree, loss="arctan")
                fitted_coeffs.append(coef)
            fitted_coeffs = np.array(fitted_coeffs)

        return fitted_coeffs

    def eval(self, peaks, order, fitted_coeffs):
        """Evaluate fitted curvature coefficients at given positions.

        Parameters
        ----------
        peaks : array
            Column positions to evaluate at
        order : array
            Order indices (same shape as peaks)
        fitted_coeffs : array
            Fitted coefficients from fit() method

        Returns
        -------
        coeffs : array of shape (len(peaks), curve_degree)
            Evaluated curvature coefficients at each position
        """
        coeffs = np.zeros((len(peaks), self.curve_degree))

        if self.mode == "1D":
            # fitted_coeffs has shape (n_orders, curve_degree, fit_degree + 1)
            for i in np.unique(order):
                idx = order == i
                for j in range(self.curve_degree):
                    coeffs[idx, j] = np.polyval(fitted_coeffs[int(i), j], peaks[idx])
        elif self.mode == "2D":
            # fitted_coeffs has shape (curve_degree, ...)
            for j in range(self.curve_degree):
                coeffs[:, j] = polyval2d(peaks, order, fitted_coeffs[j])

        return coeffs

    def eval_legacy(self, peaks, order, fitted_coeffs):
        """Evaluate and return legacy p1, p2 format for backward compatibility."""
        coeffs = self.eval(peaks, order, fitted_coeffs)
        p1 = coeffs[:, 0] if self.curve_degree >= 1 else np.zeros(len(peaks))
        p2 = coeffs[:, 1] if self.curve_degree >= 2 else np.zeros(len(peaks))
        return p1, p2

    def plot_results(
        self, ncol, plot_peaks, plot_vec, plot_coeffs, fitted_coeffs
    ):  # pragma: no cover
        """Plot curvature fitting results.

        Parameters
        ----------
        ncol : int
            Number of columns in image
        plot_peaks : list of arrays
            Peak columns for each order
        plot_vec : list of arrays
            Middle spectrum for each order
        plot_coeffs : list of arrays
            Raw curvature coefficients for each order
        fitted_coeffs : array
            Fitted coefficients from fit() method
        """
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

        # Evaluate fitted coefficients for plotting
        t = [None for _ in range(self.n)]  # c1 (linear term)
        s = [None for _ in range(self.n)]  # c2 (quadratic term)
        for j in range(self.n):
            cr = self.column_range[j]
            x = np.arange(cr[0], cr[1])
            order = np.full(len(x), j)
            coeffs_eval = self.eval(x, order, fitted_coeffs)
            t[j] = coeffs_eval[:, 0] if self.curve_degree >= 1 else np.zeros(len(x))
            s[j] = coeffs_eval[:, 1] if self.curve_degree >= 2 else np.zeros(len(x))

        t_lower = min(arr.min() * (0.5 if arr.min() > 0 else 1.5) for arr in t)
        t_upper = max(arr.max() * (1.5 if arr.max() > 0 else 0.5) for arr in t)

        s_lower = min(arr.min() * (0.5 if arr.min() > 0 else 1.5) for arr in s)
        s_upper = max(arr.max() * (1.5 if arr.max() > 0 else 0.5) for arr in s)

        for j in range(self.n):
            cr = self.column_range[j]
            peaks = (
                plot_peaks[j].astype(int) if len(plot_peaks[j]) > 0 else np.array([])
            )
            vec = np.clip(plot_vec[j], 0, None)
            raw_coeffs = plot_coeffs[j]
            p1 = (
                raw_coeffs[:, 0]
                if raw_coeffs.shape[0] > 0 and self.curve_degree >= 1
                else np.array([])
            )
            p2 = (
                raw_coeffs[:, 1]
                if raw_coeffs.shape[0] > 0 and self.curve_degree >= 2
                else np.array([])
            )
            x = np.arange(cr[0], cr[1])

            # Figure Peaks found (and used)
            axes[j // 2, j % 2].plot(np.arange(cr[0], cr[1]), vec)
            if len(peaks) > 0:
                axes[j // 2, j % 2].plot(peaks, vec[peaks - cr[0]], "X")
            axes[j // 2, j % 2].set_xlim([0, ncol])
            if j not in (self.n - 1, self.n - 2):
                axes[j // 2, j % 2].get_xaxis().set_ticks([])

            # Figure 1st order
            if len(peaks) > 0 and len(p1) > 0:
                axes1[j // 2, j % 2].plot(peaks, p1, "rX")
            axes1[j // 2, j % 2].plot(x, t[j])
            axes1[j // 2, j % 2].set_xlim(0, ncol)

            axes1[j // 2, j % 2].set_ylim(t_lower, t_upper)
            if j not in (self.n - 1, self.n - 2):
                axes1[j // 2, j % 2].get_xaxis().set_ticks([])
            else:
                axes1[j // 2, j % 2].set_xlabel("x [pixel]")
            if j == self.n // 2 + 1:
                axes1[j // 2, j % 2].set_ylabel("c1 [pixel/pixel]")

            # Figure 2nd order
            if len(peaks) > 0 and len(p2) > 0:
                axes2[j // 2, j % 2].plot(peaks, p2, "rX")
            axes2[j // 2, j % 2].plot(x, s[j])
            axes2[j // 2, j % 2].set_xlim(0, ncol)

            axes2[j // 2, j % 2].set_ylim(s_lower, s_upper)
            if j not in (self.n - 1, self.n - 2):
                axes2[j // 2, j % 2].get_xaxis().set_ticks([])
            else:
                axes2[j // 2, j % 2].set_xlabel("x [pixel]")
            if j == self.n // 2 + 1:
                axes2[j // 2, j % 2].set_ylabel("c2 [pixel/pixel**2]")

        axes1 = trim_axs(axes1, self.n)
        axes2 = trim_axs(axes2, self.n)

        util.show_or_save("curvature_fit")

    def plot_comparison(
        self, original, coeffs_array, peaks, slitdeltas=None
    ):  # pragma: no cover
        """Plot comparison of curvature model vs data.

        Parameters
        ----------
        original : array
            Original image
        coeffs_array : array of shape (ntrace, ncol, curve_degree + 1)
            Curvature coefficients at each point
        peaks : list of arrays
            Peak columns for each order
        slitdeltas : array of shape (ntrace, nrows), optional
            Per-row residual offsets. If provided, plotted as white lines
            offset from the polynomial (red lines).
        """
        plt.figure()
        _, ncol = original.shape
        output = np.zeros((np.sum(self.curve_height) + self.ntrace, ncol))
        pos = [0]
        x = np.arange(ncol)
        for i in range(self.ntrace):
            ycen = np.polyval(self.traces[i], x)
            half = self.curve_height[i] // 2
            yb = ycen - half
            yt = yb + self.curve_height[i] - 1
            xl, xr = self.column_range[i]
            index = make_index(yb, yt, xl, xr)
            yl = pos[i]
            yr = pos[i] + index[0].shape[0]
            output[yl:yr, xl:xr] = original[index]
            pos += [yr]

        vmin, vmax = np.percentile(output[output != 0], (5, 95))
        plt.imshow(output, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")

        for i in range(self.ntrace):
            for p in peaks[i]:
                p = int(p)
                if p >= coeffs_array.shape[1]:
                    continue
                ew = self.curve_height[i]
                half = ew // 2
                curve_x = np.zeros(ew)
                y_offsets = np.arange(-half, ew - half)
                for j, yt in enumerate(y_offsets):
                    # Evaluate polynomial: dx = c1*y + c2*y^2 + ...
                    dx = 0
                    for k in range(self.curve_degree):
                        dx += coeffs_array[i, p, k + 1] * (yt ** (k + 1))
                    curve_x[j] = p + dx
                y_plot = y_offsets + pos[i] + half
                # Red line: polynomial curvature only
                plt.plot(curve_x, y_plot, "r", linewidth=1)

                # White line: polynomial + slitdeltas (if available)
                if slitdeltas is not None and i < slitdeltas.shape[0]:
                    # Interpolate slitdeltas to curve_height resolution
                    sd = slitdeltas[i]
                    if len(sd) != ew:
                        sd_x = np.linspace(0, 1, len(sd))
                        curve_x_interp = np.linspace(0, 1, ew)
                        sd = np.interp(curve_x_interp, sd_x, sd)
                    curve_x_with_delta = curve_x + sd
                    plt.plot(curve_x_with_delta, y_plot, "w", linewidth=0.5, alpha=0.8)

        locs = self.curve_height + 1
        locs = np.array([0, *np.cumsum(locs)[:-1]])
        locs[:-1] += (np.diff(locs) * 0.5).astype(int)
        locs[-1] += ((output.shape[0] - locs[-1]) * 0.5).astype(int)

        plt.yticks(locs, range(len(locs)))
        if self.plot_title is not None:
            plt.title(self.plot_title)
        plt.xlabel("x [pixel]")
        plt.ylabel("order")
        util.show_or_save("curvature_comparison")

    def _compute_slitdeltas(self, all_offsets, all_residuals, nrows):
        """Compute per-row slitdeltas from fit residuals.

        For each trace, average residuals across peaks at each offset,
        then interpolate to per-row values.

        Parameters
        ----------
        all_offsets : list of arrays
            Y-offsets used for each order
        all_residuals : list of arrays
            Fit residuals, shape (n_peaks, n_offsets) per order
        nrows : int
            Number of rows covering curve_height range

        Returns
        -------
        slitdeltas : array of shape (ntrace, nrows)
            Per-row residual offsets for each trace, covering curve_height range.
            During extraction, interpolated to match swath size.
        """
        slitdeltas = np.zeros((self.n, nrows))

        for j in range(self.n):
            offsets = all_offsets[j]
            residuals = all_residuals[j]

            if len(residuals) == 0:
                continue

            # Average residuals across peaks at each offset (ignoring NaN)
            mean_resid = np.nanmedian(residuals, axis=0)

            # Map offsets to row indices
            # offsets are relative to trace center
            xwd = self.curve_height[j]
            half = xwd // 2
            row_offsets = np.linspace(-half, xwd - half, nrows)

            # Interpolate from measured offsets to per-row
            valid = ~np.isnan(mean_resid)
            if np.sum(valid) >= 2:
                slitdeltas[j] = np.interp(
                    row_offsets, offsets[valid], mean_resid[valid], left=0, right=0
                )

        return slitdeltas

    def execute(self, original, compute_slitdeltas=True):
        """Execute curvature determination using row-tracking method.

        Parameters
        ----------
        original : array of shape (nrow, ncol)
            Original image
        compute_slitdeltas : bool
            Whether to compute slitdeltas from fit residuals (default: True)

        Returns
        -------
        curvature : SlitCurvature
            Curvature data including polynomial coefficients and optionally slitdeltas.
        """
        logger.info("Determining the Slit Curvature")

        _, ncol = original.shape

        self._fix_inputs(original)

        peaks, all_coeffs, all_offsets, all_residuals, vec = (
            self._determine_curvature_all_lines(original)
        )

        fitted_coeffs = self.fit(peaks, all_coeffs)

        if self.plot:  # pragma: no cover
            self.plot_results(ncol, peaks, vec, all_coeffs, fitted_coeffs)

        # Create output array (ntrace, ncol, curve_degree + 1)
        iorder, ipeaks = np.indices((self.n, ncol))
        coeffs_flat = self.eval(ipeaks.ravel(), iorder.ravel(), fitted_coeffs)
        # coeffs_flat has shape (n * ncol, curve_degree)

        # Build full coefficient array with c0 = 0
        coeffs = np.zeros((self.n, ncol, self.curve_degree + 1))
        coeffs[:, :, 1:] = coeffs_flat.reshape(self.n, ncol, self.curve_degree)

        # Compute slitdeltas from residuals
        slitdeltas = None
        if compute_slitdeltas:
            # Use max curve height to determine nrows (full measured range)
            max_curve = int(np.max(self.curve_height)) + 1
            slitdeltas = self._compute_slitdeltas(all_offsets, all_residuals, max_curve)
            logger.info(
                "Computed slitdeltas with shape %s, range [%.4f, %.4f]",
                slitdeltas.shape,
                np.nanmin(slitdeltas),
                np.nanmax(slitdeltas),
            )

        if self.plot:  # pragma: no cover
            self.plot_comparison(original, coeffs, peaks, slitdeltas=slitdeltas)

        # Build compact fitted_coeffs with c0=0 row prepended
        # fitted_coeffs from fit() has shape (ntrace, curve_degree, fit_degree+1) for 1D mode
        # We want (ntrace, curve_degree+1, fit_degree+1) with c0 row = 0
        if self.mode == "1D":
            compact = np.zeros(
                (self.n, self.curve_degree + 1, self.fit_degree + 1),
                dtype=np.float64,
            )
            compact[:, 1:, :] = fitted_coeffs
            fit_deg = self.fit_degree
        else:
            # 2D mode: fitted_coeffs has shape (curve_degree, ...) - more complex
            # For now, only store compact form for 1D mode
            compact = None
            fit_deg = None

        curvature = SlitCurvature(
            coeffs=coeffs,
            slitdeltas=slitdeltas,
            degree=self.curve_degree,
            fitted_coeffs=compact,
            fit_degree=fit_deg,
        )
        return curvature


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
