# -*- coding: utf-8 -*-
"""
Wavelength Calibration
by comparison to a reference spectrum
Loosely bases on the IDL wavecal function
"""

import logging
from os.path import dirname, join

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from numpy.polynomial.polynomial import Polynomial, polyval2d
from scipy import signal
from scipy.constants import speed_of_light
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.morphology import grey_closing
from scipy.optimize import curve_fit
from tqdm import tqdm

from . import util

logger = logging.getLogger(__name__)


def polyfit(x, y, deg):
    res = Polynomial.fit(x, y, deg, domain=[])
    coef = res.coef[::-1]
    return coef


class AlignmentPlot:
    """
    Makes a plot which can be clicked to align the two spectra, reference and observed
    """

    def __init__(self, ax, obs, lines, offset=(0, 0), plot_title=None):
        self.im = ax
        self.first = True
        self.nord, self.ncol = obs.shape
        self.RED, self.GREEN, self.BLUE = 0, 1, 2

        self.obs = obs
        self.lines = lines
        self.plot_title = plot_title

        self.order_first = 0
        self.spec_first = ""
        self.x_first = 0
        self.offset = list(offset)

        self.make_ref_image()

    def make_ref_image(self):
        """create and show the reference plot, with the two spectra"""
        ref_image = np.zeros((self.nord * 2, self.ncol, 3))
        for iord in range(self.nord):
            ref_image[iord * 2, :, self.RED] = 10 * np.ma.filled(self.obs[iord], 0)
            if 0 <= iord + self.offset[0] < self.nord:
                for line in self.lines[self.lines["order"] == iord]:
                    first = int(np.clip(line["xfirst"] + self.offset[1], 0, self.ncol))
                    last = int(np.clip(line["xlast"] + self.offset[1], 0, self.ncol))
                    order = (iord + self.offset[0]) * 2 + 1
                    ref_image[order, first:last, self.GREEN] = (
                        10
                        * line["height"]
                        * signal.gaussian(last - first, line["width"])
                    )
        ref_image = np.clip(ref_image, 0, 1)
        ref_image[ref_image < 0.1] = 0

        self.im.imshow(
            ref_image,
            aspect="auto",
            origin="lower",
            extent=(-0.5, self.ncol - 0.5, -0.5, self.nord - 0.5),
        )
        title = "Alignment, Observed: RED, Reference: GREEN\nGreen should be above red!"
        if self.plot_title is not None:
            title = f"{self.plot_title}\n{title}"
        self.im.figure.suptitle(title)
        self.im.axes.set_xlabel("x [pixel]")
        self.im.axes.set_ylabel("Order")

        self.im.figure.canvas.draw()

    def connect(self):
        """connect the click event with the appropiate function"""
        self.cidclick = self.im.figure.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

    def on_click(self, event):
        """On click offset the reference by the distance between click positions"""
        if event.ydata is None:
            return
        order = int(np.floor(event.ydata))
        spec = "ref" if (event.ydata - order) > 0.5 else "obs"  # if True then reference
        x = event.xdata
        print("Order: %i, Spectrum: %s, x: %g" % (order, "ref" if spec else "obs", x))

        # on every second click
        if self.first:
            self.first = False
            self.order_first = order
            self.spec_first = spec
            self.x_first = x
        else:
            # Clicked different spectra
            if spec != self.spec_first:
                self.first = True
                direction = -1 if spec == "ref" else 1
                offset_orders = int(order - self.order_first) * direction
                offset_x = int(x - self.x_first) * direction
                self.offset[0] -= offset_orders - 1
                self.offset[1] -= offset_x
                self.make_ref_image()


class LineAtlas:
    def __init__(self, element, medium="vac"):
        self.element = element
        self.medium = medium

        fname = element.lower() + ".fits"
        folder = dirname(__file__)
        self.fname = join(folder, "wavecal/atlas", fname)
        self.wave, self.flux = self.load_fits(self.fname)

        try:
            # If a specific linelist file is provided
            fname_list = element.lower() + "_list.txt"
            self.fname_list = join(folder, "wavecal/atlas", fname_list)
            linelist = np.genfromtxt(self.fname_list, dtype="f8,U8")
            wpos, element = linelist["f0"], linelist["f1"]
            indices = self.wave.searchsorted(wpos)
            heights = self.flux[indices]
            self.linelist = np.rec.fromarrays(
                [wpos, heights, element], names=["wave", "heights", "element"]
            )
        except (FileNotFoundError, IOError):
            # Otherwise fit the line positions from the spectrum
            logger.warning(
                "No dedicated linelist found for %s, determining peaks based on the reference spectrum instead.",
                element,
            )
            module = WavelengthCalibration(plot=False)
            n, peaks = module._find_peaks(self.flux)
            wpos = np.interp(peaks, np.arange(len(self.wave)), self.wave)
            element = np.full(len(wpos), element)
            indices = self.wave.searchsorted(wpos)
            heights = self.flux[indices]
            self.linelist = np.rec.fromarrays(
                [wpos, heights, element], names=["wave", "heights", "element"]
            )

        # The data files are in vaccuum, if the instrument is in air, we need to convert
        if medium == "air":
            self.wave = util.vac2air(self.wave)
            self.linelist["wave"] = util.vac2air(self.linelist["wave"])

    def load_fits(self, fname):
        hdu = fits.open(fname)
        if len(hdu) == 1:
            # Its just the spectrum
            # with the wavelength defined via the header keywords
            header = hdu[0].header
            spec = hdu[0].data.ravel()
            wmin = header["CRVAL1"]
            wdel = header["CDELT1"]
            wave = np.arange(spec.size) * wdel + wmin
        else:
            # Its a binary Table, with two columns for the wavelength and the
            # spectrum
            data = hdu[1].data
            wave = data["wave"]
            spec = data["spec"]

        spec /= np.nanmax(spec)
        spec = np.clip(spec, 0, None)
        return wave, spec


class LineList:
    dtype = np.dtype(
        (
            np.record,
            [
                (("wlc", "WLC"), ">f8"),  # Wavelength (before fit)
                (("wll", "WLL"), ">f8"),  # Wavelength (after fit)
                (("posc", "POSC"), ">f8"),  # Pixel Position (before fit)
                (("posm", "POSM"), ">f8"),  # Pixel Position (after fit)
                (("xfirst", "XFIRST"), ">i2"),  # first pixel of the line
                (("xlast", "XLAST"), ">i2"),  # last pixel of the line
                (
                    ("approx", "APPROX"),
                    "O",
                ),  # Not used. Describes the shape used to approximate the line. "G" for Gaussian
                (("width", "WIDTH"), ">f8"),  # width of the line in pixels
                (("height", "HEIGHT"), ">f8"),  # relative strength of the line
                (("order", "ORDER"), ">i2"),  # echelle order the line is found in
                ("flag", "?"),  # flag that tells us if we should use the line or not
            ],
        )
    )

    def __init__(self, lines=None):
        if lines is None:
            lines = np.array([], dtype=self.dtype)
        self.data = lines
        self.dtype = self.data.dtype

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    @classmethod
    def load(cls, filename):
        data = np.load(filename, allow_pickle=True)
        linelist = cls(data["cs_lines"])
        return linelist

    def save(self, filename):
        np.savez(filename, cs_lines=self.data)

    def append(self, linelist):
        if isinstance(linelist, LineList):
            linelist = linelist.data
        self.data = np.append(self.data, linelist)

    def add_line(self, wave, order, pos, width, height, flag):
        lines = self.from_list([wave], [order], [pos], [width], [height], [flag])
        self.data = np.append(self.data, lines)

    @classmethod
    def from_list(cls, wave, order, pos, width, height, flag):
        lines = [
            (w, w, p, p, p - wi / 2, p + wi / 2, b"G", wi, h, o, f)
            for w, o, p, wi, h, f in zip(wave, order, pos, width, height, flag)
        ]
        lines = np.array(lines, dtype=cls.dtype)
        return cls(lines)


class WavelengthCalibration:
    """
    Wavelength Calibration Module

    Takes an observed wavelength image and the reference linelist
    and returns the wavelength at each pixel
    """

    def __init__(
        self,
        threshold=100,
        degree=(6, 6),
        iterations=3,
        dimensionality="2D",
        nstep=0,
        shift_window=0.01,
        manual=False,
        polarim=False,
        lfc_peak_width=3,
        closing=5,
        element=None,
        medium="vac",
        plot=True,
        plot_title=None,
    ):
        #:float: Residual threshold in m/s above which to remove lines
        self.threshold = threshold
        #:tuple(int, int): polynomial degree of the wavelength fit in (pixel, order) direction
        self.degree = degree
        if dimensionality == "1D":
            self.degree = int(degree)
        elif dimensionality == "2D":
            self.degree = (int(degree[0]), int(degree[1]))
        #:int: Number of iterations in the remove residuals, auto id, loop
        self.iterations = iterations
        #:{"1D", "2D"}: Whether to use 1d or 2d fit
        self.dimensionality = dimensionality
        #:bool: Whether to fit for pixel steps (offsets) in the detector
        self.nstep = nstep
        #:float: Fraction if the number of columns to use in the alignment of individual orders. Set to 0 to disable
        self.shift_window = shift_window
        #:bool: Whether to manually align the reference instead of using cross correlation
        self.manual = manual
        #:bool: Whether to use polarimetric orders instead of the usual ones. I.e. Each pair of two orders represents the same data. Not Supported yet
        self.polarim = polarim
        #:int: Whether to plot the results. Set to 2 to plot during all steps.
        self.plot = plot
        self.plot_title = plot_title
        #:str: Elements used in the wavelength calibration. Used in AutoId to find more lines from the Atlas
        self.element = element
        #:str: Medium of the detector, vac or air
        self.medium = medium
        #:int: Laser Frequency Peak width (for scipy.signal.find_peaks)
        self.lfc_peak_width = lfc_peak_width
        #:int: grey closing range for the input image
        self.closing = 5
        #:int: Number of orders in the observation
        self.nord = None
        #:int: Number of columns in the observation
        self.ncol = None

    @property
    def step_mode(self):
        return self.nstep > 0

    @property
    def dimensionality(self):
        """{"1D", "2D"}: Whether to use 1D or 2D polynomials for the wavelength solution"""
        return self._dimensionality

    @dimensionality.setter
    def dimensionality(self, value):
        accepted_values = ["1D", "2D"]
        if value in accepted_values:
            self._dimensionality = value
        else:
            raise ValueError(
                f"Value for 'dimensionality' not understood. Expected one of {accepted_values} but got {value} instead"
            )

    def normalize(self, obs, lines):
        """
        Normalize the observation and reference list in each order individually
        Copies the data if the image, but not of the linelist

        Parameters
        ----------
        obs : array of shape (nord, ncol)
            observed image
        lines : recarray of shape (nlines,)
            reference linelist

        Returns
        -------
        obs : array of shape (nord, ncol)
            normalized image
        lines : recarray of shape (nlines,)
            normalized reference linelist
        """
        # normalize order by order
        obs = np.ma.copy(obs)
        for i in range(len(obs)):
            if self.closing > 0:
                obs[i] = grey_closing(obs[i], self.closing)
            try:
                obs[i] -= np.ma.median(obs[i][obs[i] > 0])
            except ValueError:
                logger.warning(
                    f"Could not determine the minimum value in order %i. No positive values found",
                    i,
                )
            obs[i] /= np.ma.max(obs[i])

        # Remove negative outliers
        std = np.std(obs, axis=1)[:, None]
        obs[obs <= -2 * std] = np.ma.masked
        # obs[obs <= 0] = np.ma.masked

        # Normalize lines in each order
        for order in np.unique(lines["order"]):
            select = lines["order"] == order
            topheight = np.max(lines[select]["height"])
            lines["height"][select] /= topheight

        return obs, lines

    def create_image_from_lines(self, lines):
        """
        Create a reference image based on a line list
        Each line will be approximated by a Gaussian
        Space inbetween lines is 0
        The number of orders is from 0 to the maximum order

        Parameters
        ----------
        lines : recarray of shape (nlines,)
            line data

        Returns
        -------
        img : array of shape (nord, ncol)
            New reference image
        """
        min_order = int(np.min(lines["order"]))
        max_order = int(np.max(lines["order"]))
        img = np.zeros((max_order - min_order + 1, self.ncol))
        for line in lines:
            if line["order"] < 0:
                continue
            if line["xlast"] < 0 or line["xfirst"] > self.ncol:
                continue
            first = int(max(line["xfirst"], 0))
            last = int(min(line["xlast"], self.ncol))
            img[int(line["order"]) - min_order, first:last] = line[
                "height"
            ] * signal.gaussian(last - first, line["width"])
        return img

    def align_manual(self, obs, lines):
        """
        Open an AlignmentPlot window for manual selection of the alignment

        Parameters
        ----------
        obs : array of shape (nord, ncol)
            observed image
        lines : recarray of shape (nlines,)
            reference linelist

        Returns
        -------
        offset : tuple(int, int)
            offset in order and column to be applied to each line in the linelist
        """
        _, ax = plt.subplots()
        ap = AlignmentPlot(ax, obs, lines, plot_title=self.plot_title)
        ap.connect()
        plt.show()
        offset = ap.offset
        return offset

    def apply_alignment_offset(self, lines, offset, select=None):
        """
        Apply an offset to the linelist

        Parameters
        ----------
        lines : recarray of shape (nlines,)
            reference linelist
        offset : tuple(int, int)
            offset in (order, column)
        select : array of shape(nlines,), optional
            Mask that defines which lines the offset applies to

        Returns
        -------
        lines : recarray of shape (nlines,)
            linelist with offset applied
        """
        if select is None:
            select = slice(None)
        lines["xfirst"][select] += offset[1]
        lines["xlast"][select] += offset[1]
        lines["posm"][select] += offset[1]
        lines["order"][select] += offset[0]
        return lines

    def align(self, obs, lines):
        """
        Align the observation with the reference spectrum
        Either automatically using cross correlation or manually (visually)

        Parameters
        ----------
        obs : array[nrow, ncol]
            observed wavelength calibration spectrum (e.g. obs=ThoriumArgon)
        lines : struct_array
            reference line data
        manual : bool, optional
            wether to manually align the spectra (default: False)
        plot : bool, optional
            wether to plot the alignment (default: False)

        Returns
        -------
        offset: tuple(int, int)
            offset in order and column
        """
        obs = np.ma.filled(obs, 0)

        if not self.manual:
            # make image from lines
            img = self.create_image_from_lines(lines)

            # Cross correlate with obs image
            # And determine overall offset
            correlation = signal.correlate2d(obs, img, mode="same")
            offset_order, offset_x = np.unravel_index(
                np.argmax(correlation), correlation.shape
            )

            if self.plot >= 2:
                plt.imshow(correlation, aspect="auto")
                plt.vlines(offset_x, -0.5, self.nord - 0.5, color="red")
                plt.hlines(offset_order, -0.5, self.ncol - 0.5, color="red")
                if self.plot_title is not None:
                    plt.title(self.plot_title)
                plt.show()

            offset_order = offset_order - img.shape[0] / 2 + 1
            offset_x = offset_x - img.shape[1] / 2 + 1
            offset = [int(offset_order), int(offset_x)]

            # apply offset
            lines = self.apply_alignment_offset(lines, offset)

            if self.shift_window != 0:
                # Shift individual orders to fit reference
                # Only allow a small shift here (1%) ?
                img = self.create_image_from_lines(lines)
                for i in range(max(offset[0], 0), min(len(obs), len(img))):
                    correlation = signal.correlate(obs[i], img[i], mode="same")
                    width = int(self.ncol * self.shift_window) // 2
                    low, high = self.ncol // 2 - width, self.ncol // 2 + width
                    offset_x = np.argmax(correlation[low:high]) + low
                    offset_x = int(offset_x - self.ncol / 2 + 1)

                    select = lines["order"] == i
                    lines = self.apply_alignment_offset(lines, (0, offset_x), select)

        if self.plot or self.manual:
            offset = self.align_manual(obs, lines)
            lines = self.apply_alignment_offset(lines, offset)

        logger.debug(f"Offset order: {offset[0]}, Offset pixel: {offset[1]}")

        return lines

    def _fit_single_line(self, obs, center, width, plot=False):
        low = int(center - width * 5)
        low = max(low, 0)
        high = int(center + width * 5)
        high = min(high, len(obs))

        section = obs[low:high]
        x = np.arange(low, high, 1)
        x = np.ma.masked_array(x, mask=np.ma.getmaskarray(section))
        coef = util.gaussfit2(x, section)

        if self.plot >= 2 and plot:
            x2 = np.linspace(x.min(), x.max(), len(x) * 100)
            plt.plot(x, section, label="Observation")
            plt.plot(x2, util.gaussval2(x2, *coef), label="Fit")
            title = "Gaussian Fit to spectral line"
            if self.plot_title is not None:
                title = f"{self.plot_title}\n{title}"
            plt.title(title)
            plt.xlabel("x [pixel]")
            plt.ylabel("Intensity [a.u.]")
            plt.legend()
            plt.show()
        return coef

    def fit_lines(self, obs, lines):
        """
        Determine exact position of each line on the detector based on initial guess

        This fits a Gaussian to each line, and uses the peak position as a new solution

        Parameters
        ----------
        obs : array of shape (nord, ncol)
            observed wavelength calibration image
        lines : recarray of shape (nlines,)
            reference line data

        Returns
        -------
        lines : recarray of shape (nlines,)
            Updated line information (posm is changed)
        """
        # For each line fit a gaussian to the observation
        for i, line in tqdm(
            enumerate(lines), total=len(lines), leave=False, desc="Lines"
        ):
            if line["posm"] < 0 or line["posm"] >= obs.shape[1]:
                # Line outside pixel range
                continue
            if line["order"] < 0 or line["order"] >= len(obs):
                # Line outside order range
                continue

            try:
                coef = self._fit_single_line(
                    obs[int(line["order"])],
                    line["posm"],
                    line["width"],
                    plot=line["flag"],
                )
                lines[i]["posm"] = coef[1]
            except:
                # Gaussian fit failed, dont use line
                lines[i]["flag"] = False

        return lines

    def build_2d_solution(self, lines, plot=False):
        """
        Create a 2D polynomial fit to flagged lines
        degree : tuple(int, int), optional
            polynomial degree of the fit in (column, order) dimension (default: (6, 6))

        Parameters
        ----------
        lines : struc_array
            line data
        plot : bool, optional
            wether to plot the solution (default: False)

        Returns
        -------
        coef : array[degree_x, degree_y]
            2d polynomial coefficients
        """

        if self.step_mode:
            return self.build_step_solution(lines, plot=plot)

        # Only use flagged data
        mask = lines["flag"]  # True: use line, False: dont use line
        m_wave = lines["wll"][mask]
        m_pix = lines["posm"][mask]
        m_ord = lines["order"][mask]

        if self.dimensionality == "1D":
            nord = self.nord
            coef = np.zeros((nord, self.degree + 1))
            for i in range(nord):
                select = m_ord == i
                if np.count_nonzero(select) < 2:
                    # Not enough lines for wavelength solution
                    logger.warning(
                        "Not enough valid lines found wavelength calibration in order % i",
                        i,
                    )
                    coef[i] = np.nan
                    continue

                deg = max(min(self.degree, np.count_nonzero(select) - 2), 0)
                coef[i, -(deg + 1) :] = np.polyfit(
                    m_pix[select], m_wave[select], deg=deg
                )
        elif self.dimensionality == "2D":
            # 2d polynomial fit with: x = column, y = order, z = wavelength
            coef = util.polyfit2d(m_pix, m_ord, m_wave, degree=self.degree, plot=False)
        else:
            raise ValueError(
                f"Parameter 'mode' not understood. Expected '1D' or '2D' but got {self.dimensionality}"
            )

        if plot or self.plot >= 2:  # pragma: no cover
            self.plot_residuals(lines, coef, title="Residuals")

        return coef

    def g(self, x, step_coef_pos, step_coef_diff):
        try:
            bins = step_coef_pos
            digits = np.digitize(x, bins) - 1
        except ValueError as e:
            return np.inf

        cumsum = np.cumsum(step_coef_diff)
        x = x + cumsum[digits]
        return x

    def f(self, x, poly_coef, step_coef_pos, step_coef_diff):
        xdash = self.g(x, step_coef_pos, step_coef_diff)
        if np.all(np.isinf(xdash)):
            return np.inf
        y = np.polyval(poly_coef, xdash)
        return y

    def build_step_solution(self, lines, plot=False):
        """
        Fit the least squares fit to the wavelength points,
        with additional free parameters for detector gaps, e.g. due to stitching.

        The exact method of the fit depends on the dimensionality.
        Either way we are using the usual polynomial fit for the wavelength, but
        the x points are modified beforehand by shifting them some amount, at specific
        indices. We assume that the stitching effects are distributed evenly and we know how
        many steps we expect (this is set as "nstep").

        Parameters
        ----------
        lines : np.recarray
            linedata
        plot : bool, optional
            whether to plot results or not, by default False

        Returns
        -------
        coef
            coefficients of the best fit
        """
        mask = lines["flag"]  # True: use line, False: dont use line
        m_wave = lines["wll"][mask]
        m_pix = lines["posm"][mask]
        m_ord = lines["order"][mask]

        nstep = self.nstep
        ncol = self.ncol

        if self.dimensionality == "1D":
            coef = {}
            for order in np.unique(m_ord):
                select = m_ord == order
                x = xl = m_pix[select]
                y = m_wave[select]
                step_coef = np.zeros((nstep, 2))
                step_coef[:, 0] = np.linspace(ncol / (nstep + 1), ncol, nstep + 1)[:-1]

                def func(x, *param):
                    return self.f(x, poly_coef, step_coef[:, 0], param)

                for i in range(5):
                    poly_coef = np.polyfit(xl, y, self.degree)
                    res, _ = curve_fit(func, x, y, p0=step_coef[:, 1], bounds=[-1, 1])
                    step_coef[:, 1] = res
                    xl = self.g(x, step_coef[:, 0], step_coef[:, 1])

                coef[order] = [poly_coef, step_coef]
        elif self.dimensionality == "2D":
            unique = np.unique(m_ord)
            nord = len(unique)
            shape = (self.degree[0] + 1, self.degree[1] + 1)
            n = np.prod(shape)

            step_coef = np.zeros((nord, nstep, 2))
            step_coef[:, :, 0] = np.linspace(ncol / (nstep + 1), ncol, nstep + 1)[:-1]

            def func(x, *param):
                x, y = x[: len(x) // 2], x[len(x) // 2 :]
                theta = np.asarray(param).reshape((nord, nstep))
                xl = np.copy(x)
                for j, i in enumerate(unique):
                    xl[y == i] = self.g(x[y == i], step_coef[j, :, 0], theta[j])
                z = polyval2d(xl, y, poly_coef)
                return z

            # TODO: this could use some optimization
            x = np.copy(m_pix)
            x0 = np.concatenate((m_pix, m_ord))
            resid_old = np.inf
            for k in tqdm(range(5)):
                poly_coef = util.polyfit2d(
                    x, m_ord, m_wave, degree=self.degree, plot=False
                )

                res, _ = curve_fit(func, x0, m_wave, p0=step_coef[:, :, 1])
                step_coef[:, :, 1] = res.reshape((nord, nstep))
                for j, i in enumerate(unique):
                    x[m_ord == i] = self.g(
                        m_pix[m_ord == i], step_coef[j][:, 0], step_coef[j][:, 1]
                    )

                resid = polyval2d(x, m_ord, poly_coef) - m_wave
                resid = np.sum(resid ** 2)
                improvement = resid_old - resid
                resid_old = resid
                logger.info(
                    "Iteration: %i, Residuals: %.5g, Improvement: %.5g",
                    k,
                    resid,
                    improvement,
                )

            poly_coef = util.polyfit2d(x, m_ord, m_wave, degree=self.degree, plot=False)
            step_coef = {i: step_coef[j] for j, i in enumerate(unique)}
            coef = (poly_coef, step_coef)
        else:
            raise ValueError(
                f"Parameter 'dimensionality' not understood. Expected '1D' or '2D' but got {self.dimensionality}"
            )

        return coef

    def evaluate_step_solution(self, pos, order, solution):
        if not np.array_equal(np.shape(pos), np.shape(order)):
            raise ValueError("pos and order must have the same shape")
        if self.dimensionality == "1D":
            result = np.zeros(pos.shape)
            for i in np.unique(order):
                select = order == i
                result[select] = self.f(
                    pos[select],
                    solution[i][0],
                    solution[i][1][:, 0],
                    solution[i][1][:, 1],
                )
        elif self.dimensionality == "2D":
            poly_coef, step_coef = solution
            pos = np.copy(pos)
            for i in np.unique(order):
                pos[order == i] = self.g(
                    pos[order == i], step_coef[i][:, 0], step_coef[i][:, 1]
                )
            result = polyval2d(pos, order, poly_coef)
        else:
            raise ValueError(
                f"Parameter 'mode' not understood, expected '1D' or '2D' but got {self.dimensionality}"
            )
        return result

    def evaluate_solution(self, pos, order, solution):
        """
        Evaluate the 1d or 2d wavelength solution at the given pixel positions and orders

        Parameters
        ----------
        pos : array
            pixel position on the detector (i.e. x axis)
        order : array
            order of each point
        solution : array of shape (nord, ndegree) or (degree_x, degree_y)
            polynomial coefficients. For mode=1D, one set of coefficients per order.
            For mode=2D, the first dimension is for the positions and the second for the orders
        mode : str, optional
            Wether to interpret the solution as 1D or 2D polynomials, by default "1D"

        Returns
        -------
        result: array
            Evaluated polynomial

        Raises
        ------
        ValueError
            If pos and order have different shapes, or mode is of the wrong value
        """
        if not np.array_equal(np.shape(pos), np.shape(order)):
            raise ValueError("pos and order must have the same shape")

        if self.step_mode:
            return self.evaluate_step_solution(pos, order, solution)

        if self.dimensionality == "1D":
            result = np.zeros(pos.shape)
            for i in np.unique(order):
                select = order == i
                result[select] = np.polyval(solution[int(i)], pos[select])
        elif self.dimensionality == "2D":
            result = np.polynomial.polynomial.polyval2d(pos, order, solution)
        else:
            raise ValueError(
                f"Parameter 'mode' not understood, expected '1D' or '2D' but got {self.dimensionality}"
            )
        return result

    def make_wave(self, wave_solution, plot=False):
        """Expand polynomial wavelength solution into full image

        Parameters
        ----------
        wave_solution : array of shape(degree,)
            polynomial coefficients of wavelength solution
        plot : bool, optional
            wether to plot the solution (default: False)

        Returns
        -------
        wave_img : array of shape (nord, ncol)
            wavelength solution for each point in the spectrum
        """

        y, x = np.indices((self.nord, self.ncol))
        wave_img = self.evaluate_solution(x, y, wave_solution)

        return wave_img

    def auto_id(self, obs, wave_img, lines):
        """Automatically identify peaks that are close to known lines

        Parameters
        ----------
        obs : array of shape (nord, ncol)
            observed spectrum
        wave_img : array of shape (nord, ncol)
            wavelength solution image
        lines : struc_array
            line data
        threshold : int, optional
            difference threshold between line positions in m/s, until which a line is considered identified (default: 1)
        plot : bool, optional
            wether to plot the new lines

        Returns
        -------
        lines : struct_array
            line data with new flags
        """

        new_lines = []
        if self.atlas is not None:
            # For each order, find the corresponding section in the Atlas
            # Look for strong lines in the atlas and the spectrum that match in position
            # Add new lines to the linelist
            width_of_atlas_peaks = 3
            for order in range(obs.shape[0]):
                mask = ~np.ma.getmask(obs[order])
                index_mask = np.arange(len(mask))[mask]
                data_obs = obs[order, mask]
                wave_obs = wave_img[order, mask]

                threshold_of_peak_closeness = (
                    np.diff(wave_obs) / wave_obs[:-1] * speed_of_light
                )
                threshold_of_peak_closeness = np.max(threshold_of_peak_closeness)

                wmin, wmax = wave_obs[0], wave_obs[-1]
                imin, imax = np.searchsorted(self.atlas.wave, (wmin, wmax))
                wave_atlas = self.atlas.wave[imin:imax]
                data_atlas = self.atlas.flux[imin:imax]
                if len(data_atlas) == 0:
                    continue
                data_atlas = data_atlas / data_atlas.max()

                line = lines[
                    (lines["order"] == order)
                    & (lines["wll"] > wmin)
                    & (lines["wll"] < wmax)
                ]

                peaks_atlas, peak_info_atlas = signal.find_peaks(
                    data_atlas, height=0.01, width=width_of_atlas_peaks
                )
                peaks_obs, peak_info_obs = signal.find_peaks(
                    data_obs, height=0.01, width=0
                )

                for i, p in enumerate(peaks_atlas):
                    # Look for an existing line in the vicinityq
                    wpeak = wave_atlas[p]
                    diff = np.abs(line["wll"] - wpeak) / wpeak * speed_of_light
                    if np.any(diff < threshold_of_peak_closeness):
                        # Line already in the linelist, ignore
                        continue
                    else:
                        # Look for matching peak in observation
                        diff = (
                            np.abs(wpeak - wave_obs[peaks_obs]) / wpeak * speed_of_light
                        )
                        imin = np.argmin(diff)

                        if diff[imin] < threshold_of_peak_closeness:
                            # Add line to linelist
                            # Location on the detector
                            # Include the masked areas!!!
                            ipeak = peaks_obs[imin]
                            ipeak = index_mask[ipeak]

                            # relative height of the peak
                            hpeak = data_obs[peaks_obs[imin]]
                            wipeak = peak_info_obs["widths"][imin]
                            # wave, order, pos, width, height, flag
                            new_lines.append([wpeak, order, ipeak, wipeak, hpeak, True])

            # Add new lines to the linelist
            if len(new_lines) != 0:
                new_lines = np.array(new_lines).T
                new_lines = LineList.from_list(*new_lines)
                new_lines = self.fit_lines(obs, new_lines)
                lines.append(new_lines)

        # Option 1:
        # Step 1: Loop over unused lines in lines
        # Step 2: find peaks in neighbourhood
        # Step 3: Toggle flag on if close
        counter = 0
        for i, line in enumerate(lines):
            if line["flag"]:
                # Line is already in use
                continue
            if line["order"] < 0 or line["order"] >= self.nord:
                # Line outside order range
                continue
            iord = int(line["order"])
            if line["wll"] < wave_img[iord][0] or line["wll"] >= wave_img[iord][-1]:
                # Line outside pixel range
                continue

            wl = line["wll"]
            width = line["width"] * 5
            wave = wave_img[iord]
            order_obs = obs[iord]
            # Find where the line should be
            try:
                idx = np.digitize(wl, wave)
            except ValueError:
                # Wavelength solution is not monotonic
                idx = np.where(wave >= wl)[0][0]

            low = int(idx - width)
            low = max(low, 0)
            high = int(idx + width)
            high = min(high, len(order_obs))

            vec = order_obs[low:high]
            if np.all(np.ma.getmaskarray(vec)):
                continue
            # Find the best fitting peak
            # TODO use gaussian fit?
            peak_idx, _ = signal.find_peaks(vec, height=np.ma.median(vec), width=3)
            if len(peak_idx) > 0:
                peak_pos = np.copy(peak_idx).astype(float)
                for j in range(len(peak_idx)):
                    try:
                        coef = self._fit_single_line(vec, peak_idx[j], line["width"])
                        peak_pos[j] = coef[1]
                    except:
                        peak_pos[j] = np.nan
                        pass

                pos_wave = np.interp(peak_pos, np.arange(high - low), wave[low:high])
                residual = np.abs(wl - pos_wave) / wl * speed_of_light
                idx = np.argmin(residual)
                if residual[idx] < self.threshold:
                    counter += 1
                    lines["flag"][i] = True
                    lines["posm"][i] = low + peak_pos[idx]

        logger.info("AutoID identified %i new lines", counter + len(new_lines))

        return lines

    def calculate_residual(self, wave_solution, lines):
        """
        Calculate all residuals of all given lines

        Residual = (Wavelength Solution - Expected Wavelength) / Expected Wavelength * speed of light

        Parameters
        ----------
        wave_solution : array of shape (degree_x, degree_y)
            polynomial coefficients of the wavelength solution (in numpy format)
        lines : recarray of shape (nlines,)
            contains the position of the line on the detector (posm), the order (order), and the expected wavelength (wll)

        Returns
        -------
        residual : array of shape (nlines,)
            Residual of each line in m/s
        """
        x = lines["posm"]
        y = lines["order"]
        mask = ~lines["flag"]

        solution = self.evaluate_solution(x, y, wave_solution)

        residual = (solution - lines["wll"]) / lines["wll"] * speed_of_light
        residual = np.ma.masked_array(residual, mask=mask)
        return residual

    def reject_outlier(self, residual, lines):
        """
        Reject the strongest outlier

        Parameters
        ----------
        residual : array of shape (nlines,)
            residuals of all lines
        lines : recarray of shape (nlines,)
            line data

        Returns
        -------
        lines : struct_array
            line data with one more flagged line
        residual : array of shape (nlines,)
            residuals of each line, with outliers masked (including the new one)
        """

        # Strongest outlier
        ibad = np.ma.argmax(np.abs(residual))
        lines["flag"][ibad] = False

        return lines

    def reject_lines(self, lines, plot=False):
        """
        Reject the largest outlier one by one until all residuals are lower than the threshold

        Parameters
        ----------
        lines : recarray of shape (nlines,)
            Line data with pixel position, and expected wavelength
        threshold : float, optional
            upper limit for the residual, by default 100
        degree : tuple, optional
            polynomial degree of the wavelength solution (pixel, column) (default: (6, 6))
        plot : bool, optional
            Wether to plot the results (default: False)

        Returns
        -------
        lines : recarray of shape (nlines,)
            Line data with updated flags
        """

        wave_solution = self.build_2d_solution(lines)
        residual = self.calculate_residual(wave_solution, lines)
        nbad = 0
        while np.ma.any(np.abs(residual) > self.threshold):
            lines = self.reject_outlier(residual, lines)
            wave_solution = self.build_2d_solution(lines)
            residual = self.calculate_residual(wave_solution, lines)
            nbad += 1
        logger.info("Discarding %i lines", nbad)

        if plot or self.plot >= 2:  # pragma: no cover
            mask = lines["flag"]
            _, axis = plt.subplots()
            axis.plot(lines["order"][mask], residual[mask], "X", label="Accepted Lines")
            axis.plot(
                lines["order"][~mask], residual[~mask], "D", label="Rejected Lines"
            )
            axis.set_xlabel("Order")
            axis.set_ylabel("Residual [m/s]")
            axis.set_title("Residuals versus order")
            axis.legend()

            fig, ax = plt.subplots(
                nrows=self.nord // 2, ncols=2, sharex=True, squeeze=False
            )
            plt.subplots_adjust(hspace=0)
            fig.suptitle("Residuals of each order versus image columns")

            for iord in range(self.nord):
                order_lines = lines[lines["order"] == iord]
                solution = self.evaluate_solution(
                    order_lines["posm"], order_lines["order"], wave_solution
                )
                # Residual in m/s
                residual = (
                    (solution - order_lines["wll"])
                    / order_lines["wll"]
                    * speed_of_light
                )
                mask = order_lines["flag"]
                ax[iord // 2, iord % 2].plot(
                    order_lines["posm"][mask],
                    residual[mask],
                    "X",
                    label="Accepted Lines",
                )
                ax[iord // 2, iord % 2].plot(
                    order_lines["posm"][~mask],
                    residual[~mask],
                    "D",
                    label="Rejected Lines",
                )
                # ax[iord // 2, iord % 2].tick_params(labelleft=False)
                ax[iord // 2, iord % 2].set_ylim(
                    -self.threshold * 1.5, +self.threshold * 1.5
                )

            ax[-1, 0].set_xlabel("x [pixel]")
            ax[-1, 1].set_xlabel("x [pixel]")

            ax[0, 0].legend()

            plt.show()
        return lines

    def plot_results(self, wave_img, obs):
        plt.subplot(211)
        title = "Wavelength solution with Wavelength calibration spectrum\nOrders are in different colours"
        if self.plot_title is not None:
            title = f"{self.plot_title}\n{title}"
        plt.title(title)
        plt.xlabel("Wavelength")
        plt.ylabel("Observed spectrum")
        for i in range(self.nord):
            plt.plot(wave_img[i], obs[i], label="Order %i" % i)

        plt.subplot(212)
        plt.title("2D Wavelength solution")
        plt.imshow(
            wave_img, aspect="auto", origin="lower", extent=(0, self.ncol, 0, self.nord)
        )
        cbar = plt.colorbar()
        plt.xlabel("Column")
        plt.ylabel("Order")
        cbar.set_label("Wavelength [Å]")
        plt.show()

    def plot_residuals(self, lines, coef, title="Residuals"):
        orders = np.unique(lines["order"])
        norders = len(orders)
        if self.plot_title is not None:
            title = f"{self.plot_title}\n{title}"
        plt.suptitle(title)
        nplots = int(np.ceil(norders / 2))
        for i, order in enumerate(orders):
            plt.subplot(nplots, 2, i + 1)
            order_lines = lines[lines["order"] == order]
            if len(order_lines) > 0:
                residual = self.calculate_residual(coef, order_lines)
                plt.plot(order_lines["posm"], residual, "rX")
                plt.hlines([0], 0, self.ncol)

            plt.xlim(0, self.ncol)
            plt.ylim(-self.threshold, self.threshold)

            if (i + 1) not in [norders, norders - 1]:
                plt.xticks([])
            else:
                plt.xlabel("x [Pixel]")

            if (i + 1) % 2 == 0:
                plt.yticks([])
            # else:
            # plt.yticks([-self.threshold, 0, self.threshold])

        plt.subplots_adjust(hspace=0, wspace=0.1)

        # order = 0
        # order_lines = lines[lines["order"] == order]
        # if len(order_lines) > 0:
        #     residual = self.calculate_residual(coef, order_lines)
        #     plt.plot(order_lines["posm"], residual, "rX")
        #     plt.hlines([0], 0, self.ncol)
        # plt.xlim(0, self.ncol)
        # plt.ylim(-self.threshold, self.threshold)
        # plt.xlabel("x [Pixel]")
        # plt.ylabel("Residual [m/s]")

        plt.show()

    def _find_peaks(self, comb):
        # Find peaks in the comb spectrum
        # Run find_peak twice
        # once to find the average distance between peaks
        # once for real (disregarding close peaks)
        c = comb - np.ma.min(comb)
        width = self.lfc_peak_width
        height = np.ma.median(c)
        peaks, _ = signal.find_peaks(c, height=height, width=width)
        distance = np.median(np.diff(peaks)) // 4
        peaks, _ = signal.find_peaks(c, height=height, distance=distance, width=width)

        # Fit peaks with gaussian to get accurate position
        new_peaks = peaks.astype(float)
        width = np.mean(np.diff(peaks)) // 2
        for j, p in enumerate(peaks):
            idx = p + np.arange(-width, width + 1, 1)
            idx = np.clip(idx, 0, len(c) - 1).astype(int)
            try:
                coef = util.gaussfit3(np.arange(len(idx)), c[idx])
                new_peaks[j] = coef[1] + p - width
            except RuntimeError:
                new_peaks[j] = p

        n = np.arange(len(peaks))

        # keep peaks within the range
        mask = (new_peaks > 0) & (new_peaks < len(c))
        n, new_peaks = n[mask], new_peaks[mask]

        return n, new_peaks

    def calculate_AIC(self, lines, wave_solution):
        if self.step_mode:
            if self.dimensionality == "1D":
                k = 1
                for _, v in wave_solution.items():
                    k += np.size(v[0])
                    k += np.size(v[1])
            elif self.dimensionality == "2D":
                k = 1
                poly_coef, steps_coef = wave_solution
                for _, v in steps_coef.items():
                    k += np.size(v)
                k += np.size(poly_coef)
        else:
            k = np.size(wave_solution) + 1

        # We get the residuals in velocity space
        # but need to remove the speed of light component, to get dimensionless parameters
        x = lines["posm"]
        y = lines["order"]
        mask = ~lines["flag"]
        solution = self.evaluate_solution(x, y, wave_solution)
        rss = (solution - lines["wll"]) / lines["wll"]

        # rss = self.calculate_residual(wave_solution, lines)
        # rss /= speed_of_light
        n = rss.size
        rss = np.ma.sum(rss ** 2)

        # As per Wikipedia https://en.wikipedia.org/wiki/Akaike_information_criterion
        logl = np.log(rss)
        aic = 2 * k + n * logl
        self.logl = logl
        self.aicc = aic + (2 * k ** 2 + 2 * k) / (n - k - 1)
        self.aic = aic
        return aic

    def execute(self, obs, lines):
        """
        Perform the whole wavelength calibration procedure with the current settings

        Parameters
        ----------
        obs : array of shape (nord, ncol)
            observed image
        lines : recarray of shape (nlines,)
            reference linelist

        Returns
        -------
        wave_img : array of shape (nord, ncol)
            Wavelength solution for each pixel

        Raises
        ------
        NotImplementedError
            If polarimitry flag is set
        """

        if self.polarim:
            raise NotImplementedError("polarized orders not implemented yet")

        self.nord, self.ncol = obs.shape
        lines = LineList(lines)
        if self.element is not None:
            try:
                self.atlas = LineAtlas(self.element, self.medium)
            except FileNotFoundError:
                logger.warning("No Atlas file found for element %s", self.element)
                self.atlas = None
            except:
                self.atlas = None
        else:
            self.atlas = None

        obs, lines = self.normalize(obs, lines)
        # Step 1: align obs and reference
        lines = self.align(obs, lines)

        # Keep original positions for reference
        lines["posc"] = np.copy(lines["posm"])

        # Step 2: Locate the lines on the detector, and update the pixel position
        # lines["flag"] = True
        lines = self.fit_lines(obs, lines)

        for i in range(self.iterations):
            logger.info(f"Wavelength calibration iteration: {i}")
            # Step 3: Create a wavelength solution on known lines
            wave_solution = self.build_2d_solution(lines)
            wave_img = self.make_wave(wave_solution)
            # Step 4: Identify lines that fit into the solution
            lines = self.auto_id(obs, wave_img, lines)
            # Step 5: Reject outliers
            lines = self.reject_lines(lines)
        # lines = self.reject_lines(lines)

        logger.info(
            "Number of lines used for wavelength calibration: %i",
            np.count_nonzero(lines["flag"]),
        )

        # Step 6: build final 2d solution
        wave_solution = self.build_2d_solution(lines, plot=self.plot)
        wave_img = self.make_wave(wave_solution)

        if self.plot:
            self.plot_results(wave_img, obs)

        aic = self.calculate_AIC(lines, wave_solution)
        logger.info("AIC of wavelength fit: %f", aic)

        # np.savez("cs_lines.npz", cs_lines=lines.data)

        return wave_img, wave_solution


class WavelengthCalibrationComb(WavelengthCalibration):
    def execute(self, comb, wave, lines=None):
        self.nord, self.ncol = comb.shape

        # TODO give everything better names
        pixel, order, wavelengths = [], [], []
        n_all, f_all = [], []
        comb = np.ma.masked_array(comb, mask=comb <= 0)

        for i in range(self.nord):
            # Find Peak positions in current order
            n, peaks = self._find_peaks(comb[i])

            # Determine the n-offset of this order, relative to the anchor frequency
            # Use the existing absolute wavelength calibration as reference
            y_ord = np.full(len(peaks), i)
            w_old = interp1d(np.arange(len(wave[i])), wave[i], kind="cubic")(peaks)
            f_old = speed_of_light / w_old

            # fr: repeating frequency
            # fd: anchor frequency of this order, needs to be shifted to the absolute reference frame
            fr = np.median(np.diff(f_old))
            fd = np.median(f_old % fr)
            n_raw = (f_old - fd) / fr
            n = np.round(n_raw)

            if np.any(np.abs(n_raw - n) > 0.3):
                logger.warning(
                    "Bad peaks detected in the frequency comb in order %i", i
                )

            fr, fd = polyfit(n, f_old, deg=1)

            n_offset = 0
            # The first order is used as the baseline for all other orders
            # The choice is arbitrary and doesn't matter
            if i == 0:
                f0 = fd
                n_offset = 0
            else:
                # n0: shift in n, relative to the absolute reference
                # shift n to the absolute grid, so that all peaks are given by the same f0
                n_offset = (f0 - fd) / fr
                n_offset = int(round(n_offset))
                n -= n_offset
                fd += n_offset * fr

            n = np.abs(n)

            n_all += [n]
            f_all += [f_old]
            pixel += [peaks]
            order += [y_ord]

            logger.debug(
                "LFC Order: %i, f0: %.3f, fr: %.5f, n0: %.2f", i, fd, fr, n_offset
            )

        # Here we postualte that m * lambda = const
        # where m is the peak number
        # this is the result of the grating equation
        # at least const is roughly constant for neighbouring peaks
        correct = True
        if correct:
            w_all = [speed_of_light / f for f in f_all]
            mw_all = [m * w for m, w in zip(n_all, w_all)]
            y = np.concatenate(mw_all)
            gap = np.median(y)

            corr = np.zeros(self.nord)
            for i in range(self.nord):
                corri = gap / w_all[i] - n_all[i]
                corri = np.median(corri)
                corr[i] = np.round(corri)
                n_all[i] += corr[i]

            logger.debug("LFC order offset correction: %s", corr)

            for i in range(self.nord):
                coef = polyfit(n_all[i], n_all[i] * w_all[i], deg=5)
                mw = np.polyval(coef, n_all[i])
                w_all[i] = mw / n_all[i]
                f_all[i] = speed_of_light / w_all[i]

        # Merge Data
        n_all = np.concatenate(n_all)
        f_all = np.concatenate(f_all)
        pixel = np.concatenate(pixel)
        order = np.concatenate(order)

        # Fit f0 and fr to all data
        # (fr, f0), cov = np.polyfit(n_all, f_all, deg=1, cov=True)
        fr, f0 = polyfit(n_all, f_all, deg=1)

        logger.debug("Laser Frequency Comb Anchor Frequency: %.3f 10**10 Hz", f0)
        logger.debug("Laser Frequency Comb Repeating Frequency: %.5f 10**10 Hz", fr)

        # All peaks are then given by f0 + n * fr
        wavelengths = speed_of_light / (f0 + n_all * fr)

        flag = np.full(len(wavelengths), True)
        laser_lines = np.rec.fromarrays(
            (wavelengths, pixel, pixel, order, flag),
            names=("wll", "posm", "posc", "order", "flag"),
        )

        # Use now better resolution to find the new solution
        # A single pass of discarding outliers should be enough
        coef = self.build_2d_solution(laser_lines)
        # resid = self.calculate_residual(coef, laser_lines)
        # laser_lines["flag"] = np.abs(resid) < self.threshold
        # coef = self.build_2d_solution(laser_lines)
        new_wave = self.make_wave(coef)

        aic = self.calculate_AIC(laser_lines, coef)

        self.n_lines_good = np.count_nonzero(laser_lines["flag"])
        logger.info(
            f"Laser Frequency Comb solution based on {self.n_lines_good} lines."
        )
        if self.plot:
            residual = wave - new_wave
            residual = residual.ravel()

            area = np.percentile(residual, (32, 50, 68))
            area = area[0] - 5 * (area[1] - area[0]), area[0] + 5 * (area[2] - area[1])
            plt.hist(residual, bins=100, range=area)
            title = "ThAr - LFC"
            if self.plot_title is not None:
                title = f"{self.plot_title}\n{title}"
            plt.title(title)
            plt.xlabel(r"$\Delta\lambda$ [Å]")
            plt.ylabel("N")
            plt.show()

        if self.plot:
            if lines is not None:
                self.plot_residuals(
                    lines,
                    coef,
                    title="GasLamp Line Residuals in the Laser Frequency Comb Solution",
                )
            self.plot_residuals(
                laser_lines,
                coef,
                title="Laser Frequency Comb Peak Residuals in the LFC Solution",
            )

        if self.plot:
            wave_img = wave
            title = "Difference between GasLamp Solution and Laser Frequency Comb solution\nEach plot shows one order"
            if self.plot_title is not None:
                title = f"{self.plot_title}\n{title}"
            plt.suptitle(title)
            for i in range(len(new_wave)):
                plt.subplot(len(new_wave) // 4 + 1, 4, i + 1)
                plt.plot(wave_img[i] - new_wave[i])
            plt.show()

        if self.plot:
            self.plot_results(new_wave, comb)

        return new_wave


class WavelengthCalibrationInitialize(WavelengthCalibration):
    def __init__(
        self,
        degree=2,
        plot=False,
        plot_title="Wavecal Initial",
        wave_delta=20,
        nwalkers=100,
        steps=50_000,
        resid_delta=1000,
        cutoff=5,
        smoothing=0,
        element="thar",
        medium="vac",
    ):
        super().__init__(
            degree=degree,
            element=element,
            medium=medium,
            plot=plot,
            plot_title=plot_title,
            dimensionality="1D",
        )
        #:float: wavelength uncertainty on the initial guess in Angstrom
        self.wave_delta = wave_delta
        #:int: number of walkers in the MCMC
        self.nwalkers = nwalkers
        #:int: number of steps in the MCMC
        self.steps = steps
        #:float: residual uncertainty allowed when matching observation with known lines
        self.resid_delta = resid_delta
        #:float: gaussian smoothing applied to the wavecal spectrum before the MCMC in pixel scale, disable it by setting it to 0
        self.smoothing = smoothing
        #:float: minimum value in the spectrum to be considered a spectral line, if the value is above (or equal 1) it defines the percentile of the spectrum
        self.cutoff = cutoff

    def get_cutoff(self, spectrum):
        if self.cutoff == 0:
            cutoff = None
        elif self.cutoff < 1:
            cutoff = self.cutoff
        else:
            cutoff = np.nanpercentile(spectrum[spectrum != 0], self.cutoff)
        return cutoff

    def normalize(self, spectrum):
        smoothing = self.smoothing
        spectrum = np.copy(spectrum)
        spectrum -= np.nanmedian(spectrum)
        if smoothing != 0:
            spectrum = gaussian_filter1d(spectrum, smoothing)
        spectrum[spectrum < 0] = 0
        spectrum /= np.max(spectrum)
        return spectrum

    def determine_wavelength_coefficients(
        self,
        spectrum,
        atlas,
        wave_range,
    ) -> np.ndarray:
        """
        Determines the wavelength polynomial coefficients of a spectrum,
        based on an line atlas with known spectral lines,
        and an initial guess for the wavelength range.
        The calculation uses an MCMC approach to sample the probability space and
        find the best cross correlation value, between observation and atlas.

        Parameters
        ----------
        spectrum : array
            observed spectrum at each pixel
        atlas : LineAtlas
            atlas containing a known spectrum with wavelength and flux
        wave_range : 2-tuple
            initial wavelength guess (begin, end)
        degrees : int, optional
            number of degrees of the wavelength polynomial,
            lower numbers yield better results, by default 2
        w_range : float, optional
            uncertainty on the initial wavelength guess in Ansgtrom, by default 20
        nwalkers : int, optional
            number of walkers for the MCMC, more is better but increases
            the time, by default 100
        steps : int, optional
            number of steps in the MCMC per walker, more is better but increases
            the time, by default 20_000
        plot : bool, optional
            whether to plot the results or not, by default False

        Returns
        -------
        coef : array
            polynomial coefficients in numpy order
        """
        spectrum = np.asarray(spectrum)

        assert self.degree >= 2, "The polynomial degree must be at least 2"
        assert spectrum.ndim == 1, "The spectrum should only have 1 dimension"
        assert self.wave_delta > 0, "The wavelength uncertainty needs to be positive"

        n_features = spectrum.shape[0]
        n_output = ndim = self.degree + 1

        # Normalize the spectrum, and copy it just in case
        spectrum = self.normalize(spectrum)
        cutoff = self.get_cutoff(spectrum)

        # The pixel scale used for everything else
        x = np.arange(n_features)
        # Initial guess for the wavelength solution
        coef = np.zeros(n_output)
        coef[-1] = wave_range[0]
        coef[-2] = (wave_range[-1] - wave_range[0]) / n_features

        # We scale every coefficient to roughly order 1
        # this is then in units of the maximum offset due to a change in this value
        # in angstrom
        w_scale = 1 / np.power(n_features, range(n_output))
        factors = w_scale[::-1]
        coef /= factors

        # Here we define the functions we need for the MCMC
        def polyval_vectorize(p, x, where=None):
            n_poly, n_coef = p.shape
            n_points = x.shape[0]
            y = np.zeros((n_poly, n_points))
            if where is not None:
                for i in range(n_coef):
                    y[where] *= x
                    y[where] += p[where, i, None]
            else:
                for i in range(n_coef):
                    y *= x
                    y += p[:, i, None]
            return y

        def log_prior(p):
            prior = np.zeros(p.shape[0])
            prior[np.any(~np.isfinite(p), axis=1)] = -np.inf
            prior[np.any(np.abs(p - coef) > self.wave_delta, axis=1)] = -np.inf
            return prior

        def log_prior_2(w):
            # Chech that w is increasing
            prior = np.zeros(w.shape[0])
            prior[np.any(w[:, 1:] < w[:, :-1], axis=1)] = -np.inf
            prior[w[:, 0] < wave_range[0] - self.wave_delta] = -np.inf
            prior[w[:, -1] > wave_range[1] + self.wave_delta] = -np.inf
            return prior

        def log_prob(p):
            # Check that p is within bounds
            prior = log_prior(p)
            where = np.isfinite(prior)
            # Calculate the wavelength scale
            w = polyval_vectorize(p * factors, x, where=where)
            # Check that it is monotonically increasing
            prior += log_prior_2(w)
            where = np.isfinite(prior)

            y = np.zeros((p.shape[0], x.shape[0]))
            y[where, :] = np.interp(w[where, :], atlas.wave, atlas.flux)
            y[where, :] /= np.max(y[where, :], axis=1)[:, None]
            # This is the cross correlation value squared
            cross = np.sum(y * spectrum, axis=1) ** 2
            # chi2 = - np.sum((y - spectrum)**2, axis=1)
            # chi2 = - np.sum((np.where(y > 0.01, 1, 0) - np.where(spectrum > 0.01, 1, 0))**2, axis=1)
            # this is the same as above, but a lot faster thanks to the magic of bitwise xor
            if cutoff is not None:
                chi2 = (y > cutoff) ^ (spectrum > cutoff)
                chi2 = -np.count_nonzero(chi2, axis=1) / 20
            else:
                chi2 = -np.sum((y - spectrum) ** 2, axis=1) / 20
            return prior + cross + chi2

        p0 = np.zeros((self.nwalkers, ndim))
        p0 += coef[None, :]
        p0 += np.random.uniform(
            low=-self.wave_delta, high=self.wave_delta, size=(self.nwalkers, ndim)
        )
        sampler = emcee.EnsembleSampler(
            self.nwalkers,
            ndim,
            log_prob,
            vectorize=True,
            moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
        )
        state = sampler.run_mcmc(p0, self.steps, progress=True)

        tau = sampler.get_autocorr_time(quiet=True)
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

        low, mid, high = np.percentile(samples, [32, 50, 68], axis=0)
        coef = mid * factors

        if self.plot:
            fig = corner.corner(samples, truths=mid)
            plt.show()

            wave = np.polyval(coef, x)
            y = np.interp(wave, atlas.wave, atlas.flux)
            y /= np.max(y)
            plt.plot(wave, spectrum)
            plt.plot(wave, y)
            plt.show()

        return coef

    def create_new_linelist_from_solution(
        self,
        spectrum,
        wavelength,
        atlas,
        order,
    ) -> LineList:
        """
        Create a new linelist based on an existing wavelength solution for a spectrum,
        and a line atlas with known lines. The linelist is the one used by the rest of
        PyReduce wavelength calibration.

        Observed lines are matched with the lines in the atlas to
        improve the wavelength solution.

        Parameters
        ----------
        spectrum : array
            Observed spectrum at each pixel
        wavelength : array
            Wavelength of spectrum at each pixel
        atlas : LineAtlas
            Atlas with wavelength of known lines
        order : int
            Order of the spectrum within the detector
        resid_delta : float, optional
            Maximum residual allowed between a peak and the closest line in the atlas,
            to still match them, in m/s, by default 1000.

        Returns
        -------
        linelist : LineList
            new linelist with lines from this order
        """
        # The new linelist
        linelist = LineList()
        spectrum = np.asarray(spectrum)
        wavelength = np.asarray(wavelength)

        assert self.resid_delta > 0, "Residuals Delta must be positive"
        assert spectrum.ndim == 1, "Spectrum must have only 1 dimension"
        assert wavelength.ndim == 1, "Wavelength must have only 1 dimension"
        assert (
            spectrum.size == wavelength.size
        ), "Spectrum and Wavelength must have the same size"

        n_features = spectrum.shape[0]
        x = np.arange(n_features)
        smoothing = self.smoothing

        # Normalize just in case
        spectrum = self.normalize(spectrum)
        cutoff = self.get_cutoff(spectrum)

        # TODO: make this use another function, and pass the hight as a parameter
        scopy = np.copy(spectrum)
        if cutoff is not None:
            scopy[scopy < cutoff] = 0
        _, peaks = self._find_peaks(scopy)

        peak_wave = np.interp(peaks, x, wavelength)
        peak_height = np.interp(peaks, x, spectrum)

        # Here we only look at the lines within range
        atlas_linelist = atlas.linelist[
            (atlas.linelist["wave"] > wavelength[0])
            & (atlas.linelist["wave"] < wavelength[-1])
        ]

        residuals = np.zeros_like(peak_wave)
        for i, pw in enumerate(peak_wave):
            resid = np.abs(pw - atlas_linelist["wave"])
            j = np.argmin(resid)
            residuals[i] = resid[j] / pw * speed_of_light
            if residuals[i] < self.resid_delta:
                linelist.add_line(
                    atlas_linelist["wave"][j],
                    order,
                    peaks[i],
                    3,
                    peak_height[i],
                    True,
                )

        return linelist

    def execute(self, spectrum, wave_range) -> LineList:
        atlas = LineAtlas(self.element, self.medium)
        linelist = LineList()
        orders = range(spectrum.shape[0])
        x = np.arange(spectrum.shape[1])
        for order in orders:
            spec = spectrum[order]
            wrange = wave_range[order]
            coef = self.determine_wavelength_coefficients(spec, atlas, wrange)
            wave = np.polyval(coef, x)
            linelist_loc = self.create_new_linelist_from_solution(
                spec, wave, atlas, order
            )
            linelist.append(linelist_loc)
        return linelist
