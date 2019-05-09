"""
Wavelength Calibration
by comparison to a reference spectrum
Loosely bases on the IDL wavecal function
"""

import datetime as dt
import logging

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval2d
from scipy import signal
from scipy.constants import speed_of_light
from scipy.optimize import curve_fit, least_squares

from . import util
from .instruments import instrument_info


class AlignmentPlot:
    """
    Makes a plot which can be clicked to align the two spectra, reference and observed
    """

    def __init__(self, ax, thar, cs_lines, offset=[0, 0]):
        self.im = ax
        self.first = True
        self.nord, self.ncol = thar.shape
        self.RED, self.GREEN, self.BLUE = 0, 1, 2

        self.thar = thar
        self.cs_lines = cs_lines

        self.order_first = 0
        self.spec_first = ""
        self.x_first = 0
        self.offset = offset

        self.make_ref_image()

    def make_ref_image(self):
        """ create and show the reference plot, with the two spectra """
        ref_image = np.zeros((self.nord * 2, self.ncol, 3))
        for iord in range(self.nord):
            ref_image[iord * 2, :, self.RED] = 10 * np.ma.filled(self.thar[iord], 0)
            if 0 <= iord + self.offset[0] < self.nord:
                for line in self.cs_lines[self.cs_lines["order"] == iord]:
                    first = np.clip(line["xfirst"] + self.offset[1], 0, self.ncol)
                    last = np.clip(line["xlast"] + self.offset[1], 0, self.ncol)
                    ref_image[
                        (iord + self.offset[0]) * 2 + 1, first:last, self.GREEN
                    ] = (
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
            extent=(0, self.ncol, 0, self.nord),
        )
        self.im.figure.suptitle(
            "Alignment, Observed: RED, Reference: GREEN\nGreen should be above red!"
        )
        self.im.axes.set_xlabel("x [pixel]")
        self.im.axes.set_ylabel("Order")

        self.im.figure.canvas.draw()

    def connect(self):
        """ connect the click event with the appropiate function """
        self.cidclick = self.im.figure.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

    def on_click(self, event):
        """ On click offset the reference by the distance between click positions """
        if event.ydata is None:
            return
        order = int(np.floor(event.ydata))
        spec = (
            "ref" if (event.ydata - order) > 0.5 else "thar"
        )  # if True then reference
        x = event.xdata
        print("Order: %i, Spectrum: %s, x: %g" % (order, "ref" if spec else "thar", x))

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
                self.offset[0] += offset_orders
                self.offset[1] += offset_x
                self.make_ref_image()


def create_image_from_lines(cs_lines, ncol):
    """
    Create a reference image based on a line list
    Each line will be approximated by a Gaussian
    Space inbetween lines is 0
    The number of orders is from 0 to the maximum order

    Parameters
    ----------
    cs_lines : recarray of shape (nlines,)
        line data
    ncol : int
        Number of Columns in the x direction

    Returns
    -------
    img : array of shape (nord, ncol)
        New reference image
    """
    min_order = np.min(cs_lines["order"])
    max_order = np.max(cs_lines["order"])
    img = np.zeros((max_order - min_order + 1, ncol))
    for line in cs_lines:
        if line["order"] < 0:
            continue
        if line["xlast"] < 0 or line["xfirst"] > ncol:
            continue
        first = max(line["xfirst"], 0)
        last = min(line["xlast"], ncol)
        img[line["order"] - min_order, first:last] = line["height"] * signal.gaussian(
            last - first, line["width"]
        )
    return img


def align(thar, cs_lines, shift_window=0.01, manual=False, plot=False):
    """Align the observation with the reference spectrum
    Either automatically using cross correlation or manually (visually)

    Parameters
    ----------
    thar : array[nrow, ncol]
        observed wavelength calibration spectrum (e.g. ThAr=ThoriumArgon)
    cs_lines : struct_array
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
    nord, ncol = thar.shape
    thar = np.ma.filled(thar, 0)

    # Align using window like in IDL REDUCE
    if manual:
        _, ax = plt.subplots()
        ap = AlignmentPlot(ax, thar, cs_lines)
        ap.connect()
        plt.show()
        offset = ap.offset

        cs_lines["xfirst"] += offset[1]
        cs_lines["xlast"] += offset[1]
        cs_lines["posm"] += offset[1]
        cs_lines["order"] += offset[0]
    else:
        # make image from cs_lines
        img = create_image_from_lines(cs_lines, ncol)

        # Cross correlate with thar image
        # And determine overall offset
        correlation = signal.correlate2d(thar, img, mode="same")
        offset_order, offset_x = np.unravel_index(
            np.argmax(correlation), correlation.shape
        )

        offset_order = offset_order - img.shape[0] / 2 + 1
        offset_x = offset_x - img.shape[1] / 2 + 1
        offset = [int(offset_order), int(offset_x)]

        # apply offset
        cs_lines["xfirst"] += offset[1]
        cs_lines["xlast"] += offset[1]
        cs_lines["posm"] += offset[1]
        cs_lines["order"] += offset[0]

        # Shift individual orders to fit reference
        # Only allow a small shift here (1%) ?
        img = create_image_from_lines(cs_lines, ncol)

        for i in range(offset[0], min(len(thar), len(img))):
            correlation = signal.correlate(thar[i], img[i], mode="same")
            width = ncol // (2 * int(1 / shift_window))
            low = ncol // 2 - width
            high = ncol // 2 + width
            offset_x = np.argmax(correlation[low:high]) + low
            offset_x = int(offset_x - ncol / 2)
            select = cs_lines["order"] == i
            cs_lines["posm"][select] += offset_x
            cs_lines["xfirst"][select] += offset_x
            cs_lines["xlast"][select] += offset_x

        if plot:
            # Even without manual=True, allow the user to shift the image here
            # Provided it is plotted at all
            _, ax = plt.subplots()
            ap = AlignmentPlot(ax, thar, cs_lines, offset=[0, 0])
            ap.connect()
            plt.show()
            offset = ap.offset

            cs_lines["xfirst"] += offset[1]
            cs_lines["xlast"] += offset[1]
            cs_lines["posm"] += offset[1]
            cs_lines["order"] += offset[0]

    logging.debug(f"Offset order: {offset[0]}, Offset pixel: {offset[1]}")

    return offset


def fit_lines(thar, cs_lines, plot=False):
    """
    Determine exact position of each line on the detector based on initial guess

    This fits a Gaussian to each line, and uses the peak position as a new solution

    Parameters
    ----------
    thar : array of shape (nord, ncol)
        observed wavelength calibration image
    cs_lines : recarray of shape (nlines,)
        reference line data

    Returns
    -------
    cs_lines : recarray of shape (nlines,)
        Updated line information (posm is changed)
    """
    # For each line fit a gaussian to the observation
    for i, line in enumerate(cs_lines):
        if line["posm"] < 0 or line["posm"] >= thar.shape[1]:
            # Line outside pixel range
            continue
        if line["order"] < 0 or line["order"] >= len(thar):
            # Line outside order range
            continue
        low = int(line["posm"] - line["width"] * 5)
        low = max(low, 0)
        high = int(line["posm"] + line["width"] * 5)
        high = min(high, len(thar[line["order"]]))

        section = thar[line["order"], low:high]
        x = np.arange(low, high, 1)
        x = np.ma.masked_array(x, mask=np.ma.getmaskarray(section))

        try:
            coef = util.gaussfit2(x, section)
            cs_lines[i]["posm"] = coef[1]
        except:
            # Gaussian fit failed, dont use line
            cs_lines[i]["flag"] = False

        if plot:
            x2 = np.linspace(x.min(), x.max(), len(x) * 100)
            plt.plot(x, section, label="Observation")
            plt.plot(x2, util.gaussval2(x2, *coef), label="Fit")
            plt.title("Gaussian Fit to spectral line")
            plt.xlabel("x [pixel]")
            plt.ylabel("Intensity [a.u.]")
            plt.legend()
            plt.show()

    return cs_lines


def build_2d_solution(cs_lines, degree=(6, 6), mode="1D", plot=False):
    """
    Create a 2D polynomial fit to flagged lines

    Parameters
    ----------
    cs_lines : struc_array
        line data
    degree : tuple(int, int), optional
        polynomial degree of the fit in (column, order) dimension (default: (6, 6))
    plot : bool, optional
        wether to plot the solution (default: False)

    Returns
    -------
    coef : array[degree_x, degree_y]
        2d polynomial coefficients
    """
    # 2d polynomial fit with: x = column, y = order, z = wavelength
    degree_x, degree_y = degree

    # Only use flagged data
    mask = cs_lines["flag"]  # True: use line, False: dont use line
    m_wave = cs_lines["wll"][mask]
    m_pix = cs_lines["posm"][mask]
    m_ord = cs_lines["order"][mask]

    if mode == "1D":
        nord = m_ord.max() + 1
        coef = np.zeros((nord, degree_x + 1))
        for i in range(nord):
            select = m_ord == i
            coef[i] = np.polyfit(m_pix[select], m_wave[select], deg=degree_x)
    elif mode == "2D":
        coef = util.polyfit2d(
            m_pix, m_ord, m_wave, degree=(degree_x, degree_y), plot=False
        )
    else:
        raise ValueError(
            f"Parameter 'mode' not understood. Expected '1D' or '2D' but got {mode}"
        )

    if plot:
        orders = np.unique(cs_lines["order"])
        norders = len(orders)
        for i, order in enumerate(orders):
            plt.subplot(int(np.ceil(norders / 2)), 2, i + 1)
            lines = cs_lines[mask][cs_lines[mask]["order"] == order]
            if len(lines) > 0:
                residual = calculate_residual(coef, lines, mode=mode)
                plt.plot(lines["posm"], residual, "rx")
                plt.hlines([0], lines["posm"].min(), lines["posm"].max())
        plt.show()

    return coef


def evaluate_solution(pos, order, solution, mode="1D"):
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
    if not np.array_equal(pos.shape, order.shape):
        raise ValueError("pos and order must have the same shape")
    if mode == "1D":
        result = np.zeros(pos.shape)
        for i in np.unique(order):
            select = order == i
            result[select] = np.polyval(solution[i], pos[select])
    elif mode == "2D":
        result = np.polynomial.polynomial.polyval2d(pos, order, solution)
    else:
        raise ValueError(
            f"parameter 'mode' not understood, expected '1D' or '2D' but got {mode}"
        )
    return result


def make_wave(thar, wave_solution, mode="1D", plot=False):
    """Expand polynomial wavelength solution into full image

    Parameters
    ----------
    thar : array[nord, ncol]
        observed wavelength spectrum
    wave_solution : array[degree, degree]
        polynomial coefficients of wavelength solution
    plot : bool, optional
        wether to plot the solution (default: False)

    Returns
    -------
    wave_img : array[nord, ncol]
        wavelength solution for each point in the spectrum
    """

    nord, ncol = thar.shape
    y, x = np.indices((nord, ncol))
    wave_img = evaluate_solution(x, y, wave_solution, mode=mode)

    if plot:
        plt.subplot(211)
        plt.title(
            "Wavelength solution with Wavelength calibration spectrum\nOrders are in different colours"
        )
        plt.xlabel("Wavelength")
        plt.ylabel("Observed spectrum")
        for i in range(thar.shape[0]):
            plt.plot(wave_img[i], thar[i], label="Order %i" % i)
        # plt.legend(loc="best")

        plt.subplot(212)
        plt.title("2D Wavelength solution")
        plt.imshow(wave_img, aspect="auto", origin="lower", extent=(0, ncol, 0, nord))
        cbar = plt.colorbar()
        plt.xlabel("Column")
        plt.ylabel("Order")
        cbar.set_label("Wavelength [Å]")
        plt.show()

    return wave_img


def auto_id(thar, wave_img, cs_lines, threshold=100, plot=False):
    """Automatically identify peaks that are close to known lines 

    Parameters
    ----------
    thar : array[nord, ncol]
        observed spectrum
    wave_img : array[nord, ncol]
        wavelength solution image
    cs_lines : struc_array
        line data
    threshold : int, optional
        difference threshold between line positions in m/s, until which a line is considered identified (default: 1)
    plot : bool, optional
        wether to plot the new lines

    Returns
    -------
    cs_line : struct_array
        line data with new flags
    """

    # TODO: auto ID based on wavelength solution or position on detector?
    # Set flags in cs_lines
    nord, ncol = thar.shape

    # Option 1:
    # Step 1: Loop over unused lines in cs_lines
    # Step 2: find peaks in neighbourhood
    # Step 3: Toggle flag on if close
    counter = 0
    for i, line in enumerate(cs_lines):
        if line["flag"]:
            # Line is already in use
            continue
        if line["order"] < 0 or line["order"] >= nord:
            # Line outside order range
            continue
        iord = line["order"]
        if line["wll"] < wave_img[iord][0] or line["posm"] >= wave_img[iord][-1]:
            # Line outside pixel range
            continue

        wl = line["wll"]
        width = line["width"] * 10
        wave = wave_img[iord]
        obs = thar[iord]
        # Find where the line should be
        try:
            idx = np.digitize(wl, wave)
        except ValueError:
            # Wavelength solution is not monotonic
            idx = np.where(wave >= wl)[0][0]

        low = int(idx - width)
        low = max(low, 0)
        high = int(idx + width)
        high = min(high, len(obs))

        vec = obs[low:high]
        if np.all(np.ma.getmaskarray(vec)):
            continue
        # Find the best fitting peak
        # TODO use gaussian fit?
        peak_idx, _ = signal.find_peaks(vec, height=np.ma.median(vec))
        if len(peak_idx) > 0:
            pos_wave = wave[low:high][peak_idx]
            residual = np.abs(wl - pos_wave) / wl * speed_of_light
            idx = np.argmin(residual)
            if residual[idx] < threshold:
                counter += 1
                cs_lines["flag"][i] = True
                cs_lines["posm"][i] = low + peak_idx[idx]

    logging.info(f"AutoID identified {counter} new lines")

    return cs_lines


def calculate_residual(wave_solution, cs_lines, mode="1D"):
    """
    Calculate all residuals of all given lines

    Residual = (Wavelength Solution - Expected Wavelength) / Expected Wavelength * speed of light

    Parameters
    ----------
    wave_solution : array of shape (degree_x, degree_y)
        polynomial coefficients of the wavelength solution (in numpy format)
    cs_lines : recarray of shape (nlines,)
        contains the position of the line on the detector (posm), the order (order), and the expected wavelength (wll)

    Returns
    -------
    residual : array of shape (nlines,)
        Residual of each line in m/s
    """
    x = cs_lines["posm"]
    y = cs_lines["order"]
    mask = ~cs_lines["flag"]

    solution = evaluate_solution(x, y, wave_solution, mode=mode)

    residual = (solution - cs_lines["wll"]) / cs_lines["wll"] * speed_of_light
    residual = np.ma.masked_array(residual, mask=mask)
    return residual


def reject_outlier(residual, cs_lines):
    """
    Reject the strongest outlier

    Parameters
    ----------
    residual : array of shape (nlines,)
        residuals of all lines
    cs_lines : recarray of shape (nlines,)
        line data

    Returns
    -------
    cs_lines : struct_array
        line data with one more flagged line
    residual : array of shape (nlines,)
        residuals of each line, with outliers masked (including the new one)
    """

    # Strongest outlier
    ibad = np.ma.argmax(np.abs(residual))
    cs_lines["flag"][ibad] = False

    return cs_lines


def reject_lines(cs_lines, nord, threshold=100, degree=(6, 6), mode="1D", plot=False):
    """
    Reject the largest outlier one by one until all residuals are lower than the threshold

    Parameters
    ----------
    cs_lines : recarray of shape (nlines,)
        Line data with pixel position, and expected wavelength
    nord : int
        number of orders, only used for plotting
    threshold : float, optional
        upper limit for the residual, by default 100
    degree : tuple, optional
        polynomial degree of the wavelength solution (pixel, column) (default: (6, 6))
    plot : bool, optional
        Wether to plot the results (default: False)

    Returns
    -------
    cs_lines : recarray of shape (nlines,)
        Line data with updated flags
    """

    wave_solution = build_2d_solution(cs_lines, degree=degree, mode=mode)
    residual = calculate_residual(wave_solution, cs_lines, mode=mode)
    nbad = 0
    while np.ma.any(np.abs(residual) > threshold):
        cs_lines = reject_outlier(residual, cs_lines)
        wave_solution = build_2d_solution(cs_lines, degree=degree, mode=mode)
        residual = calculate_residual(wave_solution, cs_lines, mode=mode)
        nbad += 1
    logging.info("Discarding %i lines", nbad)

    if plot:
        mask = cs_lines["flag"]
        _, axis = plt.subplots()
        axis.plot(cs_lines["order"][mask], residual[mask], "+", label="Accepted Lines")
        axis.plot(
            cs_lines["order"][~mask], residual[~mask], "d", label="Rejected Lines"
        )
        axis.set_xlabel("Order")
        axis.set_ylabel("Residual [m/s]")
        axis.set_title("Residuals versus order")
        axis.legend()

        fig, ax = plt.subplots(nrows=nord // 2, ncols=2, sharex=True, squeeze=False)
        plt.subplots_adjust(hspace=0)
        fig.suptitle("Residuals of each order versus image columns")

        for iord in range(nord):
            lines = cs_lines[cs_lines["order"] == iord]
            solution = evaluate_solution(
                lines["posm"], lines["order"], wave_solution, mode=mode
            )
            # Residual in m/s
            residual = (solution - lines["wll"]) / lines["wll"] * speed_of_light
            mask = lines["flag"]
            ax[iord // 2, iord % 2].plot(
                lines["posm"][mask], residual[mask], "+", label="Accepted Lines"
            )
            ax[iord // 2, iord % 2].plot(
                lines["posm"][~mask], residual[~mask], "d", label="Rejected Lines"
            )
            # ax[iord // 2, iord % 2].tick_params(labelleft=False)
            ax[iord // 2, iord % 2].set_ylim(-threshold * 1.5, +threshold * 1.5)

        ax[-1, 0].set_xlabel("x [pixel]")
        ax[-1, 1].set_xlabel("x [pixel]")

        ax[0, 0].legend()

        plt.show()
    return cs_lines


def wavecal(
    thar,
    cs_lines,
    threshold=1000,
    degree_x=6,
    degree_y=6,
    iterations=3,
    mode="1D",
    shift_window=0.01,
    manual=False,
    polarim=False,
    plot=True,
):
    """Wavelength calibration wrapper

    Parameters
    ----------
    thar : array[nord, ncol]
        observed wavelength spectrum
    cs_lines : struct_array
        line data
    threshold : float
        Upper limit for the residuals of lines to use in m/s (default: 100)
    degree_x : int, optional
        polynomial degree of the solution in pixel direction (default: 6)
    degree_y : int, optional
        polynomial degree of the solution in order direction (default: 6)
    iterations : int, optional
        Number of AutoID, Rejection cycles to use for this wavelength calibration (default: 3)
    manual : bool, optional
        if True use manual alignment, otherwise automatic (default: False)
    polarim : bool, optional
        wether to use polarimetry orders (i.e. orders come in pairs of two), Not supported yet
    plot : bool, optional
        wether to plot the final results (default: True)

    Returns
    -------
    wave_img : array[nord, ncol]
        wavelength solution for each point in the spectrum
    """
    degree = (degree_x, degree_y)
    nord, ncol = thar.shape
    thar = np.ma.copy(thar)
    # normalize each order
    for i in range(len(thar)):
        thar[i] -= np.ma.min(thar[i][thar[i] != 0])
        thar[i] /= np.ma.max(thar[i])
    thar[thar <= 0] = np.ma.masked

    # Normalize lines in each order
    topheight = {}
    for order in np.unique(cs_lines["order"]):
        topheight[order] = np.max(cs_lines[cs_lines["order"] == order]["height"])
    for i, line in enumerate(cs_lines):
        cs_lines[i]["height"] /= topheight[line["order"]]

    if polarim:
        raise NotImplementedError("polarized orders not implemented yet")

    # Step 1: align thar and reference
    align(thar, cs_lines, plot=plot, manual=manual, shift_window=shift_window)

    # Step 2: Locate the lines on the detector, and update the pixel position
    cs_lines = fit_lines(thar, cs_lines)

    for i in range(iterations):
        logging.info(f"Wavelength calibration iteration: {i}")
        # Step 3: Create a wavelength solution on known lines
        wave_solution = build_2d_solution(cs_lines, degree=degree, mode=mode)
        wave_img = make_wave(thar, wave_solution, mode=mode)
        # Step 4: Identify lines that fit into the solution
        cs_lines = auto_id(thar, wave_img, cs_lines, threshold=threshold)
        # Step 5: Reject outliers
        cs_lines = reject_lines(
            cs_lines, nord, threshold=threshold, degree=degree, mode=mode
        )

    logging.info(
        "Number of lines used for wavelength calibration: %i",
        np.count_nonzero(cs_lines["flag"]),
    )

    # Step 6: build final 2d solution
    wave_solution = build_2d_solution(cs_lines, degree=degree, mode=mode, plot=plot)
    wave_img = make_wave(thar, wave_solution, mode=mode, plot=plot)

    return wave_img


if __name__ == "__main__":
    instrument = "UVES"
    mode = "middle"
    target = "HD132205"
    night = "2010-04-02"

    thar_file = "./Test/UVES/{target}/reduced/{night}/Reduced_{mode}/UVES.2010-04-02T11_09_23.851.thar.ech"
    thar_file = thar_file.format(
        instrument=instrument, mode=mode, night=night, target=target
    )

    thar = fits.open(thar_file)
    head = thar[0].header
    thar = thar[1].data["SPEC"][0]

    reference = instrument_info.get_wavecal_filename(head, instrument, mode)
    reference = np.load(reference)

    # cs_lines = "Wavelength_computed", "Wavelength_list", "Position_Computed", "Position_Model", "first x coordinate", "last x coordinate", "approximate ??", "width", "flag", "height", "order"
    # cs_lines = 'WLC', 'WLL', 'POSC', 'POSM', 'XFIRST', 'XLAST', 'APPROX', 'WIDTH', 'FLAG', 'HEIGHT', 'ORDER'
    cs_lines = reference["cs_lines"]

    # list of orders, bad orders marked with 1, normal orders 0
    bad_order = reference["bad_order"]

    solution = wavecal(thar, cs_lines, plot=False)

    for i in range(thar.shape[0]):
        plt.plot(solution[i], thar[i], label="Order %i" % i)

    for line in cs_lines:
        plt.axvline(x=line["wll"], ymax=line["height"])

    plt.show()

    print(solution)
