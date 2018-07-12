import numpy as np
from numpy.polynomial.polynomial import polyval2d

from scipy.io import readsav
from scipy.optimize import curve_fit
from scipy import signal
from scipy.constants import speed_of_light
import astropy.io.fits as fits

from util import save_fits


import matplotlib.pyplot as plt


class AlignmentPlot:
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
        ref_image = np.zeros((self.nord * 2, self.ncol, 3))
        for iord in range(self.nord):
            idx = self.thar[iord] > 0.1
            ref_image[iord * 2, idx, self.RED] = self.thar[iord, idx]
            if 0 <= iord + self.offset[0] < self.nord:
                for line in self.cs_lines[self.cs_lines.order == iord]:
                    first = np.clip(line.xfirst + self.offset[1], 0, self.ncol)
                    last = np.clip(line.xlast + self.offset[1], 0, self.ncol)
                    ref_image[
                        (iord + self.offset[0]) * 2 + 1, first:last, self.GREEN
                    ] = line.height * signal.gaussian(last - first, line.width)

        self.im.imshow(
            ref_image,
            aspect="auto",
            origin="lower",
            extent=(0, self.ncol, 0, self.nord),
        )
        self.im.figure.suptitle(
            "Alignment, ThAr: RED, Reference: GREEN\nGreen should be above red!"
        )
        self.im.figure.canvas.draw()

    def connect(self):
        self.cidclick = self.im.figure.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

    def on_click(self, event):
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


def align(thar, cs_lines, manual=False, plot=False):
    # Align using window like in IDL REDUCE
    if manual:
        _, ax = plt.subplots()
        ap = AlignmentPlot(ax, thar, cs_lines)
        ap.connect()
        plt.show()
        offset = ap.offset
    else:
        # make image from cs_lines
        min_order = np.min(cs_lines.order)
        img = np.zeros_like(thar)
        for line in cs_lines:
            img[line.order, line.xfirst : line.xlast] = line.height * signal.gaussian(
                line.xlast - line.xfirst, line.width
            )
        img = np.ma.masked_array(img, mask=img == 0)

        # Cross correlate with thar image
        correlation = signal.correlate2d(thar, img, mode="same")
        offset_order, offset_x = np.unravel_index(
            np.argmax(correlation), correlation.shape
        )

        offset_order = 2 * offset_order - thar.shape[0] + min_order + 1
        offset_x = offset_x - thar.shape[1] / 2
        offset = int(offset_order), int(offset_x)

        if plot:
            _, ax = plt.subplots()
            AlignmentPlot(ax, thar, cs_lines, offset=offset)
            plt.show()

    return offset


def build_2d_solution(cs_lines, plot=False):
    # Only use flagged data
    # TODO which fields to use for fit?
    mask = ~cs_lines.flag.astype(bool)
    m_wave = cs_lines.wll[mask]
    m_pix = cs_lines.posm[mask]
    m_ord = cs_lines.order[mask]

    # 2d polynomial fit with: x = column, y = order, z = wavelength
    degree_x, degree_y = 5, 5
    degree_x, degree_y = degree_x + 1, degree_y + 1  # Due to how np polyval2d workss

    def func(x, *c):
        c = np.array(c)
        c.shape = degree_x, degree_y
        value = polyval2d(x[0], x[1], c)
        return value

    popt, pcov = curve_fit(
        func, [m_pix, m_ord], m_wave, p0=np.ones(degree_x * degree_y)
    )
    popt.shape = degree_x, degree_y
    return popt


def make_wave(thar, wave_solution, plot=False):
    nord, ncol = thar.shape
    x = np.arange(ncol)
    y = np.arange(nord)
    x, y = np.meshgrid(x, y)
    wave_img = polyval2d(x, y, wave_solution)

    if plot:
        plt.imshow(wave_img, aspect="auto", origin="lower", extent=(0, ncol, 0, nord))
        cbar = plt.colorbar()
        plt.xlabel("Column")
        plt.ylabel("Order")
        cbar.set_label("Wavelength [Å]")
        plt.title("2D Wavelength solution")
        plt.show()

    return wave_img


def auto_id(thar, wave_img, cs_lines, threshold=1, plot=False):
    # TODO: auto ID based on wavelength solution or position on detector?
    # Set flags in cs_lines
    nord, ncol = thar.shape

    # Step 1: find peaks in thar
    # Step 2: compare lines in cs_lines with peaks
    # Step 3: toggle flag on if close enough
    for iord in range(nord):
        # TODO pick good settings
        vec = thar[iord, thar[iord] > 0]
        vec -= np.min(vec)
        peak_idx = signal.find_peaks_cwt(vec, np.arange(2, 5), min_snr=5, min_length=3)
        pos_wave = wave_img[iord, thar[iord] > 0][peak_idx]

        for i, line in enumerate(cs_lines):
            if line.order == iord:
                diff = np.abs(line.wll - pos_wave)
                if np.min(diff) < threshold:
                    cs_lines.flag[i] = 0  # 0 == True, 1 == False

    return cs_lines


def reject_lines(thar, wave_solution, cs_lines, clip=100, plot=True):
    # Calculate residuals
    nord, ncol = thar.shape
    x = cs_lines.posm
    y = cs_lines.order
    solution = polyval2d(x, y, wave_solution)
    # Residual in m/s
    residual = (solution - cs_lines.wll) / cs_lines.wll * speed_of_light
    cs_lines.flag[np.abs(residual) > clip] = 1  # 1 = False
    mask = ~cs_lines.flag.astype(bool)

    if plot:
        _, axis = plt.subplots()
        axis.plot(cs_lines.order[mask], residual[mask], "+")
        axis.plot(cs_lines.order[~mask], residual[~mask], "d")
        axis.set_xlabel("Order")
        axis.set_ylabel("Residual")
        axis.set_title("Residuals over order")

        fig, ax = plt.subplots(nrows=nord // 2, ncols=2, sharex=True)
        plt.subplots_adjust(hspace=0)
        fig.suptitle("Residuals over columns")

        for iord in range(nord):
            lines = cs_lines[cs_lines.order == iord]
            solution = polyval2d(lines.posm, lines.order, wave_solution)
            # Residual in m/s
            residual = (solution - lines.wll) / lines.wll * speed_of_light
            mask = ~lines.flag.astype(bool)
            ax[iord // 2, iord % 2].plot(lines.posm[mask], residual[mask], "+")
            ax[iord // 2, iord % 2].plot(lines.posm[~mask], residual[~mask], "d")
            # ax[iord // 2, iord % 2].tick_params(labelleft=False)
            ax[iord // 2, iord % 2].set_ylim(-clip * 1.5, +clip * 1.5)

        ax[-1, 0].set_xlabel("Column")
        ax[-1, 1].set_xlabel("Column")

        plt.show()

    return cs_lines


def wavecal(thar, cs_lines, plot=True):
    # normalize images
    thar = np.ma.masked_array(thar, mask=thar == 0)
    thar -= np.min(thar)
    thar /= np.max(thar)

    cs_lines.height /= np.max(cs_lines.height)

    # Step 1: align thar and reference
    offset = align(thar, cs_lines, plot=plot)
    # Step 1.5: Apply offset
    # be careful not to apply offset twice
    cs_lines.xfirst += offset[1]
    cs_lines.xlast += offset[1]
    cs_lines.order += offset[0]

    for j in range(3):
        for i in range(5):
            wave_solution = build_2d_solution(cs_lines)
            cs_lines = reject_lines(
                thar, wave_solution, cs_lines, plot=i == 4 and j == 2 and plot
            )

        wave_solution = build_2d_solution(cs_lines)
        wave_img = make_wave(thar, wave_solution)

        cs_lines = auto_id(thar, wave_img, cs_lines)

    # Step 6: build final 2d solution
    wave_solution = build_2d_solution(cs_lines, plot=plot)
    wave_img = make_wave(thar, wave_solution, plot=plot)

    return wave_img  # wavelength solution image


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

    specifier = int(head["ESO INS GRAT2 WLEN"])

    reference = (
        "./wavecal/{instrument}_{mode}_{specifier}nm_2D.sav"
    )  # ESO INS GRAT2 WLEN
    reference = reference.format(
        instrument=instrument.lower(), mode=mode, specifier=specifier
    )

    reference = readsav(reference)

    # solution_2d = [version_number, number of columns, number of orders, base order, None, None, None,
    # number of cross terms, degree of column polynomial, degree of order polynomial, *fit_coeff]
    # solution_2d = reference["solution_2d"]
    # coeff = solution_2d[10:]
    # solution_2d = {
    #     "version": solution_2d[0],
    #     "ncol": int(solution_2d[1]),
    #     "nord": int(solution_2d[2]),
    #     "obase": int(solution_2d[3]),
    #     "oincr": int(reference["oincr"]),
    #     "ncross": int(solution_2d[7]),
    #     "coldeg": int(solution_2d[8]),
    #     "orddeg": int(solution_2d[9]),
    # }
    # base order, redundant with solution_2d entry
    # obase = reference["obase"]
    # increase between orders (usually 1)
    # oincr = reference["oincr"]

    # cs_lines = "Wavelength_center", "Wavelength_left", "Position_Center", "Position_Middle", "first x coordinate", "last x coordinate", "approximate ??", "width", "flag", "height", "order"
    # cs_lines = 'WLC', 'WLL', 'POSC', 'POSM', 'XFIRST', 'XLAST', 'APPROX', 'WIDTH', 'FLAG', 'HEIGHT', 'ORDER'
    cs_lines = reference["cs_lines"]

    # list of orders, bad orders marked with 1, normal orders 0
    bad_order = reference["bad_order"]

    solution = wavecal(thar, cs_lines, plot=True)

    print(solution)
