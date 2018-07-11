import numpy as np
from numpy.polynomial.polynomial import polyval2d

from scipy.io import readsav
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt, find_peaks
import astropy.io.fits as fits

from util import save_fits


import matplotlib.pyplot as plt


class AlignmentPlot:
    def __init__(self, ax, thar, cs_lines):
        self.im = ax
        self.first = True
        self.nord, self.ncol = thar.shape
        self.RED, self.GREEN, self.BLUE = 0, 1, 2

        # TODO
        thar[thar < 2] = 0
        thar[thar > 15000] = 15000

        self.thar = thar / np.max(thar)

        norm = (np.e - 1) / np.max(cs_lines.height)
        cs_lines.height = np.log(1 + cs_lines.height * norm)
        self.cs_lines = cs_lines

        self.order_first = 0
        self.spec_first = ""
        self.x_first = 0
        self.offset = [0, 0]

        self.make_ref_image()

    def make_ref_image(self):
        ref_image = np.zeros((self.nord * 2, self.ncol, 3))
        for iord in range(self.nord):
            idx = self.thar[iord] > 0.1
            ref_image[iord * 2, idx, self.RED] = self.thar[iord, idx]
            if 0 <= iord + self.offset[0] < self.nord:
                for line in self.cs_lines[self.cs_lines.order == iord]:
                    ref_image[
                        (iord + self.offset[0]) * 2 + 1,
                        line.xfirst + self.offset[1] : line.xlast + self.offset[1],
                        self.GREEN,
                    ] = line.height

        self.im.imshow(
            ref_image,
            aspect="auto",
            origin="lower",
            extent=(0, self.ncol, 0, self.nord),
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


def align(thar, cs_lines, manual=True):
    # Align using window like in IDL REDUCE
    # TODO auto align (using cross corelation?)
    if manual:
        _, ax = plt.subplots()
        ap = AlignmentPlot(ax, thar, cs_lines)
        ap.connect()
        plt.show()
        offset = ap.offset
    else:
        raise NotImplementedError("Use manual alignment for now")
    return offset


def build_2d_solution(cs_lines, oincr=1, obase=0):
    # Only use flagged data
    #TODO whoch fields to use for fit?
    mask = ~cs_lines.flag.astype(bool)
    m_wave = cs_lines.wll[mask]
    m_pix = cs_lines.posm[mask]
    m_ord = cs_lines.order[mask] * oincr + obase

    # 2d polynomial fit with: x = m_pix, y = m_ord, z = m_wave
    # TODO: Choose polynomial degree
    degree_x, degree_y = 3, 3
    degree_x, degree_y = degree_x+1, degree_y+1 #Due to how np polyval2d workss

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


def make_wave(thar, wave_solution):
    nord, ncol = thar.shape
    x = np.arange(ncol)
    y = np.arange(nord)
    x, y = np.meshgrid(x, y)
    wave_img = polyval2d(x, y, wave_solution)
    return wave_img


def auto_id(thar, wave_img, cs_lines, threshold=1):
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
        peak_idx = find_peaks_cwt(vec, np.arange(2, 5), min_snr=5, min_length=3)
        # peak_idx, _ = find_peaks(thar[iord], height = 1100, width=3)
        pos_wave = wave_img[iord, thar[iord] > 0][peak_idx]
        # plt.plot(x, vec)
        # plt.plot(x[peak_idx], vec[peak_idx], 'r+')
        # plt.show()

        for i, line in enumerate(cs_lines):
            if line.order == iord:
                diff = np.abs(line.wll - pos_wave)
                if np.min(diff) < threshold:
                    cs_lines.flag[i] = 0  # 0 == True, 1 == False

    return cs_lines

def reject_lines(thar, wave_solution, cs_lines):
    # Calculate residuals
    x = cs_lines.posm
    y = cs_lines.order
    solution = polyval2d(x, y, wave_solution)
    residual = solution - cs_lines.wll

    plt.plot(cs_lines.order, residual, '+')
    plt.show()

    return cs_lines

def wavecal(thar, head, solution_2d, cs_lines):
    obase = solution_2d["obase"]
    oincr = solution_2d["oincr"]
    nord, ncol = thar.shape

    # Step 1: align thar and reference
    # TODO: offset = align(thar, cs_lines)
    offset = (1, 106)
    # Step 1.5: Apply offset
    # be careful not to apply offset twice
    cs_lines.xfirst += offset[1]
    cs_lines.xlast += offset[1]
    cs_lines.order += offset[0]

    # Step 2: build 2d solution
    wave_solution = build_2d_solution(cs_lines)
    # Step 2.5: calculate wavelength solution for each point
    wave_img = make_wave(thar, wave_solution)

    # Step 3: reject lines, which do not correlate to observed lines
    cs_lines = reject_lines(thar, wave_solution, cs_lines)

    # Step 4: build 2d solution
    wave_solution = build_2d_solution(cs_lines)
    # Step 4.5: calculate wavelength solution for each point
    wave_img = make_wave(thar, wave_solution)

    # Step 5: auto ID lines
    cs_lines = auto_id(thar, wave_img, cs_lines)

    # Step 6: build final 2d solution
    wave_solution = build_2d_solution(cs_lines)
    # Step 6.5: calculate wavelength solution for each point
    wave_img = make_wave(thar, wave_solution)

    #repeat steps 3-6 as necessay

    # TODO: DEBUG
    plt.imshow(wave_img, aspect="auto")
    plt.show()

    return wave_solution  # wavelength solution


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
    solution_2d = reference["solution_2d"]
    coeff = solution_2d[10:]
    solution_2d = {
        "version": solution_2d[0],
        "ncol": int(solution_2d[1]),
        "nord": int(solution_2d[2]),
        "obase": int(solution_2d[3]),
        "oincr": int(reference["oincr"]),
        "ncross": int(solution_2d[7]),
        "coldeg": int(solution_2d[8]),
        "orddeg": int(solution_2d[9]),
    }
    # cs_lines = "Wavelength_center", "Wavelength_left", "Position_Center", "Position_?", "first x coordinate", "last x coordinate", "approximate ??", "width", "flag", "height", "order"
    # cs_lines = 'WLC', 'WLL', 'POSC', 'POSM', 'XFIRST', 'XLAST', 'APPROX', 'WIDTH', 'FLAG', 'HEIGHT', 'ORDER'
    cs_lines = reference["cs_lines"]
    # base order, redundant with solution_2d entry
    obase = reference["obase"]
    # increase between orders (usually 1)
    oincr = reference["oincr"]
    # list of orders, bad orders marked with 1, normal orders 0
    bad_order = reference["bad_order"]

    solution = wavecal(thar, head, solution_2d, cs_lines)

    print(solution)
