from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# in: wavelength observation, linelist
# out: linelist
class OverallPlot:
    def __init__(self, data, linelist):
        self.data = data
        self.linelist = linelist
        self.nord, self.ncol = data.shape

        self.fig, self.ax = plt.subplots()
        self.plot()
        self.connect()

    def plot(self):
        self.ax.imshow(self.data, origin="lower", aspect="auto")
        self.fig.canvas.draw()

    def connect(self):
        """ connect the click event with the appropiate function """
        self.cidclick = self.ax.figure.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

    def on_click(self, event):
        if event.ydata is None:
            return
        y = int(np.round(event.ydata))

        # Ask for wavelength region (if not defined)
        wave0 = 25273 #input("Start wavelength")
        wave1 = 25961 #input("End wavelength")

        # Open new window with just the selected order
        op = OrderPlot(data[y], self.linelist, (wave0, wave1), y)
        plt.show()


class OrderPlot:
    def __init__(self, data, linelist, wave_region, order):
        self.data = data / np.max(data)
        self.ncol = len(data)
        self.order = order

        height = 2 * np.ma.median(data)
        self.peaks, _ = signal.find_peaks(data, height=height, width=3)

        self.linelist = linelist
        self.lines = np.array([[0, wave_region[0]], [self.ncol, wave_region[1]]])
        self.cutlines = False

        self.poly = np.polyfit(self.pos, self.wave, 1)

        self.fig, self.ax = plt.subplots()
        self.connect()

        self.plot()


    @property
    def nlines(self):
        return len(self.lines)

    @property
    def pos(self):
        return self.lines[:, 0]

    @property
    def wave(self):
        return self.lines[:, 1]

    def plot(self):
        wave = np.polyval(self.poly, np.arange(self.ncol))
        reference = np.zeros(self.ncol)

        for line in self.linelist:
            wl = line["wll"]
            if wl > wave[0] and wl < wave[-1]:
                middle = np.digitize(wl, wave)
                first = int(np.clip(middle - line["width"], 0, self.ncol))
                last = int(np.clip(middle + line["width"], 0, self.ncol))
                reference[first:last] += (
                    line["height"] * signal.gaussian(last - first, line["width"])
                )
        reference /= np.max(reference)

        plt.title(f"Order {self.order}")
        plt.plot(self.data, label="data")
        plt.plot(self.peaks, self.data[self.peaks], "d", label="peaks")
        plt.plot(reference, label="reference")
        plt.legend()

    def connect(self):
        """ connect the click event with the appropiate function """
        self.cidclick = self.ax.figure.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

    def on_click(self, event):
        print("BLA")
        if event.xdata is None:
            return
        x = int(np.round(event.xdata))

        # find closest peak
        idx = np.argmin(np.abs(x - self.peaks))

        # find closest line in linelist
        wave = np.polyval(self.poly, np.arange(self.ncol))
        pos = np.digitize(self.linelist["wll"], wave)
        idx2 = np.argmin(np.abs(pos - self.peaks[idx]))
        wl = self.linelist[idx2]["wll"]

        # connect this peak in the observation with the (nearest) line in the linelist
        self.lines = np.concatenate((self.lines, [[self.peaks[idx], wl]]))

        # Remove the initial wavelength guess
        if not self.cutlines and self.nlines >= 4:
            self.lines = self.lines[2:]
            self.cutlines = True

        degree = min(5, self.nlines - 1)
        self.poly = np.polyfit(self.pos, self.wave, deg=degree)
        self.plot()

def air2vac(wl_air):
    """
    Convert wavelengths in air to vacuum wavelength
    Author: Nikolai Piskunov
    """
    wl_vac = np.copy(wl_air)
    ii = np.where(wl_air > 1999.352)

    sigma2 = (1e4 / wl_air[ii]) ** 2  # Compute wavenumbers squared
    fact = (
        1e0
        + 8.336624212083e-5
        + 2.408926869968e-2 / (1.301065924522e2 - sigma2)
        + 1.599740894897e-4 / (3.892568793293e1 - sigma2)
    )
    wl_vac[ii] = wl_air[ii] * fact  # Convert to vacuum wavelength

    return wl_vac

if __name__ == "__main__":
    # filename of extracted wavelength spectrum
    fname = "tools/thar.npz"
    # filename of the input linelist
    flinelist = "tools/neon.line"

    hdu = np.load(fname)
    data = hdu["thar"]


    wave, height = [], []
    with open(flinelist, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            line = line.split()
            wave.append(float(line[0]))
            height.append((float(line[3])))
    wave = np.array(wave)
    height = np.array(height)

    linelist = air2vac(wave)
    positions = np.zeros(len(linelist))
    order = np.zeros(len(linelist))
    width = np.full(len(linelist), 5.)

    linelist = np.rec.fromarrays((linelist, positions, order, width, height), names=("wll", "posc", "order", "width", "height"))

    # K-band 1996-2382 Angstrom

    op = OverallPlot(data, linelist)
    plt.show()

    # Plot wl spectrum, without linelist
    # Go to each order, and ask for a wavelength range
    # Plot linelist for that wavelength range
    # Find peaks in observation
    # Manually match peaks to linelist
    # Update plot to match new fit
    # AutoID remaining peaks
    # Discard unused lines in the linelist
    # Save results

    pass
