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
        self.ax.imshow(self.data)
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
        wave0 = input("Start wavelength")
        wave1 = input("End wavelength")

        # Open new window with just the selected order
        OrderPlot(data[y], self.linelist, (wave0, wave1))


class OrderPlot:
    def __init__(self, data, linelist, wave_region):
        self.data = data
        self.ncol = len(data)

        height = np.ma.median(data)
        self.peaks = signal.find_peaks(data, height=height, width=3)

        self.linelist = linelist
        self.lines = np.array([[0, wave_region[0]], [self.ncol, wave_region[1]]])
        self.cutlines = False

        self.poly = np.polyfit(self.pos, self.wave, 1)

        self.fig, self.ax = plt.subplots()
        self.plot()

        self.connect()

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
            wl = line["wavelength"]
            if wl > wave[0] and wl < wave[-1]:
                middle = np.digitize(wl, wave)
                first = np.clip(middle - line["width"], 0, self.ncol)
                last = np.clip(middle + line["width"], 0, self.ncol)
                reference[first:last] += (
                    10 * line["height"] * signal.gaussian(last - first, line["width"])
                )

        plt.plot(self.data)
        plt.plot(self.peaks, self.data[self.peaks], "d")
        plt.plot(reference)

    def connect(self):
        """ connect the click event with the appropiate function """
        self.cidclick = self.ax.figure.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

    def on_click(self, event):
        if event.xdata is None:
            return
        x = int(np.round(event.xdata))

        # find closest peak
        idx = np.argmin(np.abs(x - self.peaks))

        # find closest line in linelist
        wave = np.polyval(self.poly, np.arange(self.ncol))
        pos = np.digitize(self.linelist["wavelength"], wave)
        idx2 = np.argmin(np.abs(pos - self.peaks[idx]))
        wl = self.linelist[idx2]["wavelength"]

        # connect this peak in the observation with the (nearest) line in the linelist
        self.lines = np.concatenate((self.lines, [[self.peaks[idx], wl]]))

        # Remove the initial wavelength guess
        if not self.cutlines and self.nlines >= 4:
            self.lines = self.lines[2:]
            self.cutlines = True

        degree = min(5, self.nlines - 1)
        self.poly = np.polyfit(self.pos, self.wave, deg=degree)
        self.plot()

if __name__ == "__main__":
    # filename of extracted wavelength spectrum
    fname = ""
    # filename of the input linelist
    flinelist = ""

    hdu = fits.open(fname)
    data = hdu[0].data

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
