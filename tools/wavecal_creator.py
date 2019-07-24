from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time


# in: wavelength observation, linelist
# out: linelist
class OverallPlot:
    def __init__(self, data, linelist, ax=None, lines=None):
        self.data = data
        self.linelist = linelist
        self.nord, self.ncol = data.shape

        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = ax.figure

        if lines is None:
            self.lines = np.array([[0, 6, 23760.12, 1, 3, -5, 5], [self.ncol, 6, 24106.88, 1, 3, self.ncol-5, self.ncol + 5], #32
                                   [0, 5, 23045.63, 1, 3, -5, 5], [self.ncol, 5, 23380.86, 1, 3, self.ncol-5, self.ncol + 5], #33
                                   [0, 4, 22373.16, 1, 3, -5, 5], [self.ncol, 4, 22697.54, 1, 3, self.ncol-5, self.ncol + 5], #34
                                   [0, 3, 21739.12, 1, 3, -5, 5], [self.ncol, 3, 22053.27, 1, 3, self.ncol-5, self.ncol + 5], #35
                                   [0, 2, 21140.30, 1, 3, -5, 5], [self.ncol, 2, 21444.79, 1, 3, self.ncol-5, self.ncol + 5], #36
                                   [0, 1, 20573.86, 1, 3, -5, 5], [self.ncol, 1, 20869.20, 1, 3, self.ncol-5, self.ncol + 5], #37
                                   [0, 0, 20037.22, 1, 3, -5, 5], [self.ncol, 0, 20323.91, 1, 3, self.ncol-5, self.ncol + 5], #38
                                ])

            # self.lines[:, :, 1] -= (self.lines[..., 1] - self.lines[..., 0]) / 3
        else:
            self.lines = lines

        self.ax.__lines__ = self.lines

        self.plot()
        self.connect()
        self.op = None

    def plot(self):
        self.ax.clear()
        self.ax.imshow(self.data, origin="lower", aspect="auto")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def connect(self):
        """ connect the click event with the appropiate function """
        self.cidclick = self.ax.figure.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidclick)

    def on_click(self, event):
        if event.ydata is None:
            return
        y = int(np.round(event.ydata))


        # Open new window with just the selected order
        self.disconnect()
        self.op = OrderPlot(data, y, self.linelist, self.lines, ax=self.ax)


class OrderPlot:
    def __init__(self, data, order, linelist, lines, ax=None):
        self.overall_data = data
        self.data = data[order] / np.max(data[order])
        self.ncol = len(self.data)
        self.order = order

        height = 2 * np.ma.median(self.data)
        self.peaks, _ = signal.find_peaks(self.data, height=height, width=3)

        self.linelist = linelist
        self.lines = lines
        self.cutlines = False

        self.poly = np.polyfit(self.pos, self.wave, self.degree)

        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = ax.figure
            self.ax = ax

        self.connect()

        self.plot()
        self.op = None
        self.firstclick = True


    @property
    def nlines(self):
        return len(self.lines[self.lines[:, 1] == self.order])

    @property
    def pos(self):
        return self.lines[self.lines[:, 1] == self.order, 0]

    @property
    def wave(self):
        return self.lines[self.lines[:, 1] == self.order, 2]

    @property
    def degree(self):
        return min(5, max(self.nlines - 1, 0)) 

    def plot(self):
        self.ax.clear()

        wave = np.polyval(self.poly, np.arange(self.ncol))
        reference = np.zeros(self.ncol)

        for i, line in enumerate(self.linelist):
            wl = line["wll"]
            if wl > wave[0] and wl < wave[-1]:
                middle = np.argmin(np.abs(wl - wave))
                self.linelist["posc"][i] = middle
                self.linelist["order"][i] = self.order
                first = int(np.clip(middle - line["width"], 0, self.ncol))
                last = int(np.clip(middle + line["width"], 0, self.ncol))
                reference[first:last] += (
                    line["height"] * signal.gaussian(last - first, line["width"])
                )
        reference /= np.max(reference)

        plt.title(f"Order {self.order}")
        plt.xlim(0, len(self.data))
        plt.plot(self.data, label="data")
        plt.plot(self.peaks, self.data[self.peaks], "d", label="peaks")
        plt.plot(reference, label="reference")
        plt.legend()

        self.fig.canvas.draw()

    def connect(self):
        """ connect the click event with the appropiate function """
        self.cidclick = self.ax.figure.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidclick)

    def on_click(self, event):
        if event.xdata is None:
            return
        # Right Click
        if event.button == 3:
            self.disconnect()
            self.op = OverallPlot(self.overall_data, self.linelist, ax=self.ax, lines=self.lines)
        # Left click
        if event.button == 1:
            if self.firstclick:
                x = int(np.round(event.xdata))
                # find closest peak
                idx = np.argmin(np.abs(x - self.peaks))
                line, = plt.plot([self.peaks[idx],], [self.data[self.peaks[idx]],], "rx")
                self.ax.figure.canvas.draw()
                self.fig.canvas.flush_events()
                self.firstclick = False

                self.position = self.peaks[idx]
                self.height = self.data[self.peaks[idx]]
            else:
                x = int(np.round(event.xdata))
                order_lines = self.linelist[self.linelist["order"] == self.order]
                idx = np.argmin(np.abs(x - order_lines["posc"]))
                # find closest peak
                line, = plt.plot(order_lines["posc"][idx], 1, "rx")
                self.ax.figure.canvas.draw()
                self.fig.canvas.flush_events()
                wave = order_lines["wll"][idx]
                print(wave)

                # connect this peak in the observation with the (nearest) line in the linelist
                self.lines = np.concatenate((self.lines, [[self.position, self.order, wave, self.height, 3, self.position - 5, self.position + 5]]))

                self.poly = np.polyfit(self.pos, self.wave, deg=self.degree)
                self.plot()

                self.firstclick = True

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

    lines = op.ax.__lines__
    # Remove initial guesses
    lines = lines[2*7:]
    posc, order, wll, height, width, xfirst, xlast = lines.T
    flag = np.full(len(posc), True)
    posm = np.copy(posc)
    lines = np.rec.fromarrays((posc, order, wll, height, width, xfirst, xlast, posm, flag), names=["posc", "order", "wll", "height", "width", "xfirst", "xlast", "posm", "flag"])
    np.savez("lines.npz", cs_lines=lines)

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
