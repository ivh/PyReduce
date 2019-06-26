import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

class SelectorWindow:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.mask = np.full(self.shape, False)
        self.fig, self.ax = plt.subplots()
        self.plot()
        self.connect()

    def plot(self):
        if len(self.ax.images) != 0:
            ylim = self.ax.get_ylim()
            xlim = self.ax.get_xlim()
        else:
            ylim = 0, self.data.shape[0]
            xlim = 0, self.data.shape[1]
        self.ax.clear()
        tmp = np.ma.masked_array(self.data, mask=self.mask)
        self.ax.imshow(tmp, origin="lower")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.fig.canvas.draw()
        # self.fig.canvas.flush()

    def connect(self):
        """ connect the click event with the appropiate function """
        self.cidclick = self.ax.figure.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

    def on_click(self, event):
        if event.ydata is None:
            return
        if not event.dblclick:
            return
        x = int(np.round(event.xdata))
        y = int(np.round(event.ydata))

        self.mask[y, x] = ~self.mask[y, x]
        self.plot()

if __name__ == "__main__":
    # Load data
    # TODO have file selector in toolbar
    instrument = "JWST_NIRISS"
    mode = "GR700XD"
    base_dir = os.path.expanduser("~/Documents/Visual Studio Code/PyReduce/datasets")
    fname = os.path.join(base_dir, instrument, "raw", "niriss_gj436_0.fits.gz")
    fname_out = f"mask_{instrument.lower()}_{mode.lower()}.fits"
    extension = 0

    hdu = fits.open(fname)
    header = hdu[0].header
    data = hdu[extension].data

    # Create mask
    sw = SelectorWindow(data)
    plt.show()
    mask = np.where(sw.mask, 0, 1, dtype="uint8")

    # Save mask
    fits.writeto(fname_out, header=header, data=mask, overwrite=True)