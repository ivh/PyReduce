import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from scipy.ndimage import morphology

class SelectorWindow:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.mask = np.full(self.shape, False)
        self.first_guess()
        self.fig, self.ax = plt.subplots()
        self.plot()
        self.connect()

    def first_guess(self):
        # self.mask[self.data > np.percentile(self.data, 99.)] = True
        # self.mask[self.data < np.percentile(self.data, 1.)] = True
        pass

    def plot(self):
        if len(self.ax.images) != 0:
            ylim = self.ax.get_ylim()
            xlim = self.ax.get_xlim()
        else:
            ylim = 0, self.data.shape[0]
            xlim = 0, self.data.shape[1]
        self.ax.clear()
        tmp = np.ma.masked_array(self.data, mask=self.mask)
        low, hig = np.nanpercentile(self.data, (5, 95))
        self.ax.imshow(tmp, origin="lower", vmin=low, vmax=hig)
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

def load(fname, extension=0):
    data = fits.getdata(fname, ext=extension)
    return data

if __name__ == "__main__":
    # Load data
    # TODO have file selector in toolbar
    instrument = "NIRSPEC"
    mode = "NIRSPEC"
    # base_dir = os.path.expanduser(f"/DATA/Keck/{instrument}/GJ1214_b/")
    # fname = os.path.join(base_dir, "raw", "cal", "NS.20100805.07223.fits.gz")
    
    fname = ["/DATA/JWST/MIRI/MIRIsim/raw/det_image_seq1_MIRIMAGE_P750Lexp1.fits"]
    fname_out = f"mask_{instrument.lower()}_{mode.lower()}.fits.gz"
    extension = 0

    data = [load(f) for f in fname]
    data = np.sum(data, axis=0)
    while data.ndim > 2:
        data = np.sum(data, axis=0)

    header = fits.getheader(fname[0])

    # Create mask
    sw = SelectorWindow(data)
    plt.show()
    mask = np.where(sw.mask, 0, 1).astype(dtype="uint8")

    # Save mask
    fits.writeto(fname_out, header=header, data=mask, overwrite=True, output_verify="fix+warn")
