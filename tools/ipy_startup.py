"""IPython startup imports for PyReduce development."""

import numpy as np
from astropy.io import fits

fo = fits.open

from matplotlib import pyplot as plt

plt.ion()


class NpzEditor(dict):
    def __init__(self, filename):
        self.filename = filename
        super().__init__(np.load(filename))

    def save(self):
        np.savez(self.filename, **self)
