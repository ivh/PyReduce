import numpy as np
from astropy.io import fits

from modeinfo import modeinfo
#from modeinfo_uves import modeinfo_uves as modeinfo
from clipnflip import clipnflip


def load_fits(fname, instrument, extension, **kwargs):
    hdu = fits.open(fname)
    header = hdu[extension].header
    header.extend(hdu[0].header, strip=False)
    instrument, mode = instrument.split('_')
    header = modeinfo(header, instrument, mode)

    if kwargs.get("header_only", False):
        return header

    data = clipnflip(hdu[extension].data, header)
    data = np.ma.masked_array(data, mask=kwargs.get("mask"))

    return data, header
