import numpy as np
from astropy.io import fits

from modeinfo_uves import modeinfo_uves as modeinfo
from clipnflip import clipnflip


def load_fits(fname, instrument, extension, **kwargs):
    hdu = fits.open(fname)
    header = hdu[extension].header
    header.extend(hdu[0].header, strip=False)
    header = modeinfo(header, instrument, **kwargs)

    if kwargs.get("header_only", False):
        return header

    data = clipnflip(hdu[extension].data, header)
    data = np.ma.masked_array(data, mask=kwargs.get("mask"))

    return data, header
