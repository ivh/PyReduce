import os

import numpy as np
from astropy.io import fits

# from modeinfo_uves import modeinfo_uves as modeinfo
from clipnflip import clipnflip
from modeinfo import modeinfo


def load_fits(fname, instrument, extension, **kwargs):
    hdu = fits.open(fname)
    header = hdu[extension].header
    header.extend(hdu[0].header, strip=False)
    instrument, mode = instrument.split("_")
    header = modeinfo(header, instrument, mode)

    if kwargs.get("header_only", False):
        return header

    data = clipnflip(hdu[extension].data, header)
    data = np.ma.masked_array(data, mask=kwargs.get("mask"))

    return data, header


def swap_extension(fname, ext, path=None):
    if path is None:
        path = os.path.dirname(fname)
    nameout = os.path.basename(fname)
    if nameout[-3:] == ".gz":
        nameout = nameout[:-3]
    nameout = nameout.rsplit(".", 1)[0]
    nameout = os.path.join(path, nameout + ext)
    return nameout


def find_first_index(arr, value):
    """ find the first element equal to value in the array arr """
    try:
        return next(i for i, v in enumerate(arr) if v == value)
    except StopIteration:
        raise Exception("Value %s not found" % value)
