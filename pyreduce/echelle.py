# -*- coding: utf-8 -*-
"""
Contains functions to read and modify echelle structures, just as in reduce

Mostly for compatibility reasons
"""

import logging

import astropy.io.fits as fits
import numpy as np
import scipy.constants

logger = logging.getLogger(__name__)


class Echelle:
    def __init__(self, head={}, filename="", data={}):
        self.filename = filename
        self.header = head
        self._data = data

    @property
    def nord(self):
        if "spec" in self._data.keys():
            return self._data["spec"].shape[0]
        return None

    @property
    def ncol(self):
        if "spec" in self._data.keys():
            return self._data["spec"].shape[1]
        return None

    @property
    def spec(self):
        if "spec" in self._data.keys():
            return self._data["spec"]
        else:
            return None

    @spec.setter
    def spec(self, value):
        self._data["spec"] = value

    @property
    def sig(self):
        if "sig" in self._data.keys():
            return self._data["sig"]
        else:
            return None

    @sig.setter
    def sig(self, value):
        self._data["sig"] = value

    @property
    def wave(self):
        if "wave" in self._data.keys():
            return self._data["wave"]
        else:
            return None

    @wave.setter
    def wave(self, value):
        self._data["wave"] = value

    @property
    def cont(self):
        if "cont" in self._data.keys():
            return self._data["cont"]
        else:
            return None

    @cont.setter
    def cont(self, value):
        self._data["cont"] = value

    @property
    def columns(self):
        if "columns" in self._data.keys():
            return self._data["columns"]
        else:
            return None

    @columns.setter
    def columns(self, value):
        self._data["columns"] = value

    @property
    def mask(self):
        if "mask" in self._data.keys():
            return self._data["mask"]
        else:
            return None

    @mask.setter
    def mask(self, value):
        self._data["mask"] = value

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __delitem__(self, index):
        del self._data[index]

    def __contains__(self, index):
        return index in self._data.keys()

    @staticmethod
    def read(
        fname,
        extension=1,
        raw=False,
        continuum_normalization=True,
        barycentric_correction=True,
        radial_velociy_correction=True,
    ):
        """
        Read data from an echelle file
        Expand wavelength and continuum polynomials
        Apply barycentric/radial velocity correction
        Apply continuum normalization

        Will load any fields in the binary table, however special attention is given only to specific names:
        "SPEC"    : Spectrum
        "SIG"     : Sigma, i.e. (absolute) uncertainty
        "CONT"    : Continuum
        "WAVE"    : Wavelength solution
        "COLUMNS" : Column range

        Parameters
        ----------
        fname : str
            filename to load
        extension : int, optional
            fits extension of the data within the file (default: 1)
        raw : bool, optional
            if true apply no corrections to the data (default: False)
        continuum_normalization : bool, optional
            apply continuum normalization (default: True)
        barycentric_correction : bool, optional
            apply barycentric correction (default: True)
        radial_velociy_correction : bool, optional
            apply radial velocity correction (default: True)

        Returns
        -------
        ech : obj
            Echelle structure, with data contained in attributes
        """

        hdu = fits.open(fname)
        header = hdu[0].header
        data = hdu[extension].data

        _data = {column.lower(): data[column][0] for column in data.dtype.names}
        ech = Echelle(filename=fname, head=header, data=_data)
        nord, ncol = ech.nord, ech.ncol

        if not raw:
            if "spec" in ech:
                base_order = header.get("obase", 1)
                ech["orders"] = np.arange(base_order, base_order + nord)

            # Wavelength
            if "wave" in ech:
                ech["wave"] = expand_polynomial(ncol, ech["wave"])

                # Correct for radial velocity and barycentric correction
                # + : away from observer
                # - : towards observer
                velocity_correction = 0
                if barycentric_correction:
                    velocity_correction -= header.get("barycorr", 0)
                    header["barycorr"] = 0
                if radial_velociy_correction:
                    velocity_correction += header.get("radvel", 0)
                    header["radvel"] = 0

                speed_of_light = scipy.constants.speed_of_light * 1e-3
                ech["wave"] *= 1 + velocity_correction / speed_of_light

            # Continuum
            if "cont" in ech:
                ech["cont"] = expand_polynomial(ncol, ech["cont"])

            # Create Mask, based on column range
            if "columns" in ech:
                ech["mask"] = np.full((nord, ncol), True)
                for iord in range(nord):
                    ech["mask"][
                        iord, ech["columns"][iord, 0] : ech["columns"][iord, 1]
                    ] = False

                if "spec" in ech:
                    ech["spec"] = np.ma.masked_array(ech["spec"], mask=ech["mask"])
                if "sig" in ech:
                    ech["sig"] = np.ma.masked_array(ech["sig"], mask=ech["mask"])
                if "cont" in ech:
                    ech["cont"] = np.ma.masked_array(ech["cont"], mask=ech["mask"])
                if "wave" in ech:
                    ech["wave"] = np.ma.masked_array(ech["wave"], mask=ech["mask"])

            # Apply continuum normalization
            if continuum_normalization and "cont" in ech:
                if "spec" in ech:
                    ech["spec"] /= ech["cont"]
                if "sig" in ech:
                    ech["sig"] /= ech["cont"]

        return ech

    def save(self, fname):
        """
        Save data in an Echelle fits, i.e. a fits file with a Binary Table in Extension 1

        Parameters
        ----------
        fname : str
            filename
        """
        save(fname, self.header, **self._data)


def calc_2dpolynomial(solution2d):
    """Expand a 2d polynomial, where the data is given in a REDUCE make_wave format
    Note that the coefficients are for order/100 and column/1000 respectively, where the order is counted from order base up

    Parameters
    ----------
    solution2d : array
        data in a REDUCE make_wave format:
        0: version
        1: number of columns
        2: number of orders
        3: order base, i.e. 0th order number (should not be 0)
        4-6: empty
        7: number of cross coefficients
        8: number of column only coefficients
        9: number of order only coefficients
        10: coefficient - constant
        11-x: column coefficients
        x-y : order coefficients
        z-  : cross coefficients (xy, xy2, x2y, x2y2, xy3, x3y), with x = orders, y = columns

    Returns
    -------
    poly : array[nord, ncol]
        expanded polynomial
    """

    # make wave style 2d fit
    ncol = int(solution2d[1])
    nord = int(solution2d[2])
    order_base = int(solution2d[3])
    deg_cross, deg_column, deg_order = (
        int(solution2d[7]),
        int(solution2d[8]),
        int(solution2d[9]),
    )
    coeff_in = solution2d[10:]

    coeff = np.zeros((deg_order + 1, deg_column + 1))
    coeff[0, 0] = coeff_in[0]
    coeff[0, 1:] = coeff_in[1 : 1 + deg_column]
    coeff[1:, 0] = coeff_in[1 + deg_column : 1 + deg_column + deg_order]
    if deg_cross in [4, 6]:
        coeff[1, 1] = coeff_in[deg_column + deg_order + 1]
        coeff[1, 2] = coeff_in[deg_column + deg_order + 2]
        coeff[2, 1] = coeff_in[deg_column + deg_order + 3]
        coeff[2, 2] = coeff_in[deg_column + deg_order + 4]
    if deg_cross == 6:
        coeff[1, 3] = coeff_in[deg_column + deg_order + 5]
        coeff[3, 1] = coeff_in[deg_column + deg_order + 6]

    x = np.arange(order_base, order_base + nord, dtype=float)
    y = np.arange(ncol, dtype=float)

    poly = np.polynomial.polynomial.polygrid2d(x / 100, y / 1000, coeff) / x[:, None]

    return poly


def calc_1dpolynomials(ncol, poly):
    """Expand a set of 1d polynomials (one per order) seperately

    Parameters
    ----------
    ncol : int
        number of columns
    poly : array[nord, degree]
        polynomial coefficients

    Returns
    -------
    poly : array[nord, ncol]
        expanded polynomials
    """

    nord = poly.shape[0]
    x = np.arange(ncol)
    result = np.zeros((nord, ncol))
    for i, coef in enumerate(poly):
        result[i] = np.polyval(coef, x)
    return result


def expand_polynomial(ncol, poly):
    """Checks if and how to expand data poly, then expands the data if necessary

    Parameters
    ----------
    ncol : int
        number of columns in the image
    poly : array[nord, ...]
        polynomial coefficients to expand, or already expanded data

    Returns
    -------
    poly : array[nord, ncol]
        expanded data
    """

    if poly.ndim == 1:
        poly = calc_2dpolynomial(poly)
    elif poly.shape[1] < 20:
        poly = calc_1dpolynomials(ncol, poly)
    return poly


def read(fname, **kwargs):
    return Echelle.read(fname, **kwargs)


def save(fname, header, **kwargs):
    """Save data in an Echelle fits, i.e. a fits file with a Binary Table in Extension 1

    The data is passed in kwargs, with the name of the binary table column as the key
    Floating point data is saved as float32 (E), Integer data as int16 (I)

    Parameters
    ----------
    fname : str
        filename
    header : fits.header
        FITS header
    **kwargs : array[]
        data to be saved in the file
    """

    if not isinstance(header, fits.Header):
        header = fits.Header(cards=header)

    primary = fits.PrimaryHDU(header=header)

    columns = []
    for key, value in kwargs.items():
        if value is None:
            continue
        arr = value.ravel()[None, :]

        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
            dtype = "E"
        elif np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int16)
            dtype = "I"
        elif np.issubdtype(arr.dtype, np.bool_):
            arr = arr.astype(np.bool_)
            dtype = "B"
        else:
            raise TypeError(f"Could not understand dtype {arr.dtype}")

        form = "%i%s" % (value.size, dtype)
        dim = str(value.shape[::-1])
        columns += [fits.Column(name=key.upper(), array=arr, format=form, dim=dim)]

    table = fits.BinTableHDU.from_columns(columns)

    hdulist = fits.HDUList(hdus=[primary, table])
    hdulist.writeto(fname, overwrite=True, output_verify="silentfix+ignore")
