"""
Abstract parent module for all other instruments
Contains some general functionality, which may be overridden by the children of course
"""
import numpy as np
import datetime
from astropy.io import fits


def find_first_index(arr, value):
    """ find the first element equal to value in the array arr """
    try:
        return next(i for i, v in enumerate(arr) if v == value)
    except StopIteration:
        raise Exception("Value %s not found" % value)


def observation_date_to_night(observation_date):
    """Convert an observation timestamp into the date of the observation night
    Nights start at 12am and end at 12 am the next day

    Parameters
    ----------
    observation_date : datetime
        timestamp of the observation

    Returns
    -------
    night : datetime.date
        night of the observation
    """
    oneday = datetime.timedelta(days=1)

    if observation_date.hour < 6:
        observation_date -= oneday
    return observation_date.date()


class getter:
    """Get data from a header/dict, based on the given mode, and applies replacements"""

    def __init__(self, header, info, mode):
        self.header = header
        self.info = info
        self.index = find_first_index(info["modes"], mode)

        # Pick values for the given mode
        for k, v in self.info.items():
            if isinstance(v, list):
                self.info[k] = v[self.index]

    def __call__(self, key, alt=None):
        return self.get(key, alt)

    def get(self, key, alt=None):
        """Get data

        Parameters
        ----------
        key : str
            key of the data in the header
        alt : obj, optional
            alternative value, if key does not exist (default: None)

        Returns
        -------
        value : obj
            value found in header (or alternatively alt)
        """

        value = self.info[key]
        if isinstance(value, list):
            value = value[self.index]
        if isinstance(value, str):
            value = value.format(**self.info)
            value = self.header.get(value, alt)
        return value


class instrument:
    """
    Abstract parent class for all instruments
    Handles the handling of instrument specific information
    """

    def load_info(self):
        """
        Load static instrument information
        Either as fits header keywords or static values

        Returns
        ------
        info : dict(str:object)
            dictionary of REDUCE names for properties to Header keywords/static values

        Raises
        ------
        NotImplementedError
            This function needs to exist for every instrument
        """

        raise NotImplementedError(
            "Instrument info must be defined for each instrument seperately"
        )

    def add_header_info(self, header, mode, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header

        Parameters
        ----------
        header : fits.header, dict
            header to read/write info from/to
        mode : str
            instrument mode

        Returns
        -------
        header : fits.header, dict
            header with added information
        """

        info = self.load_info()
        get = getter(header, info, mode)

        header["e_orient"] = get("orientation")

        naxis_x = get("naxis_x")
        naxis_y = get("naxis_y")

        prescan_x = get("prescan_x")
        overscan_x = get("overscan_x")
        prescan_y = get("prescan_y")
        overscan_y = get("overscan_y")

        header["e_xlo"] = prescan_x
        header["e_xhi"] = naxis_x - overscan_x

        header["e_ylo"] = prescan_y
        header["e_yhi"] = naxis_y - overscan_y

        header["e_gain"] = get("gain")
        header["e_readn"] = get("readnoise")
        header["e_exptim"] = get("exposure_time")

        header["e_sky"] = get("sky", 0)
        header["e_drk"] = get("dark", 0)
        header["e_backg"] = header["e_gain"] * (header["e_drk"] + header["e_sky"])

        header["e_imtype"] = get("image_type")
        header["e_ctg"] = get("category")

        header["e_ra"] = get("ra")
        header["e_dec"] = get("dec")
        header["e_jd"] = get("jd")

        header["e_obslon"] = get("longitude")
        header["e_obslat"] = get("latitude")
        header["e_obsalt"] = get("altitude")

        return header

    def sort_files(self, files, target, night, mode, **kwargs):
        """
        Sort a set of fits files into different categories
        types are: bias, flat, wavecal, orderdef, spec

        Parameters
        ----------
        files : list(str)
            files to sort
        target : str
            name of the observed target (as present in the header files of the observation)
        mode : str
            mode of the instrument to search for
        Returns
        -------
        biaslist, flatlist, wavelist, orderlist, speclist
            lists of files, one per type
        """
        info = self.load_info()

        ob = np.zeros(len(files), dtype="U20")
        ty = np.zeros(len(files), dtype="U20")

        for i, f in enumerate(files):
            h = fits.open(f)[0].header
            ob[i] = h[info["target"]]
            ty[i] = h[info["observation_type"]]

        biaslist = files[ty == info["id_bias"]]
        flatlist = files[ty == info["id_flat"]]
        wavelist = files[ob == info["id_wave"]]
        orderlist = files[ob == info["id_orders"]]
        speclist = files[ob == target]

        return biaslist, flatlist, wavelist, orderlist, speclist

    def get_wavecal_filename(self, header, mode, **kwargs):
        """Get the filename of the pre-existing wavelength solution for the current setting

        Parameters
        ----------
        header : fits.header, dict
            header of the wavelength calibration file
        mode : str
            instrument mode

        Returns
        -------
        filename : str
            name of the wavelength solution file
        """

        info = self.load_info()
        specifier = header.get(info.get("wavecal_specifier", ""), "")

        fname = "./wavecal/{instrument}_{mode}_{specifier}.sav".format(
            instrument=instrument.lower(), mode=mode, specifier=specifier
        )
        return fname
