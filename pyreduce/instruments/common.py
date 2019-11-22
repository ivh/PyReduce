"""
Abstract parent module for all other instruments
Contains some general functionality, which may be overridden by the children of course
"""
import os.path
import datetime
import glob
import logging
import json

import numpy as np
from astropy.io import fits
from astropy.time import Time
from dateutil import parser

from ..clipnflip import clipnflip

logger = logging.getLogger(__name__)


def find_first_index(arr, value):
    """ find the first element equal to value in the array arr """
    try:
        return next(i for i, v in enumerate(arr) if v == value)
    except StopIteration:
        raise KeyError("Value %s not found" % value)


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
    if observation_date == "":
        return None

    observation_date = parser.parse(observation_date)
    oneday = datetime.timedelta(days=1)

    if observation_date.hour < 12:
        observation_date -= oneday
    return observation_date.date()


class getter:
    """Get data from a header/dict, based on the given mode, and applies replacements"""

    def __init__(self, header, info, mode):
        self.header = header
        self.info = info.copy()
        try:
            self.index = find_first_index(info["modes"], mode.upper())
        except KeyError:
            logger.warning("No instrument modes found in instrument info")
            self.index = 0

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

        value = self.info.get(key, key)
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

    def __init__(self):
        #:str: Name of the instrument (lowercase)
        self.name = self.__class__.__name__.lower()
        #:dict: Information about the instrument
        self.info = self.load_info()

    def get_extension(self, header, mode):
        mode = mode.upper()
        extension = self.info.get("extension", 0)
        
        if isinstance(extension, list):
            imode = find_first_index(self.info["modes"], mode)
            extension = extension[imode]

        return extension

    def load_info(self):
        """
        Load static instrument information
        Either as fits header keywords or static values

        Returns
        ------
        info : dict(str:object)
            dictionary of REDUCE names for properties to Header keywords/static values
        """
        # Tips & Tricks:
        # if several modes are supported, use a list for modes
        # if a value changes depending on the mode, use a list with the same order as "modes"
        # you can also use values from this dictionary as placeholders using {name}, just like str.format

        this = os.path.dirname(__file__)
        fname = f"{self.name}.json"
        fname = os.path.join(this, fname)
        with open(fname) as f:
            info = json.load(f)
        return info

    def load_fits(self,
        fname, mode, extension=None, mask=None, header_only=False, dtype=None
    ):
        """
        load fits file, REDUCE style

        primary and extension header are combined
        modeinfo is applied to header
        data is clipnflipped
        mask is applied

        Parameters
        ----------
        fname : str
            filename
        instrument : str
            name of the instrument
        mode : str
            instrument mode
        extension : int
            data extension of the FITS file to load
        mask : array, optional
            mask to add to the data
        header_only : bool, optional
            only load the header, not the data
        dtype : str, optional
            numpy datatype to convert the read data to

        Returns
        --------
        data : masked_array
            FITS data, clipped and flipped, and with mask
        header : fits.header
            FITS header (Primary and Extension + Modeinfo)

        ONLY the header is returned if header_only is True
        """

        info = self.info
        mode = mode.upper()

        hdu = fits.open(fname)
        h_prime = hdu[0].header
        if extension is None:
            extension = self.get_extension(h_prime, mode)

        header = hdu[extension].header
        header.extend(h_prime, strip=False)
        header = self.add_header_info(header, mode)
        header["e_input"] = (os.path.basename(fname), "Original input filename")

        if header_only:
            hdu.close()
            return header

        data = clipnflip(hdu[extension].data, header)

        if dtype is not None:
            data = data.astype(dtype)

        data = np.ma.masked_array(data, mask=mask)

        hdu.close()
        return data, header

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

        header["INSTRUME"] = get("instrument", self.__class__.__name__)
        header["TELESCOP"] = get("telescope", "")
        header["EXPTIME"] = get("exposure_time", 0)
        header["MJD-OBS"] = get("date", 0)

        if isinstance(header["MJD-OBS"], str):
            try:
                header["MJD-OBS"] = Time(header["MJD-OBS"]).mjd
            except:
                logger.warning("Unable to determine the MJD date of the observation")

        header["e_orient"] = get("orientation", 0)

        naxis_x = get("naxis_x", 0)
        naxis_y = get("naxis_y", 0)

        prescan_x = get("prescan_x", 0)
        overscan_x = get("overscan_x", 0)
        prescan_y = get("prescan_y", 0)
        overscan_y = get("overscan_y", 0)

        header["e_xlo"] = prescan_x
        header["e_xhi"] = naxis_x - overscan_x

        header["e_ylo"] = prescan_y
        header["e_yhi"] = naxis_y - overscan_y

        header["e_gain"] = get("gain", 1)
        header["e_readn"] = get("readnoise", 0)

        header["e_sky"] = get("sky", 0)
        header["e_drk"] = get("dark", 0)
        header["e_backg"] = header["e_gain"] * (header["e_drk"] + header["e_sky"])

        header["e_imtype"] = get("image_type")
        header["e_ctg"] = get("category")

        header["e_ra"] = get("ra", 0)
        header["e_dec"] = get("dec", 0)
        header["e_jd"] = get("jd", 0)

        header["e_obslon"] = get("longitude")
        header["e_obslat"] = get("latitude")
        header["e_obsalt"] = get("altitude")

        header["HIERARCH e_wavecal_element"] = get("wavecal_element", info.get("wavecal_element", None))
        return header

    def sort_files(self, input_dir, target, night, mode, **kwargs):
        """
        Sort a set of fits files into different categories
        types are: bias, flat, wavecal, orderdef, spec

        Parameters
        ----------
        input_dir : str
            input directory containing the files to sort
        target : str
            name of the target as in the fits headers
        night : str
            observation night, possibly with wildcards
        mode : str
            instrument mode
        Returns
        -------
        files_per_night : list[dict{str:dict{str:list[str]}}]
            a list of file sets, one entry per night, where each night consists of a dictionary with one entry per setting,
            each fileset has five lists of filenames: "bias", "flat", "order", "wave", "spec", organised in another dict
        nights_out : list[datetime]
            a list of observation times, same order as files_per_night
        """

        # TODO allow several names for the target?

        info = self.load_info()
        target = target.upper()
        instrument = info.get("__instrument__", "").upper()

        # Try matching with nights
        try:
            night = parser.parse(night).date()
            individual_nights = [night]
        except ValueError:
            # if the input night can't be parsed, use all nights
            # Usually the case if wildcards are involved
            individual_nights = "all"

        # find all fits files in the input dir(s)
        input_dir = input_dir.format(
            instrument=instrument.upper(), target=target, mode=mode, night=night
        )
        files = glob.glob(input_dir + "/*.fits")
        files += glob.glob(input_dir + "/*.fits.gz")
        files = np.array(files)

        # Load the mode identifier for the current mode from the header
        # This could be anything really, e.g. the size of the data axis
        i = [i for i, m in enumerate(info["modes"]) if m == mode][0]
        mode_id = info["modes_id"][i].upper()

        # Initialize arrays
        # observed object
        ob = np.zeros(len(files), dtype="U20")
        # observation type, bias, flat, spec, etc.
        ty = np.zeros(len(files), dtype="U20")
        # instrument mode, e.g. red, blue
        mo = np.zeros(len(files), dtype="U20")
        # special setting identifier, e.g. wavelength setting
        se = np.zeros(len(files), dtype="U20")
        # observed night, parsed into a datetime object
        ni = np.zeros(len(files), dtype=datetime.datetime)
        # instrument, used for observation
        it = np.zeros(len(files), dtype="U20")

        for i, f in enumerate(files):
            h = fits.open(f)[0].header
            ob[i] = h.get(info["target"], "")
            ty[i] = h.get(info["observation_type"], "")
            # The mode descriptor has different names in different files, so try different ids
            mo[i] = h.get(info["instrument_mode"])
            ni_tmp = h.get(info["date"], "")
            it[i] = h.get(info["instrument"], "")
            se[i] = h.get(info["setting_specifier"], "")

            # Sanitize input
            ni[i] = observation_date_to_night(ni_tmp)
            ob[i] = ob[i].replace("-", "")

        if isinstance(individual_nights, str) and individual_nights == "all":
            individual_nights = np.unique(ni)
            logger.info(
                "Can't parse night %s, use all %i individual nights instead",
                night,
                len(individual_nights),
            )

        files_per_night = []
        nights_out = []
        for ind_night in individual_nights:
            # Select files for this night, this instrument, this instrument mode
            selection = (ni == ind_night) & (it == instrument) & (mo == mode_id)

            # Find all unique setting keys for this night and target
            # Only look at the settings of observation files
            keys = se[(ty == info["id_spec"]) & (ob == target) & selection]
            keys = np.unique(keys)

            files_this_night = {}
            for key in keys:
                select = selection & (se == key)

                # find all relevant files for this setting
                # bias ignores the setting
                files_this_night[key] = {
                    "bias": files[(ty == info["id_bias"]) & selection],
                    "flat": files[(ty == info["id_flat"]) & select],
                    "orders": files[(ty == info["id_orders"]) & select],
                    "wavecal": files[(ob == info["id_wave"]) & select],
                    "science": files[(ty == info["id_spec"]) & (ob == target) & select],
                }
                files_this_night[key]["scatter"] = files_this_night[key]["orders"]

            if len(keys) != 0:
                nights_out.append(ind_night)
                files_per_night.append(files_this_night)

        return files_per_night, nights_out

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
        instrument = "wavecal"

        cwd = os.path.dirname(__file__)
        fname = "{instrument}_{mode}_{specifier}.npz".format(
            instrument=instrument.lower(), mode=mode, specifier=specifier
        )
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname

    def get_supported_modes(self):
        info = self.load_info()
        return info["modes"]


class COMMON(instrument):
    def load_info(self):
        return {
            "naxis_x": "NAXIS1",
            "naxis_y": "NAXIS2",
            "modes": [""],
            "modes_id": [""],
        }

