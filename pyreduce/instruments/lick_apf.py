"""
Handles instrument specific info for the HARPS spectrograph

Mostly reading data from the header
"""
import os.path
import glob
import logging
from datetime import datetime
import fnmatch
import json

import numpy as np
from astropy.io import fits
from astropy.coordinates import EarthLocation
from dateutil import parser

from .common import getter, instrument, observation_date_to_night

logger = logging.getLogger(__name__)


class LICK_APF(instrument):
    def add_header_info(self, header, mode, **kwargs):
        """ read data from header and add it as REDUCE keyword back to the header """
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, mode)
        info = self.load_info()

        # pos = EarthLocation.of_site("Lick Observatory")
        # header["e_obslon"] = pos.lon.to_value("deg")
        # header["e_obslat"] = pos.lat.to_value("deg")
        # header["e_obsalt"] = pos.height.to_value("m")

        return header

    def sort_files(self, input_dir, target, night, mode):
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

        info = self.load_info()
        target = target.upper()
        instrument = "APF"

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
        # observed night, parsed into a datetime object
        ni = np.zeros(len(files), dtype=datetime)
        # instrument, used for observation
        it = np.zeros(len(files), dtype="U20")

        for i, f in enumerate(files):
            h = fits.open(f)[0].header
            ob[i] = h.get(info["target"], "")
            ty[i] = h.get(info["observation_type"], "")
            ni_tmp = h.get(info["date"], "")
            it[i] = h.get(info["instrument"], "").upper()
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
            selection = (ni == ind_night) & (it == instrument)
            if len(selection) == 0:
                continue
            # find all relevant files for this setting
            # bias ignores the setting
            files_this_night = {
                "bias": files[(ty == info["id_bias"]) & selection],
                "flat": files[(ob == info["id_flat"]) & selection],
                "orders": files[(ob == info["id_order"]) & selection],
                "wavecal": files[(ob == info["id_wave"]) & selection],
                # "freq_comb": files[(ty == info["id_comb"]) & select],
                "science": files[(ty == info["id_spec"]) & (ob == target) & selection],
            }
            files_this_night["scatter"] = files_this_night["orders"]

            if len(files_this_night["science"]) != 0:
                files_per_night.append(
                    ({"night": ind_night, "target": target}, files_this_night)
                )

        return files_per_night

    def get_wavecal_filename(self, header, mode, **kwargs):
        """ Get the filename of the wavelength calibration config file """
        cwd = os.path.dirname(__file__)
        fname = "lick_apf_2D.npz"
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname
