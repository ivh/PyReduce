"""
Handles instrument specific info for the UVES spectrograph

Mostly reading data from the header
"""
import os.path
import glob
import logging
from datetime import datetime

import numpy as np
from astropy.io import fits
from dateutil import parser

from .common import getter, instrument, observation_date_to_night

logger = logging.getLogger(__name__)


class XSHOOTER(instrument):
    def add_header_info(self, header, mode, **kwargs):
        """ read data from header and add it as REDUCE keyword back to the header """
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, mode)

        header["e_backg"] = (
            header["e_readn"] + header["EXPTIME"] * header["e_drk"] / 3600
        )

        header["e_ra"] /= 15
        header["e_jd"] += header["MJD-OBS"] + header["EXPTIME"] / 2 / 3600 / 24 + 0.5

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
        instrument = info["__instrument__"].upper()

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
            instrument=instrument, target=target, mode=mode, night=night
        )
        files = glob.glob(input_dir + "/*.fits")
        files += glob.glob(input_dir + "/*.fits.gz")
        files = np.array(files)

        # Initialize arrays
        # observed object
        ob = np.zeros(len(files), dtype="U20")
        ob2 = np.zeros(len(files), dtype="U20")
        # observation type, bias, flat, spec, etc.
        ty = np.zeros(len(files), dtype="U20")
        # instrument mode, e.g. red, blue
        mo = np.zeros(len(files), dtype="U20")
        # observation category
        cg = np.zeros(len(files), dtype="U20")
        # observed night, parsed into a datetime object
        ni = np.zeros(len(files), dtype=datetime)
        # instrument, used for observation
        it = np.zeros(len(files), dtype="U20")

        for i, f in enumerate(files):
            h = fits.open(f)[0].header
            ob[i] = h.get(info["target"], "")
            ob2[i] = h.get(info["object"], "")
            ty[i] = h.get(info["observation_type"], "")
            mo[i] = h.get(info["instrument_mode"])
            cg[i] = h.get(info["observation_category"], "")
            ni_tmp = h.get(info["date"], "")
            it[i] = h.get(info["instrument"], "")

            # Sanitize input
            ni[i] = observation_date_to_night(ni_tmp)
            ob[i] = ob[i].replace("-", "").upper()

        if isinstance(individual_nights, str) and individual_nights == "all":
            individual_nights = [n for n in ni if n is not None]
            individual_nights = np.unique(individual_nights)
            logger.info(
                "Can't parse night %s, use all %i individual nights instead",
                night,
                len(individual_nights),
            )

        instrument = "SHOOT"
        target = target.replace("-", "").upper()

        files_per_night = []
        nights_out = []
        for ind_night in individual_nights:
            # Select files for this night, this instrument, this instrument mode
            selection = (ni == ind_night) & (it == instrument) & (mo == mode)

            files_this_night = {
                "bias": files[(ty == info["id_dark"]) & selection],
                "flat": files[(ty == info["id_flat"]) & selection],
                "orders": files[(ty == info["id_orders"]) & selection],
                "wavecal": files[(ty == info["id_wavecal"]) & selection],
                "science": files[(ty == info["id_spec"]) & (ob == target) & selection],
                "telluric": files[(ty == info["id_tell"]) & selection],
            }

            for step in ["bias", "flat", "orders", "wavecal"]:
                if len(files_this_night[step]) == 0:
                    id_step = ty == info[f"id_{step}"]
                    try:
                        i = np.argsort(np.abs(ni[id_step] - ind_night))[0]
                        closest = ni[id_step][i]
                        files_this_night[step] = files[
                            id_step
                            & (ni == closest)
                            & (it == instrument)
                            & (mo == mode)
                        ]
                    except IndexError:
                        pass

            files_this_night["curvature"] = files_this_night["wavecal"]
            files_this_night["scatter"] = files_this_night["orders"]
            files_per_night.append(
                ({"night": ind_night, "mode": mode, "target": target}, files_this_night)
            )

        return files_per_night

    def get_wavecal_filename(self, header, mode, **kwargs):
        """ Get the filename of the wavelength calibration config file """
        info = self.load_info()

        cwd = os.path.dirname(__file__)
        fname = f"xshooter_{mode.lower()}.npz"
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname
