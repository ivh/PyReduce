"""
Handles instrument specific info for the HARPS spectrograph

Mostly reading data from the header
"""
import os.path
import glob
import logging
from datetime import datetime
import fnmatch
import re

import numpy as np
from astropy.io import fits
from astropy.time import Time
from dateutil import parser

from .common import getter, instrument, observation_date_to_night

logger = logging.getLogger(__name__)


class MCDONALD(instrument):
    def _convert_time_deg(self, v):
        v = [float(s) for s in v.split(":")]
        v = v[0] + v[1] / 60 + v[2] / 3600
        return v

    def add_header_info(self, header, mode, **kwargs):
        """ read data from header and add it as REDUCE keyword back to the header """
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works

        header = super().add_header_info(header, mode, **kwargs)
        info = self.load_info()
        get = getter(header, info, mode)

        header["e_orient"] = get("orientation", 0)

        trimsec = get("trimsec")

        if trimsec is not None:
            pattern = r"\[(\d*?):(\d*?),(\d*?):(\d*?)\]"
            res = re.match(pattern, trimsec)
            prescan_x = int(res[1]) + 1
            overscan_x = int(res[2])
            prescan_y = int(res[3])
            overscan_y = int(res[4])
        else:
            prescan_x = 2
            overscan_x = 2048
            prescan_y = 2
            overscan_y = 2047

        header["e_xlo"] = prescan_x
        header["e_xhi"] = overscan_x

        header["e_ylo"] = prescan_y
        header["e_yhi"] = overscan_y

        amp = get("amplifier")
        gain = info["gain"].format(amplifier=amp)
        readnoise = info["readnoise"].format(amplifier=amp)

        header["e_gain"] = get(gain, 1)
        header["e_readn"] = get(readnoise, 0)
        header["e_exptim"] = get("exposure_time", 0)

        header["e_sky"] = get("sky", 0)
        header["e_drk"] = get("dark", 0)
        header["e_backg"] = header["e_gain"] * (header["e_drk"] + header["e_sky"])

        header["e_imtype"] = get("observation_type")
        header["e_ctg"] = get("observation_type")

        obs_date = get("date")
        ut = get("universal_time")
        dark_time = get("dark_time")
        ra = get("ra")
        dec = get("dec")

        if ra is not None:
            ra = self._convert_time_deg(ra)
        if dec is not None:
            dec = self._convert_time_deg(dec)
        if ut is not None and dark_time is not None:
            tmid = self._convert_time_deg(ut) + dark_time / 2
        else:
            tmid = 0
        if obs_date is not None:
            jd = Time(obs_date).mjd + tmid + 0.5
        else:
            jd = 0

        header["e_ra"] = ra
        header["e_dec"] = dec
        header["e_jd"] = jd

        header["e_obslon"] = self._convert_time_deg(info["longitude"])
        header["e_obslat"] = self._convert_time_deg(info["latitude"])
        header["e_obsalt"] = info["altitude"]

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
        instrument = "MCDONALD"

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
        mode_id = info["modes_id"][i].lower()

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
        ni = np.zeros(len(files), dtype=datetime)
        # instrument, used for observation
        it = np.zeros(len(files), dtype="U20")

        for i, f in enumerate(files):
            h = fits.open(f)[0].header
            ob[i] = h.get(info["target"], "").upper()
            ty[i] = h.get(info["observation_type"], "").upper()
            # The mode descriptor has different names in different files, so try different ids
            mo[i] = h.get(info["instrument_mode"])
            ni_tmp = h.get(info["date"], "") + "T" + h.get(info["universal_time"], "")

            it[i] = h.get(info["instrument"], "")
            se[i] = "MCDONALD"

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
            match_ty = np.array([fnmatch.fnmatch(t, info["id_spec"]) for t in ty])
            match_ob = np.array([fnmatch.fnmatch(t, target) for t in ob])

            keys = se[match_ty & match_ob & selection]
            keys = np.unique(keys)

            files_this_night = {}
            for key in keys:
                select = selection & (se == key)

                # find all relevant files for this setting
                # bias ignores the setting
                files_this_night = {
                    "bias": files[(ob == info["id_bias"]) & selection],
                    "flat": files[(ty == info["id_flat"]) & select],
                    "orders": files[(ty == info["id_flat"]) & select],
                    "wavecal": files[(ob == info["id_wave"]) & select],
                    "curvature": files[(ob == info["id_wave"]) & select],
                    "science": files[match_ty & match_ob & select],
                }
                # Use science frame to find orders
                files_this_night["orders"] = [files_this_night["science"][0]]
                files_this_night["scatter"] = files_this_night["flat"]
                files_per_night.append(
                    (
                        {"night": ind_night, "key": key, "target": target},
                        files_this_night,
                    )
                )

        return files_per_night

    def get_wavecal_filename(self, header, mode, **kwargs):
        """ Get the filename of the wavelength calibration config file """
        cwd = os.path.dirname(__file__)
        fname = "{instrument}_{mode}_2D.npz".format(instrument="harps", mode=mode)
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname
