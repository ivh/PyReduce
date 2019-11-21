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
from dateutil import parser

from .common import getter, instrument, observation_date_to_night

logger = logging.getLogger(__name__)


class HARPS(instrument):
    def get_extension(self, header, mode):
        extension = super().get_extension(header, mode)

        try:
            if (
                header["NAXIS"] == 2
                and header["NAXIS1"] == 4296
                and header["NAXIS2"] == 4096
            ):
                extension = 0
        except KeyError:
            pass

        return extension

    def add_header_info(self, header, mode, **kwargs):
        """ read data from header and add it as REDUCE keyword back to the header """
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, mode)
        info = self.info

        try:
            header["e_ra"] /= 15
            header["e_jd"] += header["e_exptim"] / (7200 * 24) + 0.5

            pol_angle = header.get("eso ins ret25 pos")
            if pol_angle is None:
                pol_angle = header.get("eso ins ret50 pos")
                if pol_angle is None:
                    pol_angle = "no polarimeter"
                else:
                    pol_angle = "lin %i" % pol_angle
            else:
                pol_angle = "cir %i" % pol_angle

            header["e_pol"] = (pol_angle, "polarization angle")
        except:
            pass

        try:
            if (
                header["NAXIS"] == 2
                and header["NAXIS1"] == 4296
                and header["NAXIS2"] == 4096
            ):
                # both modes are in the same image
                prescan_x = 50
                overscan_x = 50
                naxis_x = 2148
                if mode == "BLUE":
                    header["e_xlo"] = prescan_x
                    header["e_xhi"] = naxis_x - overscan_x
                elif mode == "RED":
                    header["e_xlo"] = naxis_x + prescan_x
                    header["e_xhi"] = 2 * naxis_x - overscan_x
        except KeyError:
            pass

        return header

    def sort_files(self, input_dir, target, night, mode, fiber, polarimetry):
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
        target = target.replace("-", "").upper()
        instrument = info["__instrument__"].upper()

        if fiber == "AB":
            template = "{a},{a},{c}"
        elif fiber == "A":
            template = "{a},{b},{c}"
        elif fiber == "B":
            template = "{b},{a},{c}"
        else:
            raise ValueError(
                "fiber keyword not understood, possible values are 'AB', 'A', 'B'"
            )

        id_orddef = template.format(a="LAMP", b="DARK", c="*")
        id_flat = "LAMP,LAMP,*" #template.format(a="LAMP", b="LAMP", c="*")
        id_spec = template.format(a="STAR", b="*", c="*")

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
        i = [i for i, m in enumerate(info["modes"]) if m == mode.upper()][0]
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
        ni = np.zeros(len(files), dtype=datetime)
        # instrument, used for observation
        it = np.zeros(len(files), dtype="U20")

        for i, f in enumerate(files):
            h = fits.open(f)[0].header
            ob[i] = h.get(info["target"], "")
            ty[i] = h.get(info["observation_type"], "")
            # The mode descriptor has different names in different files, so try different ids
            mo[i] = h.get(info["instrument_mode"])
            if mo[i] is None:
                mo[i] = h.get(info["instrument_mode_alternative"], "")[:3]
            ni_tmp = h.get(info["date"], "")
            it[i] = h.get(info["instrument"], "")
            se[i] = "HARPS"

            # Sanitize input
            ni[i] = observation_date_to_night(ni_tmp)
            ob[i] = ob[i].replace("-", "").upper()

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
            match_ty = np.array([fnmatch.fnmatch(t, id_spec) for t in ty])
            match_ob = np.array([fnmatch.fnmatch(t, target) for t in ob])
            match_flat = np.array([fnmatch.fnmatch(t, id_flat) for t in ty])
            match_ord = np.array([fnmatch.fnmatch(t, id_orddef) for t in ty])
            match_bias = ty == info["id_bias"]
            match_wave = ty == info["id_wave"]
            match_comb = ty == info["id_comb"]

            keys = se[match_ty & match_ob & selection]
            keys = np.unique(keys)

            files_this_night = {}
            for key in keys:
                select = selection & (se == key)

                # find all relevant files for this setting
                # bias ignores the setting
                files_this_night[key] = {
                    "bias": files[match_bias & selection],
                    "flat": files[match_flat & select],
                    "orders": files[match_ord & select],
                    "wavecal": files[match_wave & select],
                    "freq_comb": files[match_comb & select],
                    "science": files[match_ty & match_ob & select],
                }

                if len(files_this_night[key]["bias"]) == 0:
                    next_best_night = ni[match_bias][
                        np.argsort(np.abs(ni[match_bias] - ind_night))
                    ]
                    if len(next_best_night) > 0:
                        files_this_night[key]["bias"] = files[
                            match_bias & (ni == next_best_night[0])
                        ]
                        logger.warning(
                            "Using bias from night %s for observations of night %s",
                            next_best_night[0],
                            ind_night,
                        )

                if len(files_this_night[key]["flat"]) == 0:
                    next_best_night = ni[match_flat][
                        np.argsort(np.abs(ni[match_flat] - ind_night))
                    ]
                    if len(next_best_night) > 0:
                        files_this_night[key]["flat"] = files[
                            match_flat & (ni == next_best_night[0])
                        ]
                        logger.warning(
                            "Using flat from night %s for observations of night %s",
                            next_best_night[0],
                            ind_night,
                        )

                if len(files_this_night[key]["orders"]) == 0:
                    next_best_night = ni[match_ord][
                        np.argsort(np.abs(ni[match_ord] - ind_night))
                    ]
                    if len(next_best_night) > 0:
                        files_this_night[key]["orders"] = files[
                            match_ord & (ni == next_best_night[0])
                        ]
                        logger.warning(
                            "Using order definition from night %s for observations of night %s",
                            next_best_night[0],
                            ind_night,
                        )

                if len(files_this_night[key]["wavecal"]) == 0:
                    next_best_night = ni[match_wave][
                        np.argsort(np.abs(ni[match_wave] - ind_night))
                    ]
                    if len(next_best_night) > 0:
                        files_this_night[key]["wavecal"] = files[
                            match_wave & (ni == next_best_night[0])
                        ]
                        logger.warning(
                            "Using wavecal from night %s for observations of night %s",
                            next_best_night[0],
                            ind_night,
                        )

                files_this_night[key]["curvature"] = (
                    files_this_night[key]["freq_comb"]
                    if len(files_this_night[key]["freq_comb"]) != 0
                    else files_this_night[key]["wavecal"]
                )

                files_this_night[key]["scatter"] = files_this_night[key]["orders"]

            if len(keys) != 0:
                nights_out.append(ind_night)
                files_per_night.append(files_this_night)

        return files_per_night, nights_out

    def get_wavecal_filename(self, header, mode, **kwargs):
        """ Get the filename of the wavelength calibration config file """
        cwd = os.path.dirname(__file__)
        fname = "{instrument}_{mode}_2D.npz".format(
            instrument="harps", mode=mode.lower()
        )
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname
