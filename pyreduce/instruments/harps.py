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
from itertools import product

import numpy as np
from astropy.io import fits
from dateutil import parser

import re

from .common import getter, instrument, observation_date_to_night
from .filters import Filter, ObjectFilter, InstrumentFilter, NightFilter

logger = logging.getLogger(__name__)


class TypeFilter(Filter):
    def __init__(self, keyword="ESO DPR TYPE"):
        super().__init__(keyword, regex=True)

    def match(self, value):
        regex = re.compile(value)
        match = [regex.match(f) is not None for f in self.data]
        result = np.asarray(match)
        return result

    def classify(self, value):
        if value is not None:
            match = self.match(value)
            data = np.asarray(self.data)
            data = np.unique(data[match])
            try:
                regex = re.compile(value)
                data = [regex.match(f) for f in data]
                data = [[g for g in d.groups() if g is not None][0] for d in data]
                data = np.unique(data)
            except IndexError:
                data = np.asarray(self.data)
                data = np.unique(data[match])
        else:
            data = np.unique(self.data)
        data = [(d, self.match(d)) for d in data]
        return data


class FiberFilter(Filter):
    def __init__(self, keyword="ESO DPR TYPE"):
        super().__init__(keyword, regex=True)

    def collect(self, header):
        value = header.get(self.keyword)
        value = value.split(",")
        if value[0] in ["LAMP", "STAR"] and value[1] in ["LAMP", "STAR"]:
            value = "AB"
        elif value[1] in ["LAMP", "STAR"]:
            value = "B"
        elif value[0] in ["LAMP", "STAR"]:
            value = "A"
        else:
            value = ""

        self.data.append(value)
        return value


class PolarizationFilter(Filter):
    def __init__(self, keyword="ESO INS RET?? POS"):
        super().__init__(keyword)

    def collect(self, header):
        value = header.get("eso ins ret50 pos", None)
        if value is not None:
            value = "linear"
        else:
            value = header.get("eso ins ret25 pos", None)
            if value is not None:
                value = "circular"
            else:
                value = "none"
        self.data.append(value)
        return value


class HARPS(instrument):
    def __init__(self):
        super().__init__()
        self.filters = {
            "instrument": InstrumentFilter(self.info["instrument"]),
            "night": NightFilter(self.info["date"]),
            # "branch": Filter(, regex=True),
            "mode": Filter(self.info["instrument_mode"]),
            "type": TypeFilter(self.info["observation_type"]),
            "polarization": PolarizationFilter(),
            "target": ObjectFilter(self.info["target"], regex=True),
            "fiber": FiberFilter(),
        }
        self.night = "night"
        self.science = "science"
        self.shared = ["instrument", "night", "mode", "polarization", "fiber"]
        self.find_closest = [
            "bias",
            "flat",
            "wavecal",
            "orders",
            "scatter",
            "curvature",
        ]

    def get_expected_values(self, target, night, branch, fiber, polarimetry):
        """Determine the default expected values in the headers for a given observation configuration
        
        Any parameter may be None, to indicate that all values are allowed

        Parameters
        ----------
        target : str
            Name of the star / observation target
        night : str
            Observation night/nights
        fiber : "A", "B", "AB"
            Which of the fibers should carry observation signal
        polarimetry : "none", "linear", "circular", bool
            Whether the instrument is used in HARPS or HARPSpol mode
            and which polarization is observed. Set to true for both kinds
            of polarisation.
        
        Returns
        -------
        expectations: dict
            Dictionary of expected header values, with one entry per step.
            The entries for each step refer to the filters defined in self.filters
        
        Raises
        ------
        ValueError
            Invalid combination of parameters
        """
        # target = target.replace(" ", r"[\s*-]")
        if fiber == "AB":
            template = r"$({a},{a}),{c}^"
        elif fiber == "A":
            template = r"$({a},{b}),{c}^"
        elif fiber == "B":
            template = r"$({b},{a}),{c}^"
        elif fiber is None:
            template = None
            fiber = "(AB)|(A)|(B)"
        else:
            raise ValueError(
                "fiber keyword not understood, possible values are 'AB', 'A', 'B'"
            )

        if polarimetry == "none" or not polarimetry:
            mode = "HARPS"
            if template is not None:
                id_orddef = template.format(a="LAMP", b="DARK", c=".*?")
                id_spec = template.format(a="STAR", b="(?!STAR).*?", c=".*?")
            else:
                id_spec = (
                    r"^(STAR,(?!STAR).*),.*$|^((?!STAR).*?,STAR),.*$|^(STAR,STAR),.*$"
                )
                id_orddef = r"^(LAMP,DARK),.*$|^(DARK,LAMP),.*$|^(LAMP,LAMP),.*$"
            polarimetry = "none"
        else:
            mode = "HARPSpol"
            id_orddef = r"(LAMP,LAMP),.*"
            if polarimetry == r"linear":
                id_spec = r"(STAR,LINPOL),.*"
            elif polarimetry == "circular":
                id_spec = r"(STAR,CIRPOL),.*"
            else:
                raise ValueError(
                    f"polarization parameter not recognized. Expected one of 'none', 'linear', 'circular', but got {polarimetry}"
                )

        expectations = {
            "bias": {"instrument": "HARPS", "night": night, "type": r"BIAS,BIAS"},
            "flat": {"instrument": "HARPS", "night": night, "type": r"(LAMP,LAMP),.*"},
            "orders": {
                "instrument": "HARPS",
                "night": night,
                "fiber": fiber,
                "type": id_orddef,
            },
            "scatter": {
                "instrument": "HARPS",
                "night": night,
                "type": id_orddef,  # Same as orders
            },
            "curvature": {
                "instrument": "HARPS",
                "night": night,
                "type": [r"(WAVE,WAVE,COMB)", r"(WAVE,WAVE,THAR)\d?"],
            },
            "wavecal": {
                "instrument": "HARPS",
                "night": night,
                "type": r"(WAVE,WAVE,THAR)\d?",
            },
            "freq_comb": {
                "instrument": "HARPS",
                "night": night,
                "type": r"(WAVE,WAVE,COMB)",
            },
            "science": {
                "instrument": "HARPS",
                "night": night,
                "mode": mode,
                "type": id_spec,
                "fiber": fiber,
                "polarization": polarimetry,
                "target": target,
            },
        }
        return expectations

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

        if polarimetry and polarimetry != "none":
            id_orddef = "LAMP,LAMP,*"
            id_flat = "LAMP,LAMP,*"
            id_spec = "STAR,*POL,*"
        else:
            id_orddef = template.format(a="LAMP", b="DARK", c="*")
            id_flat = "LAMP,LAMP,*"  # template.format(a="LAMP", b="LAMP", c="*")
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
        if polarimetry and polarimetry != "none":
            mode_id = info["modes_id_polarimetry"][i].upper()
        else:
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
        # polarization
        po = np.zeros(len(files), dtype="U20")

        for i, f in enumerate(files):
            h = fits.open(f)[0].header
            ob[i] = h.get(info["target"], "")
            ty[i] = h.get(info["observation_type"], "")
            # The mode descriptor has different names in different files, so try different ids
            mo[i] = h.get(info["instrument_mode"]).upper()
            if mo[i] is None:
                mo[i] = h.get(info["instrument_mode_alternative"], "").upper()
            ni_tmp = h.get(info["date"], "")
            it[i] = h.get(info["instrument"], "")
            se[i] = "HARPS"
            if h.get(info["polarization_linear"]) is not None:
                po[i] = "linear"
            elif h.get(info["polarization_circular"]) is not None:
                po[i] = "circular"
            else:
                po[i] = "none"
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

            if polarimetry in ["linear", "circular", "none"]:
                match_po = po == polarimetry
            elif polarimetry:
                match_po = po != "none"
            else:
                match_po = po == "none"

            # Find all unique setting keys for this night and target
            # Only look at the settings of observation files
            match_ty = np.array([fnmatch.fnmatch(t, id_spec) for t in ty])
            match_ob = np.array([fnmatch.fnmatch(t, target) for t in ob])
            match_flat = np.array([fnmatch.fnmatch(t, id_flat) for t in ty])
            match_ord = np.array([fnmatch.fnmatch(t, id_orddef) for t in ty])
            match_bias = ty == info["id_bias"]
            match_wave = ty == info["id_wave"]
            match_comb = ty == info["id_comb"]

            # Check if there are science files
            keys = se[match_ty & match_ob & match_po & selection]
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
                    "science": files[match_ty & match_ob & match_po & select],
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
            else:
                logger.warning(f"No science files found for night: {ind_night}")
                logger.debug("------------------")
                logger.debug(f"files: {list(files)}")
                logger.debug(f"nights: {list(ni==ind_night)}")
                logger.debug(f"instrument: {list(it==instrument)}")
                logger.debug(f"mode: {list(mo == mode_id)}")
                logger.debug(f"observation type: {list(match_ty)}")
                logger.debug(f"target: {list(match_ob)}")
                logger.debug(f"polarization: {list(match_po)}")
                logger.debug("------------------")

        return files_per_night, nights_out

    def get_wavecal_filename(self, header, mode, polarimetry, **kwargs):
        """ Get the filename of the wavelength calibration config file """
        cwd = os.path.dirname(__file__)
        if polarimetry:
            pol = "_pol"
        else:
            pol = ""
        fname = f"harps_{mode}{pol}_2D.npz"
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname
