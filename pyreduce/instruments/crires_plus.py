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

from .common import getter, Instrument, observation_date_to_night
from .filters import Filter

logger = logging.getLogger(__name__)


class CRIRES_PLUS(Instrument):
    def __init__(self):
        super().__init__()
        self.filters["lamp"] = Filter(self.info["id_lamp"])
        self.filters["band"] = Filter(self.info["id_band"])
        self.filters["decker"] = Filter(self.info["id_decker"])

    def add_header_info(self, header, mode, **kwargs):
        """ read data from header and add it as REDUCE keyword back to the header """
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, mode)
        info = self.load_info()

        return header

    def get_expected_values(self, target, night, mode):
        expectations = super().get_expected_values(target, night)

        band, decker = mode.rsplit("_", 1)
        for key in expectations.keys():
            if key == "bias":
                continue
            expectations[key]["band"] = band
            expectations[key]["decker"] = decker

        expectations["wavecal"]["lamp"] = self.info["id_lamp_wavecal"]
        expectations["freq_comb"]["lamp"] = self.info["id_lamp_etalon"]

        return expectations

    # def sort_files(self, input_dir, target, night, mode):
    #     """
    #     Sort a set of fits files into different categories
    #     types are: bias, flat, wavecal, orderdef, spec

    #     Parameters
    #     ----------
    #     input_dir : str
    #         input directory containing the files to sort
    #     target : str
    #         name of the target as in the fits headers
    #     night : str
    #         observation night, possibly with wildcards
    #     mode : str
    #         instrument mode
    #     Returns
    #     -------
    #     files_per_night : list[dict{str:dict{str:list[str]}}]
    #         a list of file sets, one entry per night, where each night consists of a dictionary with one entry per setting,
    #         each fileset has five lists of filenames: "bias", "flat", "order", "wave", "spec", organised in another dict
    #     nights_out : list[datetime]
    #         a list of observation times, same order as files_per_night
    #     """

    #     info = self.load_info()
    #     target = target.upper().replace("-", "")
    #     instrument = "CRIRES"

    #     # Try matching with nights
    #     try:
    #         night = parser.parse(night).date()
    #         individual_nights = [night]
    #     except ValueError:
    #         # if the input night can't be parsed, use all nights
    #         # Usually the case if wildcards are involved
    #         individual_nights = "all"

    #     # find all fits files in the input dir(s)
    #     input_dir = input_dir.format(
    #         instrument=instrument.upper(), target=target, mode=mode, night=night
    #     )
    #     files = glob.glob(input_dir + "/*.fits")
    #     files += glob.glob(input_dir + "/*.fits.gz")
    #     files = np.array(files)

    #     # Initialize arrays
    #     # observed object
    #     ob = np.zeros(len(files), dtype="U20")
    #     # observation type, bias, flat, spec, etc.
    #     ty = np.zeros(len(files), dtype="U20")
    #     # instrument mode, e.g. red, blue
    #     mo = np.zeros(len(files), dtype="U20")
    #     band = np.zeros(len(files), dtype="U20")
    #     decker = np.zeros(len(files), dtype="U20")
    #     lamp = np.zeros(len(files), dtype="U20")

    #     se = np.zeros(len(files), dtype="U20")

    #     # observed night, parsed into a datetime object
    #     ni = np.zeros(len(files), dtype=datetime)
    #     # instrument, used for observation
    #     it = np.zeros(len(files), dtype="U20")

    #     for i, f in enumerate(files):
    #         h = fits.open(f)[0].header
    #         ob[i] = h.get(info["target"], "")
    #         ty[i] = h.get(info["observation_type"], "")
    #         mo[i] = h.get(info["id_mode"])
    #         band[i] = h.get(info["id_band"], "").replace(r"/", "_")
    #         decker[i] = h.get(info["id_decker"], "").upper()
    #         lamp_tmp = h.get(info["id_lamp"], "")
    #         if len(lamp_tmp) != 0:
    #             lamp[i] = lamp_tmp[0]
    #         se[i] = band[i] + "_" + decker[i]
    #         ni_tmp = h.get(info["date"], "")
    #         it[i] = h.get(info["instrument"], "")
    #         # Sanitize input
    #         ni[i] = observation_date_to_night(ni_tmp)
    #         ob[i] = ob[i].replace("-", "")

    #     if isinstance(individual_nights, str) and individual_nights == "all":
    #         individual_nights = np.unique(ni)
    #         logger.info(
    #             "Can't parse night %s, use all %i individual nights instead",
    #             night,
    #             len(individual_nights),
    #         )

    #     files_per_night = []
    #     nights_out = []
    #     for ind_night in individual_nights:
    #         # Select files for this night, this instrument, this instrument mode
    #         selection = (ni == ind_night) & (it == instrument)

    #         # Find all unique setting keys for this night and target
    #         # Only look at the settings of observation files
    #         # match_ty = np.array([fnmatch.fnmatch(t, info["id_spec"]) for t in ty])
    #         # match_ob = np.array([fnmatch.fnmatch(t, target) for t in ob])

    #         if mode != "":
    #             keys = np.unique(se[selection & (se == mode)])
    #         else:
    #             keys = np.unique(se[selection])

    #         files_this_night = {}
    #         for key in keys:
    #             select = selection & (se == key)

    #             # find all relevant files for this setting
    #             # bias ignores the setting
    #             files_this_night = {
    #                 "bias": files[(ty == info["id_bias"]) & selection],
    #                 "flat": files[(ty == info["id_flat"]) & select],
    #                 "orders": files[(ty == info["id_flat"]) & select],
    #                 "wavecal": files[
    #                     (ty == info["id_wave"])
    #                     & (lamp == info["id_lamp_wavecal"])
    #                     & select
    #                 ],
    #                 "freq_comb": files[
    #                     (ty == info["id_wave"])
    #                     & (lamp == info["id_lamp_etalon"])
    #                     & select
    #                 ],
    #                 "science": [],
    #             }
    #             files_this_night["curvature"] = (
    #                 files_this_night["freq_comb"]
    #                 if len(files_this_night["freq_comb"]) != 0
    #                 else files_this_night["wavecal"]
    #             )
    #             files_this_night["scatter"] = files_this_night["orders"]

    #             files_per_night.append(
    #                 (
    #                     {"night": ind_night, "key": key, "target": target},
    #                     files_this_night,
    #                 )
    #             )

    #     return files_per_night

    def get_wavecal_filename(self, header, mode, **kwargs):
        """ Get the filename of the wavelength calibration config file """
        cwd = os.path.dirname(__file__)
        fname = "{instrument}_{mode}_2D.npz".format(instrument="harps", mode=mode)
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname
