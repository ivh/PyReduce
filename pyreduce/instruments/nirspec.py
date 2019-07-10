"""
Handles instrument specific info for the UVES spectrograph

Mostly reading data from the header
"""
import os.path
import glob
import logging
from datetime import datetime

from tqdm import tqdm
import numpy as np
from astropy.io import fits
from dateutil import parser

from .common import getter, instrument, observation_date_to_night


class NIRSPEC(instrument):

    def add_header_info(self, header, mode, **kwargs):
        """ read data from header and add it as REDUCE keyword back to the header """
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, mode)
        return header

    def sort_files(self, input_dir, target, night, mode, calibration_dir, **kwargs):
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
        target = target.casefold()
        instrument = self.__class__.__name__

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


        # Initialize arrays
        # observed object
        ob = np.zeros(len(files), dtype="U20")
        # observed night, parsed into a datetime object
        ni = np.zeros(len(files), dtype=datetime)
        # instrument, used for observation
        it = np.zeros(len(files), dtype="U20")

        for i, f in enumerate(files):
            h = fits.open(f)[0].header
            ob[i] = h.get(info["target"], "")
            ni_tmp = h.get(info["date"], "")
            it[i] = h.get(info["instrument"], "")

            # Sanitize input
            ni[i] = observation_date_to_night(ni_tmp)
            ob[i] = ob[i].replace("-", "").replace(" ", "").casefold()

        if isinstance(individual_nights, str) and individual_nights == "all":
            individual_nights = np.unique(ni)
            logging.info(
                "Can't parse night %s, use all %i individual nights instead",
                night,
                len(individual_nights),
            )

        files_per_observation = []
        nights_out = []
        cache = {}

        for ind_night in tqdm(individual_nights):
            # Select files for this night, this instrument, this instrument mode
            selection = (
                (ni == ind_night)
                & (it == instrument)
                & (ob == target)
                )

            files_this_observation = {}
            for f in tqdm(files[selection]):

                # Read caliblist
                caliblist = f[:-8] + ".caliblist"
                caliblist = np.genfromtxt(caliblist, skip_header=8, dtype=str, delimiter=" ", usecols=(0))
                caliblist = np.array([os.path.join(input_dir, calibration_dir, c) + ".gz" for c in caliblist])

                # Cache calibration file types

                tp = np.zeros(len(caliblist), dtype="U20")
                for i, c in enumerate(caliblist):
                    try:
                        tp[i] = cache[c]
                    except KeyError:
                        h = fits.open(c)[0].header
                        if h[info["id_flat"]] == 1:
                            tp[i] = "flat"
                        elif h[info["id_neon"]] == 1 or h[info["id_argon"]] == 1 or h[info["id_krypton"]] == 1 or h[info["id_xenon"]] == 1:
                            tp[i] = "wavecal"
                        elif h[info["id_etalon"]] == 1:
                            tp[i] = "freq_comb"
                        elif h["OBJECT"] != "test":
                            tp[i] = "bias"
                        else:
                            tp[i] = "-"
                        cache[c] = tp[i]

                files_this_observation["NIRSPEC"] = {
                    "bias": caliblist[tp == "bias"],
                    "flat": caliblist[tp == "flat"],
                    "orders": caliblist[tp == "flat"],
                    "wavecal": caliblist[tp == "wavecal"],
                    "freq_comb": caliblist[tp == "freq_comb"],
                    "science": [f]
                }
                files_this_observation["NIRSPEC"]["curvature"] = files_this_observation["NIRSPEC"]["freq_comb"] if len(files_this_observation["NIRSPEC"]["freq_comb"]) != 0 else files_this_observation["NIRSPEC"]["wavecal"]

                files_per_observation.append(files_this_observation)
                nights_out.append(ind_night)

        return files_per_observation, nights_out

    def get_wavecal_filename(self, header, mode, **kwargs):
        """ Get the filename of the wavelength calibration config file """
        info = self.load_info()
        specifier = int(header[info["wavecal_specifier"]])

        cwd = os.path.dirname(__file__)
        fname = "{instrument}_{mode}_{specifier}nm_2D.npz".format(
            instrument="uves", mode=mode, specifier=specifier
        )
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname
