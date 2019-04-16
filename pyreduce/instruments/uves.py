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


class UVES(instrument):
    def load_info(self):
        """ Load harcoded information about this instrument """

        # Tips & Tricks:
        # if several modes are supported, use a list for modes
        # if a value changes depending on the mode, use a list with the same order as "modes"
        # you can also use values from this dictionary as placeholders using {name}, just like str.format

        # red and middle are in the same fits file, with different extensions,
        # i.e. share the same mode identifier, but have different extensions
        info = {
            "__instrument__": "UVES",
            # General information
            "instrument": "INSTRUME",
            "date": "DATE-OBS",
            "modes": ["blue", "middle", "red"],
            "modes_id": ["blue", "red", "red"],
            "extension": [0, 2, 1],
            # Header info for reduction
            "id": [[1, 5], [1, 4], [1, 4]],
            "orientation": [6, 1, 1],
            "prescan_x": "HIERARCH ESO DET OUT{id[0]} PRSCX",
            "overscan_x": "HIERARCH ESO DET OUT{id[0]} OVSCX",
            "prescan_y": 0,
            "overscan_y": 0,
            "naxis_x": "NAXIS1",
            "naxis_y": "NAXIS2",
            "gain": "HIERARCH ESO DET OUT{id[0]} CONAD",
            "readnoise": "HIERARCH ESO DET OUT{id[0]} RON",
            "dark": "HIERARCH ESO INS DET{id[1]} OFFDRK",
            "sky": "HIERARCH ESO INS DET{id[1]} OFFSKY",
            "exposure_time": "EXPTIME",
            "image_type": "OBJECT",
            "category": "HIERARCH ESO DPR CATG",
            "ra": "RA",
            "dec": "DEC",
            "jd": "MJD-OBS",
            "longitude": "HIERARCH ESO TEL GEOLON",
            "latitude": "HIERARCH ESO TEL GEOLAT",
            "altitude": "HIERARCH ESO TEL GEOELEV",
            "wavecal_specifier": "ESO INS GRAT2 WLEN",
            # Ids for file sorting
            "target": "OBJECT",
            "observation_type": "ESO DPR TYPE",
            "instrument_mode": "ESO INS MODE",
            "instrument_mode_alternative": "ESO TPL NAME",
            "id_bias": "BIAS",
            "id_flat": "LAMP,FLAT",
            "id_wave": "LAMP,WAVE",
            "id_orders": "LAMP,ORDERDEF",
            "id_spec": "OBJECT,POINT",
        }
        return info

    def add_header_info(self, header, mode, **kwargs):
        """ read data from header and add it as REDUCE keyword back to the header """
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, mode)

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
            se[i] = h.get(info["wavecal_specifier"], "")

            # Sanitize input
            ni[i] = observation_date_to_night(ni_tmp)
            ob[i] = ob[i].replace("-", "")

        if isinstance(individual_nights, str) and individual_nights == "all":
            individual_nights = np.unique(ni)
            logging.info(
                "Can't parse night %s, use all %i individual nights instead",
                night,
                len(individual_nights),
            )

        files_per_night = []
        nights_out = []
        for ind_night in individual_nights:
            # Select files for this night, this instrument, this instrument mode
            selection = (
                (ni == ind_night)
                & (it == instrument)
                & ((mo == mode_id) | (mo == "None"))
            )

            # Find all unique setting keys for this night and target
            # Only look at the settings of observation files
            keys = se[(ty == info["id_spec"]) & (ob == target) & selection]
            keys = np.unique(keys)

            files_this_night = {}
            for key in keys:
                select = selection & (se == key)

                # find all relevant files for this setting
                # bias ignores the setting
                files_this_night[key + "nm"] = {
                    "bias": files[(ty == info["id_bias"]) & selection],
                    "flat": files[(ty == info["id_flat"]) & select],
                    "order": files[(ty == info["id_orders"]) & select],
                    "wave": files[(ob == info["id_wave"]) & select],
                    "spec": files[(ty == info["id_spec"]) & (ob == target) & select],
                }

            if len(keys) != 0:
                nights_out.append(ind_night)
                files_per_night.append(files_this_night)

        return files_per_night, nights_out

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
