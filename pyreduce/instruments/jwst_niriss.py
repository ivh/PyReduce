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
from astropy import units as q
from astropy.time import Time
from dateutil import parser

from .common import getter, instrument, observation_date_to_night

logger = logging.getLogger(__name__)


class JWST_NIRISS(instrument):
    def add_header_info(self, header, mode, **kwargs):
        """ read data from header and add it as REDUCE keyword back to the header """
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, mode)
        info = self.load_info()

        # TODO: this references some files, I dont know where they should be
        header["e_gain"] = 1.61
        header["e_readn"] = 18.32
        header["e_dark"] = 0.0257

        # total exposure time
        header["exptime"] = header.get("TFRAME", 0)

        return header

    def split_observation(self, fname, mode):
        hdu = fits.open(fname)
        dirname = os.path.dirname(fname)
        fname = os.path.basename(fname)

        header = hdu[0].header
        if "mjd-obs" not in header:
            if len(header["DATE-OBS"]) <= 10:
                time = header["DATE-OBS"] + "T" + header["TIME-OBS"]
            else:
                time = header["DATE-OBS"]
            header["MJD-OBS"] = Time(time).mjd

        header2 = fits.Header()
        header2["EXTNAME"] = "SCI"
        shape = hdu["SCI"].data.shape

        nframes = shape[0]
        ngroups = shape[1]

        if nframes == 1 and ngroups == 1:
            return [os.path.join(dirname, fname)]

        data = hdu["SCI"].data.reshape((-1, *shape[-2:]))
        bias = data[0]
        primary = fits.PrimaryHDU(header=header)
        files = []
        os.makedirs(os.path.join(dirname, "split"), exist_ok=True)
        for i in range(1, nframes * ngroups):
            this = data[i] - bias
            bias = data[i]
            fname_this = os.path.join(dirname, "split", f"pyreduce_{i}_{fname}")

            header["MJD-OBS"] += header["TFRAME"] * q.s.to(q.day)
            secondary = fits.ImageHDU(data=data, header=header2)
            hdu_this = fits.HDUList([primary, secondary])
            hdu_this.writeto(fname_this, overwrite=True)
            files.append(fname_this)
        return files

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
        instrument = "NIRISS"

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
            instrument=self.name.upper(), target=target, mode=mode, night=night
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
            se[i] = instrument

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
            # match_ty = np.array([fnmatch.fnmatch(t, info["id_spec"]) for t in ty])
            # match_ob = np.array([fnmatch.fnmatch(t, target) for t in ob])

            keys = se[selection]
            keys = np.unique(keys)

            files_this_night = {}
            for key in keys:
                select = selection & (se == key)

                # find all relevant files for this setting
                # bias ignores the setting
                # files_this_night[key] = {
                #     "bias": files[(ty == info["id_bias"]) & selection],
                #     "flat": files[(ty == info["id_flat"]) & select],
                #     "orders": files[(ty == info["id_flat"]) & select],
                #     "wavecal": []],
                #     "science": files[select],
                # }
                # TODO find actual flat files
                files_this_night = {}
                files_this_night["bias"] = []
                files_this_night["flat"] = [f for f in files if f.endswith("flat.fits")]
                files_this_night["orders"] = [files_this_night["flat"][0]]
                files_this_night["wavecal"] = []
                files_this_night["curvature"] = files_this_night["wavecal"]
                files_this_night["science"] = []
                files_this_night["scatter"] = files_this_night["orders"]

                f_science = [
                    f
                    for f in files
                    if not (f.endswith("flat.fits") or f.endswith("bias.fits"))
                ]
                for f in f_science:
                    files_this_night["science"] += self.split_observation(f, mode)
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
