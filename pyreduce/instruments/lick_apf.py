# -*- coding: utf-8 -*-
"""
Handles instrument specific info for the HARPS spectrograph

Mostly reading data from the header
"""
import logging
import os.path

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.io import fits
from dateutil import parser

from .common import Instrument, getter, observation_date_to_night

logger = logging.getLogger(__name__)


class LICK_APF(Instrument):
    def add_header_info(self, header, mode, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, mode)
        info = self.load_info()

        # pos = EarthLocation.of_site("Lick Observatory")
        # header["e_obslon"] = pos.lon.to_value("deg")
        # header["e_obslat"] = pos.lat.to_value("deg")
        # header["e_obsalt"] = pos.height.to_value("m")

        return header

    def get_wavecal_filename(self, header, mode, **kwargs):
        """Get the filename of the wavelength calibration config file"""
        cwd = os.path.dirname(__file__)
        fname = "lick_apf_2D.npz"
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname
