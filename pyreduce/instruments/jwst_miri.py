# -*- coding: utf-8 -*-
"""
Handles instrument specific info for the HARPS spectrograph

Mostly reading data from the header
"""
import logging
import os.path

import numpy as np
from astropy.io import fits
from dateutil import parser

from .common import Instrument, getter, observation_date_to_night

logger = logging.getLogger(__name__)


class JWST_MIRI(Instrument):
    def add_header_info(self, header, mode, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, mode)
        info = self.load_info()
        return header

    def get_wavecal_filename(self, header, mode, **kwargs):
        """Get the filename of the wavelength calibration config file"""
        cwd = os.path.dirname(__file__)
        fname = "{instrument}_{mode}_2D.npz".format(instrument="harps", mode=mode)
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname
