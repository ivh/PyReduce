# -*- coding: utf-8 -*-
"""
Handles instrument specific info for the HARPS spectrograph

Mostly reading data from the header
"""
import logging
import os.path
import re
from itertools import product

import numpy as np
from astropy.io import fits
from dateutil import parser

from .common import Instrument, getter, observation_date_to_night
from .filters import Filter

logger = logging.getLogger(__name__)


class CRIRES_PLUS(Instrument):
    def __init__(self):
        super().__init__()
        self.filters["lamp"] = Filter(self.info["id_lamp"])
        self.filters["band"] = Filter(self.info["id_band"])
        self.filters["decker"] = Filter(self.info["id_decker"])
        self.shared += ["band", "decker"]

    def add_header_info(self, header, mode, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        band, decker, detector = self.parse_mode(mode)
        header = super().add_header_info(header, band)
        info = self.load_info()

        return header

    def get_supported_modes(self):
        settings = self.info["settings"]
        deckers = self.info["deckers"]
        detectors = self.info["chips"]
        modes = [
            "_".join([s, d, c]) for s, d, c in product(settings, deckers, detectors)
        ]
        return modes

    def parse_mode(self, mode):
        pattern = r"([YJHKLM]\d{4})(_(Open|pos1|pos2))?_det(\d)"
        match = re.match(pattern, mode, flags=re.IGNORECASE)
        band = match.group(1).upper()
        if match.group(3) is not None:
            decker = match.group(3).lower().capitalize()
        else:
            decker = "Open"
        detector = match.group(4)
        return band, decker, detector

    def get_expected_values(self, target, night, mode):
        expectations = super().get_expected_values(target, night)
        band, decker, detector = self.parse_mode(mode)

        for key in expectations.keys():
            if key == "bias":
                continue
            expectations[key]["band"] = band
            expectations[key]["decker"] = decker

        return expectations

    def get_extension(self, header, mode):
        band, decker, detector = self.parse_mode(mode)
        extension = int(detector)
        return extension

    def get_wavecal_filename(self, header, mode, **kwargs):
        """Get the filename of the wavelength calibration config file"""
        cwd = os.path.dirname(__file__)
        fname = "{instrument}_{mode}.npz".format(instrument=self.name, mode=mode)
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname

    def get_mask_filename(self, mode, **kwargs):
        i = self.name.lower()
        band, decker, detector = self.parse_mode(mode)

        fname = f"mask_{i}_det{detector}.fits.gz"
        cwd = os.path.dirname(__file__)
        fname = os.path.join(cwd, "..", "masks", fname)
        return fname

    def get_wavelength_range(self, header, mode, **kwargs):
        wmin = [header["ESO INS WLEN MIN%i" % i] for i in range(1, 11)]
        wmax = [header["ESO INS WLEN MAX%i" % i] for i in range(1, 11)]

        wavelength_range = np.array([wmin, wmax]).T
        # Invert the order numbering
        wavelength_range = wavelength_range[::-1]
        # Convert from nm to Angstrom
        wavelength_range *= 10
        return wavelength_range
