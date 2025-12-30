"""
Handles instrument specific info for the HARPS spectrograph

Mostly reading data from the header
"""

import logging
import os.path
import re
from itertools import product

import numpy as np

from .common import Instrument
from .filters import Filter

logger = logging.getLogger(__name__)


class CRIRES_PLUS(Instrument):
    def __init__(self):
        super().__init__()
        self.filters["lamp"] = Filter(self.info["id_lamp"])
        self.filters["band"] = Filter(self.info["id_band"])
        self.shared += ["band"]

    def add_header_info(self, header, channel, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        setting, detector = self.parse_channel(channel)
        header = super().add_header_info(header, setting)
        self.load_info()

        return header

    def get_supported_channels(self):
        settings = self.info["settings"]
        detectors = self.info["chips"]
        channels = [f"{s}_{c}" for s, c in product(settings, detectors)]
        return channels

    def parse_channel(self, channel):
        pattern = r"([YJHKLM]\d{4})_det(\d)"
        match = re.match(pattern, channel, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid channel format: {channel}")
        setting = match.group(1).upper()
        detector = match.group(2)
        return setting, detector

    def get_expected_values(self, target, night, channel):
        expectations = super().get_expected_values(target, night)
        setting, detector = self.parse_channel(channel)

        for key in expectations.keys():
            if key == "bias":
                continue
            expectations[key]["band"] = setting

        return expectations

    def get_extension(self, header, channel):
        setting, detector = self.parse_channel(channel)
        extension = int(detector)
        return extension

    def get_wavecal_filename(self, header, channel, **kwargs):
        """Get the filename of the wavelength calibration config file"""
        cwd = os.path.dirname(__file__)
        fname = f"{self.name}_{channel}.npz"
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname

    def get_mask_filename(self, channel, **kwargs):
        i = self.name.lower()
        setting, detector = self.parse_channel(channel)

        fname = f"mask_{i}_det{detector}.fits.gz"
        cwd = os.path.dirname(__file__)
        fname = os.path.join(cwd, "..", "masks", fname)
        return fname

    def get_wavelength_range(self, header, channel, **kwargs):
        wmin = [header["ESO INS WLEN MIN%i" % i] for i in range(1, 11)]
        wmax = [header["ESO INS WLEN MAX%i" % i] for i in range(1, 11)]

        wavelength_range = np.array([wmin, wmax]).T
        # Invert the order numbering
        wavelength_range = wavelength_range[::-1]
        # Convert from nm to Angstrom
        wavelength_range *= 10
        return wavelength_range
