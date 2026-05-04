"""
Handles instrument specific info for the MICADO spectrograph

Mostly reading data from the header
"""

import logging
import os.path
import re

from ..common import Instrument

logger = logging.getLogger(__name__)


class MICADO(Instrument):
    def add_header_info(self, header, channel, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, channel)

        # header["e_backg"] = (
        #     header["e_readn"] + header["e_exptime"] * header["e_drk"] / 3600
        # )
        #
        # header["e_ra"] /= 15
        # if header["e_jd"] is not None:
        #     header["e_jd"] += header["e_exptime"] / 2 / 3600 / 24 + 0.5

        return header

    def parse_channel(self, channel):
        """Parse channel string into wavelength and detector.

        Parameters
        ----------
        channel : str
            Channel identifier, e.g. "SPEC_det1"

        Returns
        -------
        detector : str
            Detector number, e.g. "1"
        """
        pattern = r"SPEC_det(\d)"
        match = re.match(pattern, channel, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid channel format: {channel}")
        detector = match.group(1)
        return detector

    def get_extension(self, header, channel):
        detector = self.parse_channel(channel)
        return f"DET{detector}.IMG"

    def get_wavecal_filename(self, header, channel, **kwargs):
        """Get the filename of the wavelength calibration config file"""
        cwd = os.path.dirname(__file__)
        fname = "wavecal_HK_3arcsec_chip5.npz"
        fname = os.path.join(cwd, fname)
        return fname
