"""
Handles instrument specific info for the NTE spectrograph

Mostly reading data from the header
"""

import logging
import os.path

from ..common import Instrument

logger = logging.getLogger(__name__)


class NTE(Instrument):
    def add_header_info(self, header, channel, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, channel)

        header["e_ra"] /= 15
        if header["e_jd"] is not None:
            header["e_jd"] += header["e_exptime"] / (2 * 24 * 3600)

        return header

    def get_wavelength_range(self, header, channel, **kwargs):
        return 7 * [[7000, 20_000]]
