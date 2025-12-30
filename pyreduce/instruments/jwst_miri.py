"""
Handles instrument specific info for the HARPS spectrograph

Mostly reading data from the header
"""

import logging
import os.path

from .common import Instrument

logger = logging.getLogger(__name__)


class JWST_MIRI(Instrument):
    def add_header_info(self, header, channel, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, channel)
        self.load_info()
        return header

    def get_wavecal_filename(self, header, channel, **kwargs):
        """Get the filename of the wavelength calibration config file"""
        cwd = os.path.dirname(__file__)
        fname = "{instrument}_{channel}_2D.npz".format(
            instrument="harps", channel=channel
        )
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname
