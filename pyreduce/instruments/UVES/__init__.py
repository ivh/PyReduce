"""
Handles instrument specific info for the UVES spectrograph

Mostly reading data from the header
"""

import logging
import os.path

from ..common import Instrument

logger = logging.getLogger(__name__)


class UVES(Instrument):
    def add_header_info(self, header, channel, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, channel)

        header["e_ra"] /= 15
        if header["e_jd"] is not None:
            header["e_jd"] += header["e_exptime"] / (7200 * 24) + 0.5

        return header

    def get_wavecal_filename(self, header, channel, **kwargs):
        """Get the filename of the wavelength calibration config file"""
        info = self.info
        specifier = int(header[info["wavecal_specifier"]])

        cwd = os.path.dirname(__file__)
        fname = f"wavecal_{channel.lower()}_{specifier}nm_2D.npz"
        fname = os.path.join(cwd, fname)
        return fname

    def get_mask_filename(self, channel, **kwargs):
        c = channel.lower()
        fname = f"mask_{c}.fits.gz"
        cwd = os.path.dirname(__file__)
        fname = os.path.join(cwd, fname)
        return fname
