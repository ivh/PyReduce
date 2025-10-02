"""
Handles instrument specific info for the METIS_LSS LSS spectrograph

Mostly reading data from the header
"""

import logging
import os.path

from .common import InstrumentWithModes

logger = logging.getLogger(__name__)


class METIS_LSS(InstrumentWithModes):
    def add_header_info(self, header, mode, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, mode)

        # header["e_backg"] = (
        #     header["e_readn"] + header["e_exptime"] * header["e_drk"] / 3600
        # )
        #
        # header["e_ra"] /= 15
        # if header["e_jd"] is not None:
        #     header["e_jd"] += header["e_exptime"] / 2 / 3600 / 24 + 0.5

        return header

    def get_extension(self, header, mode):
        extension = 1

        return extension

    def get_wavecal_filename(self, header, mode, **kwargs):
        """Get the filename of the wavelength calibration config file"""
        # info = self.load_info()
        cwd = os.path.dirname(__file__)
        fname = f"metis_lss_{mode.lower()}_2D.npz"
        # fname = f"metis_lss_LSS_L_2D.npz" ## f"micado_IJ_2D_det1.npz"
        fname = os.path.join(cwd, "..", "wavecal", fname)

        return fname
