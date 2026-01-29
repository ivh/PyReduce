"""ANDES R-band (RIZ) instrument."""

import os

from ..common import Instrument


class ANDES_RIZ(Instrument):
    """ANDES R-band spectrograph (E2E simulation data)."""

    def get_wavecal_filename(self, header, channel, **kwargs):
        """Get the filename of the wavelength calibration config file."""
        cwd = os.path.dirname(__file__)
        fname = f"wavecal_{channel.lower()}_HDF.npz"
        return os.path.join(cwd, fname)
