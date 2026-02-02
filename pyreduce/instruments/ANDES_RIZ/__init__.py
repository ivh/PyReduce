"""ANDES R-band (RIZ) instrument."""

import os

import numpy as np

from ..common import Instrument


class ANDES_RIZ(Instrument):
    """ANDES R-band spectrograph (E2E simulation data)."""

    def get_wavecal_filename(self, header, channel, **kwargs):
        """Get the filename of the wavelength calibration config file."""
        cwd = os.path.dirname(__file__)
        fname = f"wavecal_{channel.lower()}_HDF.npz"
        return os.path.join(cwd, fname)

    def get_wavelength_range(self, header, channel, **kwargs):
        """Get wavelength range from header WL_MIN/WL_MAX (in Angstrom)."""
        # Use header values from E2E simulation (in nm, convert to Angstrom)
        wl_min = header.get("WL_MIN")
        wl_max = header.get("WL_MAX")
        if wl_min is not None and wl_max is not None:
            # Convert nm to Angstrom, return same range for all orders
            return [[wl_min * 10, wl_max * 10]]
        return None
