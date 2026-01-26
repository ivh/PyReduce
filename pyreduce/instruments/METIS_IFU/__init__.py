"""
METIS IFU instrument - ELT/METIS integral field unit spectrograph.

L/M band IFU with ~100 wavelength settings and 4 detectors.
Channels are {wavelength}_{detector}, e.g. 4.555_det1.
"""

import logging
import os.path
import re
from glob import glob

from astropy.io import fits

from ..common import Instrument
from ..filters import Filter

logger = logging.getLogger(__name__)


class METIS_IFU(Instrument):
    def __init__(self):
        super().__init__()
        self.filters["wavelength"] = Filter(self.info["id_wavelength"])
        self.shared += ["wavelength"]

    def add_header_info(self, header, channel, **kwargs):
        """Read data from header and add it as REDUCE keyword back to the header."""
        wavelength, detector = self.parse_channel(channel)
        header = super().add_header_info(header, wavelength)
        self.load_info()
        return header

    def get_supported_channels(self):
        """Return sample channels for testing.

        Can't enumerate all ~400 channels; actual channels are discovered
        dynamically via discover_channels().
        """
        return ["4.555_det1", "4.555_det2", "4.555_det3", "4.555_det4"]

    def discover_channels(self, input_dir):
        """Discover available channels from METIS IFU files.

        Extracts wavelength setting from headers and combines with detector
        numbers from extension names to form channel identifiers.
        """
        channels = set()
        files = glob(os.path.join(input_dir, "*.fits"))
        for f in files:
            try:
                with fits.open(f) as hdul:
                    wlen_cen = hdul[0].header.get("ESO INS WLEN CEN")
                    if wlen_cen is None:
                        continue
                    for hdu in hdul[1:]:
                        name = hdu.name
                        if name.startswith("DET") and ".DATA" in name:
                            det_num = name[3]  # "DET1.DATA" -> "1"
                            channels.add(f"{wlen_cen}_det{det_num}")
            except Exception:
                continue
        return sorted(channels) if channels else [None]

    def parse_channel(self, channel):
        """Parse channel string into wavelength and detector.

        Parameters
        ----------
        channel : str
            Channel identifier, e.g. "4.555_det1"

        Returns
        -------
        wavelength : str
            Wavelength setting, e.g. "4.555"
        detector : str
            Detector number, e.g. "1"
        """
        pattern = r"([\d.]+)_det(\d)"
        match = re.match(pattern, channel, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid channel format: {channel}")
        wavelength = match.group(1)
        detector = match.group(2)
        return wavelength, detector

    def get_expected_values(self, target, night, channel):
        """Get expected header values for file classification."""
        expectations = super().get_expected_values(target, night)
        wavelength, detector = self.parse_channel(channel)

        for key in expectations.keys():
            if key == "bias":
                continue
            expectations[key]["wavelength"] = float(wavelength)

        return expectations

    def get_extension(self, header, channel):
        """Get FITS extension for the given channel."""
        wavelength, detector = self.parse_channel(channel)
        return f"DET{detector}.DATA"

    def get_wavecal_filename(self, header, channel, **kwargs):
        """Get the filename of the wavelength calibration config file."""
        cwd = os.path.dirname(__file__)
        fname = f"wavecal_{channel}.npz"
        fname = os.path.join(cwd, fname)
        return fname

    def get_mask_filename(self, channel, **kwargs):
        """Get bad pixel mask filename (per detector)."""
        wavelength, detector = self.parse_channel(channel)
        fname = f"mask_det{detector}.fits.gz"
        cwd = os.path.dirname(__file__)
        fname = os.path.join(cwd, fname)
        return fname
