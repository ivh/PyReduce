"""
Handles instrument specific info for the NEID spectrograph

NEID is a fiber-fed, high-resolution (R~110,000) spectrograph on the
WIYN 3.5m telescope at Kitt Peak. It has three fibers:
- Science fiber (SCI-OBJ): Target or calibration light
- Calibration fiber (CAL-OBJ): Simultaneous calibration (Etalon, LFC, etc.)
- Sky fiber (SKY-OBJ): Sky background or calibration

L0 data has 16 amplifiers stored in separate FITS extensions.
"""

import logging
import re
from os.path import dirname, join

from ..common import Instrument
from ..filters import Filter, InstrumentFilter, NightFilter, ObjectFilter

logger = logging.getLogger(__name__)


class NEID(Instrument):
    def __init__(self):
        super().__init__()
        self.filters = {
            "instrument": InstrumentFilter(self.config.instrument),
            "night": NightFilter(self.config.date),
            "mode": Filter(
                self.config.instrument_mode, regex=True, flags=re.IGNORECASE
            ),
            "obstype": Filter(self.config.observation_type),
            "target": ObjectFilter(self.config.target, regex=True),
            "sci_obj": Filter("SCI-OBJ", regex=True),
        }
        self.night = "night"
        self.science = "science"
        self.shared = ["instrument", "night", "mode"]
        self.find_closest = [
            "bias",
            "flat",
            "wavecal_master",
            "freq_comb_master",
            "trace",
            "scatter",
            "curvature",
        ]

    def get_expected_values(self, target, night, channel=None, mode="HR", **kwargs):
        """Determine the expected header values for file classification.

        Parameters
        ----------
        target : str
            Name of the observation target
        night : str
            Observation night(s)
        channel : str
            Instrument channel (HR mode only for now)
        mode : str
            Observation mode (HR or HE)

        Returns
        -------
        expectations : dict
            Expected header values per reduction step
        """
        if target is not None:
            target = target.replace(" ", r"(?:\s*|-)")
        else:
            target = ".*"

        expectations = {
            "bias": {
                "instrument": "NEID",
                "night": night,
                "sci_obj": "Bias",
            },
            "flat": {
                "instrument": "NEID",
                "night": night,
                "sci_obj": "Flat",
            },
            "trace": {
                "instrument": "NEID",
                "night": night,
                "sci_obj": "Flat",
            },
            "scatter": {
                "instrument": "NEID",
                "night": night,
                "sci_obj": "Flat",
            },
            "curvature": {
                "instrument": "NEID",
                "night": night,
                "sci_obj": "LFC",
            },
            "wavecal_master": {
                "instrument": "NEID",
                "night": night,
                "sci_obj": r"(ThAr|UNe).*",
            },
            "freq_comb_master": {
                "instrument": "NEID",
                "night": night,
                "sci_obj": "LFC",
            },
            "science": {
                "instrument": "NEID",
                "night": night,
                "mode": mode,
                "obstype": "Sci",
                "target": target,
            },
        }
        return expectations

    def get_wavecal_filename(self, header, channel, **kwargs):
        """Get the filename of the wavelength calibration config file."""
        cwd = dirname(__file__)
        fname = f"wavecal_{channel.lower()}.npz"
        fname = join(cwd, fname)
        return fname

    def get_wavelength_range(self, header, channel, **kwargs):
        """NEID covers ~380-930 nm."""
        return [[3800, 9300]]
