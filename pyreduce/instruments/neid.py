"""
Handles instrument specific info for the NEID spectrograph
"""

import logging
import re
from os.path import dirname, join

from .common import Instrument
from .filters import Filter, InstrumentFilter, NightFilter, ObjectFilter

logger = logging.getLogger(__name__)


class NEID(Instrument):
    def __init__(self):
        super().__init__()
        self.filters = {
            "instrument": InstrumentFilter(self.config.instrument),
            "night": NightFilter(self.config.date),
            # "branch": Filter(, regex=True),
            "mode": Filter(
                self.config.instrument_mode, regex=True, flags=re.IGNORECASE
            ),
            "type": Filter(self.config.observation_type),
            "target": ObjectFilter(self.config.target, regex=True),
        }
        self.night = "night"
        self.science = "science"
        self.shared = [
            "instrument",
            "night",
            "mode",
        ]
        self.find_closest = [
            "bias",
            "flat",
            "wavecal_master",
            "freq_comb_master",
            "orders",
            "scatter",
        ]

    def get_expected_values(
        self, target, night, channel=None, mode=None, fiber=None, **kwargs
    ):
        """Determine the default expected values in the headers for a given observation configuration

        Any parameter may be None, to indicate that all values are allowed

        Parameters
        ----------
        target : str
            Name of the star / observation target
        night : str
            Observation night/nights
        Returns
        -------
        expectations: dict
            Dictionary of expected header values, with one entry per step.
            The entries for each step refer to the filters defined in self.filters

        Raises
        ------
        ValueError
            Invalid combination of parameters
        """
        if target is not None:
            target = target.replace(" ", r"(?:\s*|-)")
        else:
            target = ".*"

        id_orddef = "LAMP,DARK,TUN"
        id_spec = "STAR,WAVE"

        expectations = {
            "flat": {"instrument": "NEID", "night": night, "type": r"LAMP,LAMP,TUN"},
            "orders": {
                "instrument": "NEID",
                "night": night,
                "type": id_orddef,
            },
            "scatter": {
                "instrument": "NEID",
                "night": night,
                "type": id_orddef,  # Same as orders or same as flat?
            },
            "wavecal_master": {
                "instrument": "NEID",
                "night": night,
                "type": r"WAVE,WAVE,THAR2",
            },
            "freq_comb_master": {
                "instrument": "NEID",
                "night": night,
                "type": r"WAVE,WAVE,COMB",
            },
            "science": {
                "instrument": "NEID",
                "night": night,
                "mode": mode,
                "type": id_spec,
                "target": target,
            },
        }
        return expectations

    def get_extension(self, header, channel):
        extension = super().get_extension(header, channel)

        return extension

    def add_header_info(self, header, channel, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, channel)

        try:
            header["e_ra"] /= 15
            header["e_jd"] += header["e_exptim"] / (7200 * 24) + 0.5

        except:
            pass

        try:
            if (
                header["NAXIS"] == 2
                and header["NAXIS1"] == 4296
                and header["NAXIS2"] == 4096
            ):
                # both channels are in the same image
                prescan_x = 50
                overscan_x = 50
                naxis_x = 2148
                if channel == "BLUE":
                    header["e_xlo"] = prescan_x
                    header["e_xhi"] = naxis_x - overscan_x
                elif channel == "RED":
                    header["e_xlo"] = naxis_x + prescan_x
                    header["e_xhi"] = 2 * naxis_x - overscan_x
        except KeyError:
            pass

        return header

    def get_wavecal_filename(self, header, channel, **kwargs):
        """Get the filename of the wavelength calibration config file"""
        cwd = dirname(__file__)
        fname = f"NEID_{channel.lower()}_2D.npz"
        fname = join(cwd, "..", "wavecal", fname)
        return fname

    def get_wavelength_range(self, header, channel, **kwargs):
        wave_range = super().get_wavelength_range(header, channel, **kwargs)
        return wave_range
