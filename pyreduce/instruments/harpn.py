"""
Handles instrument specific info for the HARPN spectrograph

Mostly reading data from the header
"""

import logging
import re
from os.path import dirname, join

import numpy as np

from .common import Instrument
from .filters import Filter, InstrumentFilter, NightFilter, ObjectFilter

logger = logging.getLogger(__name__)


class TypeFilter(Filter):
    def __init__(self, keyword="TNG DPR TYPE"):
        super().__init__(keyword, regex=True)

    def classify(self, value):
        if value is not None:
            match = self.match(value)
            data = np.asarray(self.data)
            data = np.unique(data[match])
            try:
                regex = re.compile(value)
                keys = [regex.match(f) for f in data]
                keys = [[g for g in d.groups() if g is not None][0] for d in keys]
                unique = np.unique(keys)
                assign = {
                    u: [d for k, d in zip(keys, data, strict=False) if k == u]
                    for u in unique
                }
                data = [(u, self.match("|".join(a))) for u, a in assign.items()]
            except IndexError:
                data = np.asarray(self.data)
                data = np.unique(data[match])
                data = [(d, self.match(d)) for d in data]
        else:
            data = np.unique(self.data)
            data = [(d, self.match(d)) for d in data]
        return data


class HARPN(Instrument):
    def __init__(self):
        super().__init__()
        self.filters = {
            "instrument": InstrumentFilter(self.config.instrument),
            "night": NightFilter(self.config.date),
            # "branch": Filter(, regex=True),
            "mode": Filter(
                self.config.instrument_mode, regex=True, flags=re.IGNORECASE
            ),
            "type": TypeFilter(self.config.observation_type),
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
            "bias": {"instrument": "HARPN", "night": night, "type": r"BIAS,BIAS"},
            "flat": {"instrument": "HARPN", "night": night, "type": r"LAMP,LAMP,TUN"},
            "orders": {
                "instrument": "HARPN",
                "night": night,
                "type": id_orddef,
            },
            "scatter": {
                "instrument": "HARPN",
                "night": night,
                "type": id_orddef,  # Same as orders or same as flat?
            },
            "wavecal_master": {
                "instrument": "HARPN",
                "night": night,
                "type": r"WAVE,WAVE,THAR2",
            },
            "freq_comb_master": {
                "instrument": "HARPN",
                "night": night,
                "type": r"WAVE,WAVE,COMB",
            },
            "science": {
                "instrument": "HARPN",
                "night": night,
                "mode": mode,
                "type": id_spec,
                "target": target,
            },
        }
        return expectations

    def get_extension(self, header, channel):
        extension = super().get_extension(header, channel)

        try:
            if (
                header["NAXIS"] == 2
                and header["NAXIS1"] == 4296
                and header["NAXIS2"] == 4096
            ):
                extension = 0
        except KeyError:
            pass

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
        fname = f"harpn_{channel.lower()}_2D.npz"
        fname = join(cwd, "..", "wavecal", fname)
        return fname

    def get_wavelength_range(self, header, channel, **kwargs):
        wave_range = super().get_wavelength_range(header, channel, **kwargs)
        # The wavelength orders are in inverse order in the .json file
        # because I was to lazy to invert them in the file
        wave_range = wave_range[::-1]
        return wave_range
