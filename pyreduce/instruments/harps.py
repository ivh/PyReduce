# -*- coding: utf-8 -*-
"""
Handles instrument specific info for the HARPS spectrograph

Mostly reading data from the header
"""
import logging
import re
from itertools import product
from os.path import dirname, join

import numpy as np

from .common import Instrument
from .filters import Filter, InstrumentFilter, NightFilter, ObjectFilter

logger = logging.getLogger(__name__)


class TypeFilter(Filter):
    def __init__(self, keyword="ESO DPR TYPE"):
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
                assign = {u: [d for k, d in zip(keys, data) if k == u] for u in unique}
                data = [(u, self.match("|".join(a))) for u, a in assign.items()]
            except IndexError:
                data = np.asarray(self.data)
                data = np.unique(data[match])
                data = [(d, self.match(d)) for d in data]
        else:
            data = np.unique(self.data)
            data = [(d, self.match(d)) for d in data]
        return data


class FiberFilter(Filter):
    def __init__(self, keyword="ESO DPR TYPE"):
        super().__init__(keyword, regex=True)
        self.lamp_values = ["LAMP", "STAR", "CIRPOL", "LINPOL"]

    def collect(self, header):
        value = header.get(self.keyword)
        if value is None:
            value = ""
        else:
            value = value.split(",")
            if value[0] in self.lamp_values and value[1] in self.lamp_values:
                value = "AB"
            elif value[1] in self.lamp_values:
                value = "B"
            elif value[0] in self.lamp_values:
                value = "A"
            else:
                value = ""

        self.data.append(value)
        return value


class PolarizationFilter(Filter):
    def __init__(self, keyword="ESO INS RET?? POS"):
        super().__init__(keyword, regex=True)

    def collect(self, header):
        dpr_type = header.get("ESO DPR TYPE", "")
        match = re.match(r"^.*,(CIR|LIN)POL,.*$", dpr_type)
        if match is None:
            value = "none"
        elif match.group(1) == "CIR":
            value = "circular"
        elif match.group(1) == "LIN":
            value = "linear"
        else:
            raise ValueError("Polarization not recognised")
        self.data.append(value)
        return value


class HARPS(Instrument):
    def __init__(self):
        super().__init__()
        self.filters = {
            "instrument": InstrumentFilter(self.info["instrument"]),
            "night": NightFilter(self.info["date"]),
            # "branch": Filter(, regex=True),
            "mode": Filter(
                self.info["instrument_mode"], regex=True, flags=re.IGNORECASE
            ),
            "type": TypeFilter(self.info["observation_type"]),
            "polarization": PolarizationFilter(),
            "target": ObjectFilter(self.info["target"], regex=True),
            "fiber": FiberFilter(),
        }
        self.night = "night"
        self.science = "science"
        self.shared = ["instrument", "night", "mode", "polarization", "fiber"]
        self.find_closest = [
            "bias",
            "flat",
            "wavecal_master",
            "freq_comb_master",
            "orders",
            "scatter",
            "curvature",
        ]

    def get_expected_values(self, target, night, mode, fiber, polarimetry):
        """Determine the default expected values in the headers for a given observation configuration

        Any parameter may be None, to indicate that all values are allowed

        Parameters
        ----------
        target : str
            Name of the star / observation target
        night : str
            Observation night/nights
        fiber : "A", "B", "AB"
            Which of the fibers should carry observation signal
        polarimetry : "none", "linear", "circular", bool
            Whether the instrument is used in HARPS or HARPSpol mode
            and which polarization is observed. Set to true for both kinds
            of polarisation.

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

        if fiber == "AB":
            template = r"({a},{a}),{c}"
        elif fiber == "A":
            template = r"({a},{b}),{c}"
        elif fiber == "B":
            template = r"({b},{a}),{c}"
        elif fiber is None:
            template = None
            fiber = "(AB)|(A)|(B)"
        else:
            raise ValueError(
                "fiber keyword not understood, possible values are 'AB', 'A', 'B'"
            )

        if polarimetry == "none" or not polarimetry:
            mode = "HARPS"
            if template is not None:
                id_orddef = template.format(a="LAMP", b="DARK", c=".*?")
                id_spec = template.format(a="STAR", b="(?!STAR).*?", c=".*?")
            else:
                id_spec = (
                    r"^(STAR,(?!STAR).*),.*$|^((?!STAR).*?,STAR),.*$|^(STAR,STAR),.*$"
                )
                id_orddef = r"^(LAMP,DARK),.*$|^(DARK,LAMP),.*$|^(LAMP,LAMP),.*$"
            polarimetry = "none"
        else:
            mode = "HARPSpol"
            id_orddef = r"(LAMP,LAMP),.*"
            if polarimetry == r"linear":
                id_spec = r"(STAR,LINPOL),.*"
            elif polarimetry == "circular":
                id_spec = r"(STAR,CIRPOL),.*"
            elif polarimetry:
                id_spec = r"(STAR,(?:LIN|CIR)POL),.*"
                polarimetry = r"(circular|linear)"
            else:
                raise ValueError(
                    f"polarization parameter not recognized. Expected one of 'none', 'linear', 'circular', but got {polarimetry}"
                )

        expectations = {
            "bias": {"instrument": "HARPS", "night": night, "type": r"BIAS,BIAS"},
            "flat": {"instrument": "HARPS", "night": night, "type": r"(LAMP,LAMP),.*"},
            "orders": {
                "instrument": "HARPS",
                "night": night,
                "fiber": fiber,
                "type": id_orddef,
            },
            "scatter": {
                "instrument": "HARPS",
                "night": night,
                "type": id_orddef,  # Same as orders or same as flat?
            },
            "curvature": {
                "instrument": "HARPS",
                "night": night,
                "type": [r"(WAVE,WAVE,COMB)", r"(WAVE,WAVE,THAR)\d?"],
            },
            "wavecal_master": {
                "instrument": "HARPS",
                "night": night,
                "type": r"(WAVE,WAVE,THAR)\d?",
            },
            "freq_comb_master": {
                "instrument": "HARPS",
                "night": night,
                "type": r"(WAVE,WAVE,COMB)",
            },
            "science": {
                "instrument": "HARPS",
                "night": night,
                "mode": mode,
                "type": id_spec,
                "fiber": fiber,
                "polarization": polarimetry,
                "target": target,
            },
        }
        return expectations

    def get_extension(self, header, mode):
        extension = super().get_extension(header, mode)

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

    def add_header_info(self, header, mode, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, mode)

        try:
            header["e_ra"] /= 15
            header["e_jd"] += header["e_exptim"] / (7200 * 24) + 0.5

            pol_angle = header.get("eso ins ret25 pos")
            if pol_angle is None:
                pol_angle = header.get("eso ins ret50 pos")
                if pol_angle is None:
                    pol_angle = "no polarimeter"
                else:
                    pol_angle = "lin %i" % pol_angle
            else:
                pol_angle = "cir %i" % pol_angle

            header["e_pol"] = (pol_angle, "polarization angle")
        except:
            pass

        try:
            if (
                header["NAXIS"] == 2
                and header["NAXIS1"] == 4296
                and header["NAXIS2"] == 4096
            ):
                # both modes are in the same image
                prescan_x = 50
                overscan_x = 50
                naxis_x = 2148
                if mode == "BLUE":
                    header["e_xlo"] = prescan_x
                    header["e_xhi"] = naxis_x - overscan_x
                elif mode == "RED":
                    header["e_xlo"] = naxis_x + prescan_x
                    header["e_xhi"] = 2 * naxis_x - overscan_x
        except KeyError:
            pass

        return header

    def get_wavecal_filename(self, header, mode, polarimetry, **kwargs):
        """Get the filename of the wavelength calibration config file"""
        cwd = dirname(__file__)
        if polarimetry:
            pol = "_pol"
        else:
            pol = ""
        fname = f"harps_{mode.lower()}{pol}_2D.npz"
        fname = join(cwd, "..", "wavecal", fname)
        return fname

    def get_wavelength_range(self, header, mode, **kwargs):
        wave_range = super().get_wavelength_range(header, mode, **kwargs)
        # The wavelength orders are in inverse order in the .json file
        # because I was to lazy to invert them in the file
        wave_range = wave_range[::-1]
        return wave_range
