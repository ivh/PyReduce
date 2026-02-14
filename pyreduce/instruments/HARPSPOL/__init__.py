"""
Handles instrument specific info for the HARPSpol (HARPS polarimeter) spectrograph.

HARPSpol uses a Wollaston prism that splits each echelle order into two beams
on the detector. The two beams are treated as fiber groups "upper" and "lower"
using PyReduce's fibers_per_order auto-pairing mode.
"""

import logging
from os.path import dirname, join

from ..common import Instrument
from ..filters import Filter, InstrumentFilter, NightFilter, ObjectFilter

logger = logging.getLogger(__name__)


class HARPSPOL(Instrument):
    def __init__(self):
        super().__init__()
        self.filters = {
            "instrument": InstrumentFilter(self.config.instrument),
            "night": NightFilter(self.config.date),
            "type": Filter(self.config.observation_type, regex=True),
            "target": ObjectFilter(self.config.target, regex=True),
        }
        self.night = "night"
        self.science = "science"
        # No mode/fiber/polarization filtering — all files in the dataset are pol-mode
        self.shared = ["instrument", "night"]
        self.find_closest = [
            "bias",
            "flat",
            "wavecal_master",
            "trace",
        ]

    def get_expected_values(
        self,
        target,
        night,
        channel=None,
        **kwargs,
    ):
        if target is not None:
            target = target.replace(" ", r"(?:\s*|-)")
        else:
            target = ".*"

        expectations = {
            "bias": {"instrument": "HARPS", "night": night, "type": r"BIAS,BIAS"},
            "flat": {
                "instrument": "HARPS",
                "night": night,
                "type": r"(LAMP,LAMP),TUN",
            },
            "trace": {
                "instrument": "HARPS",
                "night": night,
                "type": r"(LAMP,LAMP),TUN",
            },
            "scatter": {
                "instrument": "HARPS",
                "night": night,
                "type": r"(LAMP,LAMP),TUN",
            },
            "curvature": {
                "instrument": "HARPS",
                "night": night,
                "type": r"(WAVE,WAVE,THAR)\d?",
            },
            "wavecal_master": {
                "instrument": "HARPS",
                "night": night,
                "type": r"(WAVE,WAVE,THAR)\d?",
            },
            "science": {
                "instrument": "HARPS",
                "night": night,
                "type": r"(STAR,(?:LIN|CIR)POL),.*",
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
        """Read data from header and add it as REDUCE keyword back to the header."""
        header = super().add_header_info(header, channel)

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
        except Exception:
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
        """Get the filename of the wavelength calibration config file.

        Uses the HARPS non-pol NPZ files, since both beams see the same
        spectral orders and the non-pol linelist has N orders matching
        per-group trace counts.
        """
        harps_dir = join(dirname(dirname(__file__)), "HARPS")
        fname = f"wavecal_{channel.lower()}_2D.npz"
        fname = join(harps_dir, fname)
        return fname

    def get_mask_filename(self, channel, **kwargs):
        """Delegate to HARPS mask files."""
        harps_dir = join(dirname(dirname(__file__)), "HARPS")
        c = channel.lower() if channel else ""
        fname = f"mask_{c}.fits.gz" if c else "mask.fits.gz"
        return join(harps_dir, fname)

    def get_wavelength_range(self, header, channel, **kwargs):
        wave_range = super().get_wavelength_range(header, channel, **kwargs)
        # Reverse order (same as HARPS)
        wave_range = wave_range[::-1]
        return wave_range
