"""
MOSAIC instrument - ELT multi-object spectrograph with fiber bundles.

Channels follow the E2E ``ESO INS MODE`` header verbatim:

NIR: ``J_LR``, ``H_LR``, ``H_HR`` -- one 4096x4096 H4RG detector per mode.

VIS: ``B_LR``, ``B1_HR``, ``B2_HR``, ``R_LR``, ``R1_HR``, ``R2_HR`` -- each mode
is a single 12788x12394 image stitching four detectors in a 2x2 mosaic. The
detectors are slightly misaligned with a non-uniform gap, so each is reduced
independently. A VIS channel is therefore ``<mode>_<quadrant>`` with quadrant in
{LL, LR, UL, UR}: the mode half selects which files belong to the channel
(``kw_channel = "ESO INS MODE"``), the quadrant half selects the detector crop.
"""

import os

from ..common import Instrument

QUADRANTS = ("LL", "LR", "UL", "UR")
NIR_MODES = ("J_LR", "H_LR", "H_HR")
# VIS modes are fanned out per detector quadrant. Note the resolution token
# "LR" (low-res) collides spelling-wise with quadrant "LR" (lower-right), so a
# trailing _LR/_LL/... is only a quadrant when the prefix is a known VIS mode.
VIS_MODES = ("B_LR", "B1_HR", "B2_HR", "R_LR", "R1_HR", "R2_HR")


class MOSAIC(Instrument):
    # VIS detector quadrant crop boundaries within the 12788 (y) x 12394 (x)
    # mosaic. Identical across all VIS modes (the optical layout is the same;
    # only the dispersed wavelengths differ). Horizontal gap ~rows 6144-6644,
    # vertical gap ~cols 6144-6250.
    # TODO: re-measure from a June-2026 VIS flat once that data is available;
    #       these bounds are inherited from the earlier as-built simulation.
    QUADRANTS = {
        "LL": {"xlo": 0, "xhi": 6144, "ylo": 1780, "yhi": 6000},  # lower-left
        "LR": {"xlo": 6249, "xhi": 12393, "ylo": 1810, "yhi": 6005},  # lower-right
        "UL": {"xlo": 0, "xhi": 6144, "ylo": 6775, "yhi": 10990},  # upper-left
        "UR": {"xlo": 6249, "xhi": 12393, "ylo": 6644, "yhi": 10945},  # upper-right
    }

    # Initial wavelength guess for wavecal_init, keyed by mode (a single
    # [min, max] per mode, broadcast across orders).
    # TODO: split the HR sub-band ranges (B1/B2, R1/R2) once VIS data lands;
    #       for now each HR mode reuses its arm's full LR range.
    WAVELENGTH_RANGE = {
        "J_LR": [[9500.0, 13400.0]],
        "H_LR": [[14300.0, 18000.0]],
        "H_HR": [[14300.0, 18000.0]],
        "B_LR": [[3900.0, 5096.8]],
        "B1_HR": [[3900.0, 5096.8]],
        "B2_HR": [[3900.0, 5096.8]],
        "R_LR": [[5117.3, 6250.0]],
        "R1_HR": [[5117.3, 6250.0]],
        "R2_HR": [[5117.3, 6250.0]],
    }

    @staticmethod
    def quadrant_of(channel):
        """Return the VIS quadrant suffix (LL/LR/UL/UR), or None for NIR modes."""
        if channel is None:
            return None
        for q in QUADRANTS:
            if channel.endswith("_" + q) and channel[: -(len(q) + 1)] in VIS_MODES:
                return q
        return None

    @classmethod
    def mode_of(cls, channel):
        """Strip the VIS quadrant suffix to get the bare mode (header value)."""
        q = cls.quadrant_of(channel)
        if q is not None:
            return channel[: -(len(q) + 1)]
        return channel

    def add_header_info(self, header, channel, **kwargs):
        """Override to crop VIS channels to their detector quadrant."""
        header = super().add_header_info(header, channel, **kwargs)

        quadrant = self.quadrant_of(channel)
        if quadrant is not None:
            q = self.QUADRANTS[quadrant]
            header["e_xlo"] = q["xlo"]
            header["e_xhi"] = q["xhi"]
            header["e_ylo"] = q["ylo"]
            header["e_yhi"] = q["yhi"]

        return header

    def get_settings_fallbacks(self, channel):
        """Quadrants share one settings file per mode (settings_<mode>.json)."""
        if not channel:
            return []
        mode = self.mode_of(channel)
        return [channel, mode] if mode != channel else [channel]

    def get_wavelength_range(self, header, channel, **kwargs):
        return self.WAVELENGTH_RANGE.get(self.mode_of(channel))

    def get_wavelength_range_per_bundle(self, header, channel, **kwargs):
        """Per-bundle guess from wavelength_range_<channel>.yaml if present.

        Keyed by trace bundle id. These files were seeded once from the E2E
        WAVEMAP middle fiber (see the file header); the actual solution is still
        fit from the ThAr lines.
        """
        import yaml

        path = os.path.join(self._inst_dir, f"wavelength_range_{channel.lower()}.yaml")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = yaml.safe_load(f)
        return {int(k): list(v) for k, v in data.items()}

    def get_expected_values(self, target, night, channel=None, **kwargs):
        # Classify files by mode (the header value), not by the full channel
        # name which also carries the detector quadrant. Call super without a
        # channel so it does not inject its own (index-based) channel filter,
        # then add the mode filter ourselves.
        expectations = super().get_expected_values(
            target, night, channel=None, **kwargs
        )
        if channel is not None and self.config.kw_channel is not None:
            mode = self.mode_of(channel)
            for key in expectations:
                expectations[key]["channel"] = mode
        return expectations
