"""
MOSAIC instrument - ELT multi-object spectrograph with 630 fibers.

NIR channel: 90 fiber groups (7 fibers each), 4096x4096 H4RG detector.
VIS channel: 4 quadrants (VIS1-VIS4) from a 12788x12394 mosaic detector.
Fibers split into two halves with a gap in the middle.
"""

from ..common import Instrument


class MOSAIC(Instrument):
    # VIS detector quadrant boundaries (from gap analysis)
    # Full VIS frame: 12788 (height) x 12394 (width) pixels
    # Horizontal gap: rows 6144-6644, Vertical gap: cols 6144-6250
    # Active region: rows 2038-10734
    VIS_QUADRANTS = {
        "VIS1": {"xlo": 0, "xhi": 6144, "ylo": 2020, "yhi": 6144},  # lower-left
        "VIS2": {"xlo": 6250, "xhi": 12394, "ylo": 2038, "yhi": 6144},  # lower-right
        "VIS3": {"xlo": 0, "xhi": 6144, "ylo": 6644, "yhi": 10734},  # upper-left
        "VIS4": {"xlo": 6250, "xhi": 12394, "ylo": 6644, "yhi": 10734},  # upper-right
    }

    def add_header_info(self, header, channel, **kwargs):
        """Override to handle VIS quadrant extraction."""
        # Let parent populate all standard header info
        header = super().add_header_info(header, channel, **kwargs)

        # For VIS quadrants, override the clip bounds
        channel_upper = channel.upper()
        if channel_upper in self.VIS_QUADRANTS:
            q = self.VIS_QUADRANTS[channel_upper]
            header["e_xlo"] = q["xlo"]
            header["e_xhi"] = q["xhi"]
            header["e_ylo"] = q["ylo"]
            header["e_yhi"] = q["yhi"]

        return header
