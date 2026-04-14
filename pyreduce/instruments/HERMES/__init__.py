"""
Handles instrument specific info for the HERMES spectrograph

High Efficiency and Resolution Mercator Echelle Spectrograph
at the Mercator telescope, La Palma.
"""

import logging

from ..common import Instrument

logger = logging.getLogger(__name__)


class HERMES(Instrument):
    def add_header_info(self, header, channel, **kwargs):
        header = super().add_header_info(header, channel)
        return header
