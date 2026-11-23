"""
SOXS (Son Of X-Shooter) at the ESO NTT.

Two arms with one detector each, selected by ``ESO SEQ ARM``:
VIS (CCD, 4 quasi-orders) and NIR (IR array, echelle orders 10-24).
"""

import logging
import os.path

import numpy as np

from ..common import Instrument

logger = logging.getLogger(__name__)

# Science region of the raw frame, (x0, x1, y0, y1) in raw pixels
REGIONS = {"VIS": (50, 858, 0, 4096), "NIR": (4, 2044, 4, 2044)}

# NIR detector gain (e-/ADU, commissioning report) and read noise (e-) per
# up-the-ramp sample count, both as used by soxspipe (soxs_detector_parameters.yaml)
NIR_GAIN = 2.3
NIR_RON_NDSAMPLES = np.array([2, 4, 7, 10, 39, 77, 128])
NIR_RON = np.array([17.50, 16.94, 14.62, 13.14, 8.95, 7.62, 7.55])


class SOXS(Instrument):
    def add_header_info(self, header, channel, **kwargs):
        header = super().add_header_info(header, channel)
        arm = channel.upper()

        x0, x1, y0, y1 = REGIONS[arm]
        header["e_xlo"], header["e_xhi"] = x0, x1
        header["e_ylo"], header["e_yhi"] = y0, y1

        if arm == "VIS":
            header["e_gain"] = header.get("HIERARCH ESO DET OUT1 CONAD", 1.0)
            header["e_readn"] = header.get("HIERARCH ESO DET OUT1 RON", 3.8)
        else:
            # A raw NIR frame is the mean of NDIT ramps: in ADU of that mean the
            # effective gain is NDIT * g and the read noise sqrt(NDIT) * RON
            ndit = max(int(header.get("HIERARCH ESO DET NDIT", 1) or 1), 1)
            nsamp = header.get("HIERARCH ESO DET NDSAMPLES", 128) or 128
            nsamp = float(np.clip(nsamp, NIR_RON_NDSAMPLES[0], NIR_RON_NDSAMPLES[-1]))
            ron = np.interp(np.log(nsamp), np.log(NIR_RON_NDSAMPLES), NIR_RON)
            header["e_gain"] = NIR_GAIN * ndit
            header["e_readn"] = float(ron * np.sqrt(ndit))

        header["e_backg"] = header["e_gain"] * (header["e_drk"] + header["e_sky"])
        if header["e_ra"] is not None:
            header["e_ra"] /= 15
        if header["e_jd"] is not None:
            header["e_jd"] += header["e_exptime"] / (2 * 24 * 3600)
        return header

    def get_wavecal_filename(self, header, channel, **kwargs):
        return os.path.join(self._inst_dir, f"wavecal_{channel.lower()}.npz")

    def get_mask_filename(self, channel, **kwargs):
        return os.path.join(self._inst_dir, f"mask_{channel.lower()}.fits.gz")
