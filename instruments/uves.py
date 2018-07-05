"""
Handles instrument specific info for the UVES spectrograph

Mostly reading data from the header
"""
import numpy as np
from astropy.io import fits

from .common import instrument, getter


class UVES(instrument):
    def load_info(self):
        """ Load harcoded information about this instrument """
        info = {
            "instrument": "UVES",
            "modes": ["blue", "middle", "red"],
            "id": [[1, 5], [1, 4], [1, 4]],
            "orientation": [6, 1, 1],
            "prescan_x": "HIERARCH ESO DET OUT{id[0]} PRSCX",
            "overscan_x": "HIERARCH ESO DET OUT{id[0]} OVSCX",
            "prescan_y": 0,
            "overscan_y": 0,
            "naxis_x": "NAXIS1",
            "naxis_y": "NAXIS2",
            "gain": "HIERARCH ESO DET OUT{id[0]} CONAD",
            "readnoise": "HIERARCH ESO DET OUT{id[0]} RON",
            "dark": "HIERARCH ESO INS DET{id[1]} OFFDRK",
            "sky": "HIERARCH ESO INS DET{id[1]} OFFSKY",
            "exposure_time": "EXPTIME",
            "image_type": "OBJECT",
            "category": "HIERARCH ESO DPR CATG",
            "ra": "RA",
            "dec": "DEC",
            "jd": "MJD-OBS",
            "longitude": "HIERARCH ESO TEL GEOLON",
            "latitude": "HIERARCH ESO TEL GEOLAT",
            "altitude": "HIERARCH ESO TEL GEOELEV",
            "target": "OBJECT",
            "observation_type": "ESO DPR TYPE",
            "instrument_mode": "ESO INS MODE",
            "id_bias": "BIAS",
            "id_flat": "LAMP,FLAT",
            "id_wave": "LAMP,WAVE",
            "id_orders": "LAMP,ORDERDEF",
            "id_spec": "OBJECT,POINT",
        }
        return info

    def add_header_info(self, header, mode, *args, **kwargs):
        """ read data from header and add it as REDUCE keyword back to the header """
        header = super().add_header_info(header, mode)

        header["e_ra"] /= 15
        header["e_jd"] += header["e_exptim"] / (7200 * 24) + 0.5

        return header

    def sort_files(self, files, target, mode, *args, **kwargs):
        """
        Sort a set of fits files into different categories
        types are: bias, flat, wavecal, orderdef, spec

        Parameters
        ----------
        files : list(str)
            files to sort
        Returns
        -------
        biaslist, flatlist, wavelist, orderlist, orderdef_fiber_a, orderdef_fiber_b, speclist
            lists of files, one per type
        """
        info = self.load_info()

        # TODO is this also instrument specific? Probably
        # TODO use instrument info instead of settings for labels?
        ob = np.zeros(len(files), dtype="U20")
        ty = np.zeros(len(files), dtype="U20")
        mo = np.zeros(len(files), dtype="U20")

        for i, f in enumerate(files):
            h = fits.open(f)[0].header
            ob[i] = h[info["target"]].replace("-", "")
            ty[i] = h[info["observation_type"]]
            mo[i] = h.get(info["instrument_mode"], "")

        # TODO instrument mode check
        biaslist = files[ty == info["id_bias"]]
        flatlist = files[ty == info["id_flat"]]
        wavelist = files[ob == info["id_wave"]]
        orderlist = files[ob == info["id_orders"]]
        speclist = files[ob == target]

        return biaslist, flatlist, wavelist, orderlist, speclist
