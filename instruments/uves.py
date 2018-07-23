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

        # Tips & Tricks:
        # if several modes are supported, use a list for modes
        # if a value changes depending on the mode, use a list with the same order as "modes"
        # you can also use values from this dictionary as placeholders using {name}, just like str.format

        # red and middle are in the same fits file, with different extensions,
        # i.e. share the same mode identifier, but have different extensions
        info = {
            # General information
            "instrument": "UVES",
            "modes": ["blue", "middle", "red"],
            "modes_id": ["blue", "red", "red"],
            "extension": [0, 2, 1],
            # Header info for reduction
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
            "wavecal_specifier": "ESO INS GRAT2 WLEN",
            # Ids for file sorting
            "target": "OBJECT",
            "observation_type": "ESO DPR TYPE",
            "instrument_mode": "ESO INS MODE",
            "instrument_mode_alternative": "ESO TPL NAME",
            "id_bias": "BIAS",
            "id_flat": "LAMP,FLAT",
            "id_wave": "LAMP,WAVE",
            "id_orders": "LAMP,ORDERDEF",
            "id_spec": "OBJECT,POINT",
        }
        return info

    def add_header_info(self, header, mode, **kwargs):
        """ read data from header and add it as REDUCE keyword back to the header """
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, mode)

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

        return header

    def sort_files(self, files, target, mode, **kwargs):
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
        target = target.upper()

        # Load the mode identifier for the current mode from the header
        # This could be anything really, e.g. the size of the data axis
        i = [i for i, m in enumerate(info["modes"]) if m == mode][0]
        mode_id = info["modes_id"][i].casefold()

        ob = np.zeros(len(files), dtype="U20")
        ty = np.zeros(len(files), dtype="U20")
        mo = np.zeros(len(files), dtype="U20")

        for i, f in enumerate(files):
            h = fits.open(f)[0].header
            ob[i] = h[info["target"]]
            ty[i] = h[info["observation_type"]]
            # The mode descriptor has different names in different files, so try different ids
            mo[i] = h.get(
                info["instrument_mode"],
                h.get(info["instrument_mode_alternative"], "")[:3],
            )

            # Fix naming
            ob[i] = ob[i].replace("-", "")
            mo[i] = mo[i].casefold()

        # TODO allow several names for the target?
        biaslist = files[(ty == info["id_bias"]) & (mo == mode_id)]
        flatlist = files[(ty == info["id_flat"]) & (mo == mode_id)]
        wavelist = files[(ob == info["id_wave"]) & (mo == mode_id)]
        orderlist = files[(ob == info["id_orders"]) & (mo == mode_id)]
        speclist = files[(ty == info["id_spec"]) & (ob == target) & (mo == mode_id)]

        return biaslist, flatlist, wavelist, orderlist, speclist

    def get_wavecal_filename(self, header, mode, **kwargs):
        """ Get the filename of the wavelength calibration config file """
        info = self.load_info()
        specifier = int(header[info["wavecal_specifier"]])

        fname = "./wavecal/{instrument}_{mode}_{specifier}nm_2D.sav".format(
            instrument=info["instrument"].lower(), mode=mode, specifier=specifier
        )
        return fname
