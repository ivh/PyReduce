"""
Handles instrument specific info for the HARPS spectrograph

Mostly reading data from the header
"""

import logging
import os.path

from astropy import units as q
from astropy.io import fits
from astropy.time import Time

from .common import Instrument

logger = logging.getLogger(__name__)


class JWST_NIRISS(Instrument):
    def add_header_info(self, header, channel, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        header = super().add_header_info(header, channel)
        self.load_info()

        # TODO: this references some files, I dont know where they should be
        header["e_gain"] = 1.61
        header["e_readn"] = 18.32
        header["e_dark"] = 0.0257

        # total exposure time
        header["exptime"] = header.get("TFRAME", 0)

        return header

    def split_observation(self, fname, channel):
        hdu = fits.open(fname)
        dirname = os.path.dirname(fname)
        fname = os.path.basename(fname)

        header = hdu[0].header
        if "mjd-obs" not in header:
            if len(header["DATE-OBS"]) <= 10:
                time = header["DATE-OBS"] + "T" + header["TIME-OBS"]
            else:
                time = header["DATE-OBS"]
            header["MJD-OBS"] = Time(time).mjd

        header2 = fits.Header()
        header2["EXTNAME"] = "SCI"
        shape = hdu["SCI"].data.shape

        nframes = shape[0]
        ngroups = shape[1]

        if nframes == 1 and ngroups == 1:
            return [os.path.join(dirname, fname)]

        data = hdu["SCI"].data.reshape((-1, *shape[-2:]))
        bias = data[0]
        primary = fits.PrimaryHDU(header=header)
        files = []
        os.makedirs(os.path.join(dirname, "split"), exist_ok=True)
        for i in range(1, nframes * ngroups):
            data[i] - bias
            bias = data[i]
            fname_this = os.path.join(dirname, "split", f"pyreduce_{i}_{fname}")

            header["MJD-OBS"] += header["TFRAME"] * q.s.to(q.day)
            secondary = fits.ImageHDU(data=data, header=header2)
            hdu_this = fits.HDUList([primary, secondary])
            hdu_this.writeto(fname_this, overwrite=True)
            files.append(fname_this)
        hdu.close()
        return files

    def sort_files(
        self, input_dir, target, night, channel, allow_calibration_only=False
    ):
        files = super().sort_files(
            input_dir,
            target,
            night,
            channel,
            allow_calibration_only=allow_calibration_only,
        )
        for i, (_k, file) in enumerate(files):
            files_split = []
            for f in file["science"]:
                files_split += self.split_observation(f, channel)
            files[i][1]["science"] = files_split
        return files

    def get_wavecal_filename(self, header, channel, **kwargs):
        """Get the filename of the wavelength calibration config file"""
        cwd = os.path.dirname(__file__)
        fname = "{instrument}_{channel}_2D.npz".format(
            instrument="harps", channel=channel
        )
        fname = os.path.join(cwd, "..", "wavecal", fname)
        return fname
