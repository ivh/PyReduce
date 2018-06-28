import json
import numpy as np


def find_first_index(arr, value):
    """ find the first element equal to value in the array arr """
    try:
        return next(i for i, v in enumerate(arr) if v == value)
    except StopIteration:
        raise Exception("mode info not found")


class getter:
    def __init__(self, header, info, mode):
        self.header = header
        self.info = info
        self.index = find_first_index(info["modes"], mode)
        self.id = info["id"][self.index]

    def __call__(self, key, alt=None):
        return self.get(key, alt)

    def get(self, key, alt=None):
        value = self.info[key]
        if isinstance(value, list):
            value = value[self.index]
        if isinstance(value, str):
            value = value.format(id=self.id)
            value = self.header.get(value, alt)
        return value


def modeinfo(header, instrument, mode):
    fname = "info_%s.json" % instrument.lower()

    with open(fname) as file:
        info = json.load(file)

    get = getter(header, info, mode)

    header["e_orient"] = get("orientation")

    naxis_x = get("naxis_x")
    naxis_y = get("naxis_y")

    prescan_x = get("prescan_x")
    overscan_x = get("overscan_x")
    prescan_y = get("prescan_y")
    overscan_y = get("overscan_y")

    header["e_xlo"] = prescan_x
    header["e_xhi"] = naxis_x - overscan_x

    header["e_ylo"] = prescan_y
    header["e_yhi"] = naxis_y - overscan_y

    header["e_gain"] = get("gain")
    header["e_readn"] = get("readnoise")
    header["e_exptim"] = get("exposure_time")

    header["e_sky"] = get("sky", 0)
    header["e_drk"] = get("dark", 0)
    header["e_backg"] = header["e_gain"] * (header["e_drk"] + header["e_sky"])

    header["e_imtype"] = get("image_type")
    header["e_ctg"] = get("category")

    header["e_ra"] = get("ra") / 15
    header["e_dec"] = get("dec")
    header["e_jd"] = get("jd") + get("exposure_time") / (7200 * 24) + 0.5

    header["e_obslon"] = get("longitude")
    header["e_obslat"] = get("latitude")
    header["e_obsalt"] = get("altitude")

    return header
