import numpy as np


def modeinfo_uves(newhead, mode, **kwargs):
    # ===========================================================================================
    # The ESO VLT UVES spectrometer (red and blue arms)

    if mode == 'uves_blue':  # eev blue
        id1 = '1'  # gain, readn and sizes
        id2 = '5'  # drk and sky
        reorient = 6
    if mode == 'uves_middle':  # eev red
        id1 = '1'  # for the new format (2 ccds in 2 separate fits)
        # id1 = '4' ;for the old format (2 CCDs in one FITS)
        id2 = '6'
        reorient = 1
    if mode == 'uves_red':  # mit red
        id1 = '1'
        id2 = '6'
        reorient = 1
    if kwargs.get("orient") is not None:
        kwargs["orient"] = reorient
        newhead["e_orient"] = kwargs["orient"]
    if kwargs.get("xr") is None:
        presc = newhead.get('eso det out' + id1 + ' prscx', 0)
        ovrsc = newhead.get('eso det out' + id1 + ' ovscx',
                            0)  # oversan along x axis
        nxa = newhead['naxis1']  # full size of the whole frame
        # valid pixel range for a given ccd
        nxd = newhead['eso det out' + id1 + ' nx']
        newhead["e_xlo"] = presc
        newhead["e_xhi"] = nxa - ovrsc - 1
    else:
        newhead["e_xlo"] = kwargs["xr"][0]
        newhead["e_xhi"] = kwargs["xr"][1]
    if kwargs.get("yr") is None:
        ny = newhead['naxis2']
        newhead["e_ylo"] = 0
        newhead["e_yhi"] = ny - 1  # - 101
    else:
        newhead["e_ylo"] = kwargs["yr"][0]
        newhead["e_yhi"] = kwargs["yr"][1]

    if kwargs.get("gain") is not None:
        kwargs["gain"] = newhead['eso det out' + id1 + ' conad']
        newhead["e_gain"] = kwargs["gain"]

    if kwargs.get("readn") is not None:
        kwargs["readn"] = newhead['eso det out' + id1 + ' ron']
        newhead["e_readn"] = kwargs["readn"]

    if kwargs.get("backg") is not None:
        drk = newhead.get('eso ins det' + id2 + ' offdrk', 0)
        sky = newhead.get('eso ins det' + id2 + ' offsky', 0)
        kwargs["backg"] = kwargs["gain"] * \
            (drk + sky)  # convert adu to electrons
    if kwargs.get("time") is not None:
        kwargs["time"] = newhead.get('exptime', 0)
    # For object frames, prepare for heliocentric correction
    imtype = newhead['object']
    obsctg = newhead['eso dpr catg']
    if (imtype == 'object*') or (obsctg == 'science*'):
        if kwargs.get("ra2000") is not None:
            kwargs["ra2000"] = newhead['ra']
            kwargs["ra2000"] = kwargs["ra2000"] / 15.
        if kwargs.get("de2000") is not None:
            kwargs["de2000"] = newhead['dec']
        if kwargs.get("jd") is not None:
            kwargs["jd"] = newhead['mjd-obs'] + \
                kwargs["time"] / 2. / 3600. / 24. + 0.5e0
        # observatory coordinates
        if kwargs.get("obslon") is not None:
            kwargs["obslon"] = - newhead['eso tel geolon']
        if kwargs.get("obslat") is not None:
            kwargs["obslat"] = newhead['eso tel geolat']
        if kwargs.get("obsalt") is not None:
            kwargs["obsalt"] = newhead['eso tel geoelev']

    return newhead, kwargs
