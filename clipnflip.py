import numpy as np
import astropy.io.fits as fits


def clipnflip(image, header, xr=None, yr=None, orient=None):
    """
    # Process an image and associated FITS header already in memory as follows:
    # 1) Trim image to desired subregion: newimage = image(xlo:xhi,ylo:yhi)
    # 2) Transform to standard orientation (red at top, orders run left to right)
    # 
    # Input:
    # image (array) Raw image to be processed.
    # 
    # Input/Output:
    # [header] (string array) FITS header associated with image. Keyword arguments
    # xr=, yr=, and orient= need not be specified if header contains the
    # corresponding cards E_XLO, E_XHI, E_YLO, E_YHI, and E_ORIENT.
    # [xr=] (vector(2)) first and last column to retain in input image.
    # [yr=] (vector(2)) first and last row to retain in input image.
    # [orient=] (integer) flag indicating how trimmed image should be reoriented.
    # 
    # Output:
    # Return value of function is either the processed image (array, reoriented
    # and trimmed) or an error message (scalar string).
    # 
    # History:
    # 2000-Aug-25 Valenti  Adapted from getimage.pro.
    """

    # Check that required arguments have proper structure.
    err = None
    bad = None
    if image.ndim != 2:
        raise Exception('image must be a two-dimensional array')

    # ;Check the number of regions
    # n_regions = sxpar(header, 'E_NREG', count=True)
    # if count eq 0 then n_regions=1 else n_regions=n_regions>1

    # Make sure trim region is specificied by procedure or header keyword.
    # This part depends on how many amplifiers were used for the readout
    try:
        ver = header['e_hvers']
    except KeyError:
        ver = 0
    if ver > 1.001:  # earlier versions did not support mutiple amplifiers
        n_amp = header['e_ampl']
    else:
        n_amp = 1

    if n_amp > 1:  # more than one amplifier
        try:
            xlo = header['e_xlo*']
        except KeyError:
            bad = 'xlo'

        try:
            xhi = header['e_xhi*']
        except KeyError:
            bad = 'xhi'

        try:
            ylo = header['e_ylo*']
        except KeyError:
            bad = 'ylo'
        try:
            yhi = header['e_yhi*']
        except KeyError:
            bad = 'yhi'

        # Check we have encountered inconsistent ranges
        if bad is not None:
            print(bad + ' not specified by argument or in header')
            raise Exception('unable to trim image')

        # Make sure trim region is a subset of actual image.
        sz = image.shape
        i1 = np.where((xlo < 0) | (xlo >= sz[0]))
        i2 = np.where((ylo < 0) | (ylo >= sz[1]))
        i3 = np.where((xhi < 0) | (xhi >= sz[0]))
        i4 = np.where((yhi < 0) | (yhi >= sz[1]))
        i5 = np.where(xlo >= xhi)
        i6 = np.where(ylo >= yhi)

        err = None
        if len(i1) > 0:
            err = 'error specifying x trim region:' + ' xlo'
        if len(i2) > 0:
            err = 'error specifying y trim region:' + ' ylo'
        if len(i3) > 0:
            err = 'error specifying x trim region:' + ' xhi'
        if len(i4) > 0:
            err = 'error specifying y trim region:' + ' yhi'
        if len(i5) > 0:
            err = 'error specifying x region boundaries:' + ' xlo>=xhi'
        if len(i6) > 0:
            err = 'error specifying y region boundaries:' + ' ylo>=yhi'
        if err is not None:
            print(err)
            raise Exception('unable to trim image')

        try:
            linear = header['e_linear']
        except KeyError:
            pref = header['e_prefmo']
            # TODO
            raise NotImplementedError("only linear for now")
            image = call_function('nonlinear_' + pref, image, header)

            i = np.where(header['e_linear'] >= 0)
            if (len(i) > 0):
                header = header[0:i - 1 + 1] +  header[i + 1:]
            header['e_linear'] = ('t', 'image corrected of non-linearity')
            ii = np.where(header['e_gain*'] >= 0)
            if (len(ii) > 0):
                for i in np.arange(0, len(ii) - 1 + 1, 1):
                    k = ii[i]
                    header = [header[0:k - 1 + 1], header[k + 1:]]
            header['e_gain'] = (1, 'image was converted to e-')

        # Trim image to leave only the subimage containing valid image data.
        # For two amplifiers we assume a single vertical or horizontal gap.
        # With four amplifiers we can have a cross.

        if (n_amp == 2):
            if (xlo[0] == xlo[1]):
                xsize = xhi[0] - xlo[0] + 1
                ysize = yhi[0] - ylo[0] + 1 + yhi[1] - ylo[1] + 1
                timage = np.empty((xsize, ysize), dtype=image.dtype)
                ysize = yhi[0] - ylo[0] + 1
                timage[0:ysize - 1 + 1, 0:xsize - 1 + 1] = image[xlo[0]]
                timage[ysize:ysize + yhi[1] - ylo[1] +
                       1, 0:xsize - 1 + 1] = image[xlo[1]]
            elif (ylo[0] == ylo[1]):
                xsize = xhi[0] - xlo[0] + 1 + xhi[1] - xlo[1] + 1
                ysize = yhi[0] - ylo[0] + 1
                timage = np.empty((xsize, ysize), dtype=image.dtype)
                xsize = xhi[0] - xlo[0] + 1
                timage[0:ysize - 1 + 1, 0:xsize - 1 + 1] = image[xlo[0]]
                timage[0:ysize - 1 + 1, xsize:xsize +
                       xhi[1] - xlo[1] + 1] = image[xlo[1]]
            else:
                print('the two ccd sections are aligned neither in x nor in y')
        elif (n_amp == 4):
            print('4-amplifier section is not implemented yet')
    else:
        if xr is not None:
            xlo = xr[0]
            xhi = xr[1]
        else:
            try:
                xlo = header['e_xlo']
            except KeyError:
                bad = 'xlo'
            try:
                xhi = header['e_xhi']
            except KeyError:
                bad = 'xhi'
        if yr is not None:
            ylo = yr[0]
            yhi = yr[1]
        else:
            try:
                ylo = header['e_ylo']
            except KeyError:
                bad = 'ylo'
            try:
                yhi = header['e_yhi']
            except KeyError:
                bad = 'yhi'

        # Check we have encountered inconsistent ranges
        if bad is not None:
            print(bad + ' not specified by argument or in header')
            raise Exception('unable to trim image')

        # Make sure trim region is a subset of actual image.
        sz = image.shape
        err = None
        if xhi < xlo:
            err = 'error specifying x trim region: xhi<xlo (%i < %i)' % (
                xhi, xlo)
        if yhi < ylo:
            err = 'error specifying y trim region: yhi<ylo (%i < %i)' % (
                yhi, ylo)
        if xlo < 0 or xhi >= sz[1]:
            err = 'x trim region [%i,%i] not contained in image (0<x<%i)' % (
                xlo, xhi, sz[1] - 1)
        if ylo < 0 or yhi >= sz[0]:
            err = 'y trim region [%i,%i] not contained in image (0<y<%i)' % (
                ylo, yhi, sz[0] - 1)
        if err is not None:
            print(err)
            raise Exception('unable to trim image')

        # Trim image to leave only the subimage containing valid image data.
        timage = image[ylo:yhi + 1, xlo:xhi + 1]  # trimmed image

    # Flip image (if necessary) to achieve standard image orientation.
    if orient is not None:
        try:
            orient = header['e_orient']
        except KeyError:
            print('orient not specified by argument or in header')
            raise Exception('unable to reorient image')
        timage = np.rot90(timage, -1 * orient)
    return timage.swapaxes(0, 1)
