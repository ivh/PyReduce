import numpy as np
import astropy.io.fits as fits


def clipnflip(image, header, **kwargs):
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

    if image.ndim != 2:
        raise ValueError('image must be a two-dimensional array')

    # Make sure trim region is specificied by procedure or header keyword.
    # This part depends on how many amplifiers were used for the readout
    # ver = header.get("e_hvers", 0)  # TODO default version number?
    n_amp = header.get("e_ampl", 1)

    if n_amp > 1:  # more than one amplifier
        xlo = np.array(header['e_xlo*'].values())
        xhi = np.array(header['e_xhi*'].values())
        ylo = np.array(header['e_ylo*'].values())
        yhi = np.array(header['e_yhi*'].values())

        # Make sure trim region is a subset of actual image.
        sz = image.shape
        if (np.any(xlo < 0) | np.any(xlo >= sz[1]) | np.any(ylo < 0) |
                np.any(ylo >= sz[0]) | np.any(xhi < 0) | np.any(xhi > sz[1]) |
                np.any(yhi < 0) | np.any(yhi > sz[0])):
            raise ValueError('Error specifying trim region')

        linear = header.get('e_linear', False)
        if not linear:
            pref = header['e_prefmo']
            # TODO
            raise NotImplementedError("only linear for now")
            image = call_function('nonlinear_' + pref, image, header)

            i = np.where(header['e_linear'] >= 0)
            if (len(i) > 0):
                header = header[0:i - 1 + 1] + header[i + 1:]
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

        if n_amp == 2:
            # TODO this needs testing
            if xlo[0] == xlo[1]:
                xsize = xhi[0] - xlo[0]
                ysize = yhi[0] - ylo[0] + yhi[1] - ylo[1]
                timage = np.empty((xsize, ysize), dtype=image.dtype)
                ysize = yhi[0] - ylo[0]
                timage[:ysize, :xsize] = image[xlo[0]]
                timage[ysize:, :xsize] = image[xlo[1]]
            elif ylo[0] == ylo[1]:
                xsize = xhi[0] - xlo[0] + xhi[1] - xlo[1]
                ysize = yhi[0] - ylo[0]
                timage = np.empty((xsize, ysize), dtype=image.dtype)
                xsize = xhi[0] - xlo[0]
                timage[:ysize, :xsize] = image[xlo[0]]
                timage[:ysize, xsize:] = image[xlo[1]]
            else:
                raise Exception(
                    'The two ccd sections are aligned neither in x nor in y')
        elif (n_amp == 4):
            raise NotImplementedError(
                '4-amplifier section is not implemented yet')
    else:
        xlo, xhi = kwargs.get("xr", (header["e_xlo"], header["e_xhi"],))
        ylo, yhi = kwargs.get("yr", (header["e_ylo"], header["e_yhi"],))

        # Make sure trim region is a subset of actual image.
        sz = image.shape
        if sz[1] < xhi < xlo < 0 or sz[0] < yhi < ylo < 0:
            raise ValueError('Could not trim region')

        # Trim image to leave only the subimage containing valid image data.
        timage = image[ylo:yhi, xlo:xhi]  # trimmed image

    # Flip image (if necessary) to achieve standard image orientation.
    orient = kwargs.get("orient", header.get('e_orient'))
    if orient is not None:
        timage = np.rot90(timage, -orient)
    return timage
