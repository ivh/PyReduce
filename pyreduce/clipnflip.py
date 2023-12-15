# -*- coding: utf-8 -*-
"""
Module that
    - Clips remove pre- and overscan regions
    - Flips orients the image so that orders are roughly horizontal
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def clipnflip(
    image, header, xrange=None, yrange=None, orientation=None, transpose=None
):
    """
    Process an image and associated FITS header already in memory as follows:
    1. Trim image to desired subregion: newimage = image(xlo:xhi,ylo:yhi)
    2. Transform to standard orientation (red at top, orders run left to right)

    Parameters
    -----------
    image : array[nrow, ncol]
        raw image to be processed.
    header : fits.header, dict
        FITS header of the image
    xrange : tuple(int, int), optional
        column - range to keep in the image (default: data from header/instrument)
    yrange : tuple(int, int), optional
        row - range to keep in the image (default: data from header/instrument)
    orientation : int, optional
        number of counterclockwise 90 degrees rotation to apply to the image (default: data from header/instrument)
    transpose : bool, optional
        if True the image will be transposed before rotation (default: data from header/instrument)

    Returns
    -------
    image : array[yrange, xrange]
        clipped and flipped image

    Raises
    -------
    NotImplementedError
        nonlinear images are not supported yet
    """

    # This part depends on how many amplifiers were used for the readout
    n_amp = header.get("e_ampl", 1)
    image = np.asarray(image)

    if n_amp > 1:  # pragma: no cover
        raise NotImplementedError
        xlo = np.array(header["e_xlo*"].values())
        xhi = np.array(header["e_xhi*"].values())
        ylo = np.array(header["e_ylo*"].values())
        yhi = np.array(header["e_yhi*"].values())

        # Make sure trim region is a subset of actual image.
        sz = image.shape
        if (
            np.any(xlo < 0)
            | np.any(xlo >= sz[1])
            | np.any(ylo < 0)
            | np.any(ylo >= sz[0])
            | np.any(xhi < 0)
            | np.any(xhi > sz[1])
            | np.any(yhi < 0)
            | np.any(yhi > sz[0])
        ):
            raise ValueError("Error specifying trim region")

        linear = header.get("e_linear", False)
        if not linear:
            # TODO
            raise NotImplementedError("only linear for now")
            # pref = header["e_prefmo"]
            # image = call_function("nonlinear_" + pref, image, header)

            # i = np.where(header["e_linear"] >= 0)
            # if len(i) > 0:
            #     header = header[0 : i - 1 + 1] + header[i + 1 :]
            # header["e_linear"] = ("t", "image corrected of non-linearity")
            # ii = np.where(header["e_gain*"] >= 0)
            # if len(ii) > 0:
            #     for i in np.arange(0, len(ii) - 1 + 1, 1):
            #         k = ii[i]
            #         header = [header[0 : k - 1 + 1], header[k + 1 :]]
            # header["e_gain"] = (1, "image was converted to e-")

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
                    "The two ccd sections are aligned neither in x nor in y"
                )
        elif n_amp == 4:
            raise NotImplementedError("4-amplifier section is not implemented yet")
    else:
        xlo, xhi = xrange if xrange is not None else (header["e_xlo"], header["e_xhi"])
        ylo, yhi = yrange if yrange is not None else (header["e_ylo"], header["e_yhi"])

        # Make sure trim region is a subset of actual image.
        sz = image.shape[-2:]
        if not (0 <= xlo < xhi <= sz[1] and 0 <= ylo < yhi <= sz[0]):
            raise IndexError(
                "Image Clipping Indices are not within the image (or in inverse order)"
            )

        # Trim image to leave only the subimage containing valid image data.
        timage = image[..., ylo:yhi, xlo:xhi]  # trimmed image

    # Flip image (if necessary) to achieve standard image orientation.
    orientation = orientation if orientation is not None else header.get("e_orient", 0)
    # As per the old IDL definition, transpose is true for orientations larger than 4
    transpose = (
        transpose
        if transpose is not None
        else header.get("e_transpose", (orientation % 8) >= 4)
    )
    if transpose:
        timage = np.transpose(timage, axes=(-1, -2))
    timage = np.rot90(timage, -1 * orientation, axes=(-2, -1))

    # TODO just sum up groups and stuff?
    while timage.ndim > 2:
        timage = np.sum(timage, axis=0)

    return timage
