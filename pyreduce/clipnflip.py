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
        raise NotImplementedError(
            "multi-amplifier frames must be assembled by Instrument.assemble_amplifiers"
        )

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
        timage = np.swapaxes(timage, -1, -2)
    timage = np.rot90(timage, -1 * orientation, axes=(-2, -1))

    # TODO just sum up groups and stuff?
    while timage.ndim > 2:
        timage = np.sum(timage, axis=0)

    return timage
