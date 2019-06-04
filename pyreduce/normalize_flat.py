"""
Module that normalizes the Flat field image to 1
"""

import logging

import numpy as np

from .estimate_background_scatter import estimate_background_scatter
from .extract import extract


def normalize_flat(img, orders, threshold=0.5, **kwargs):
    """
    Use slit functions to normalize an echelle image of a flat field lamp.

    Parameters
    -----------
    img : array[nrow, ncol]
        image from which the orders were determined and from which the blaze spectrum is to be extracted.
    orders : array[nord, order_degree]
        polynomial coefficients that describe the location of complete orders on the image.
    threshold : float, optional
        minimum pixel value to consider for the normalized flat field.
        If threshold <= 1, then it is used as a fraction of the maximum image value (default: 0.5)
    scatter_degree : int, optional
        degree of the background scatter fit (see estimate_background_scatter for details)
    **kwargs: dict, optional
        keywords to be passed to the extraction algorithm (see extract.extract for details)

    Returns
    ---------
    im_norm : array[nrow, ncol]
        normalized flat field image
    blaze : array[nord, ncol]
        blaze function for each order
    """
    """
    History:
    ---------
    05-Jun-1999 JAV, Adapted from getspec.pro
    26-Jan-2000 NP, removed common ham_aux, replaced with data from
               inst_setup structure available in ham.common
    09-Apr-2001 NP, added parameters COLRANGE to handle partial orders,
                OSAMPLE to control oversampling in MKSLITF, SWATH_WIDTH -
                swath width used to determine the slit function in MKSLITF
                SF_SMOOTH - to control the smoothness of the slit function
                in MKSLITF. Also added the logic to handle MASK of bad pixels
                The mask is supposed to have on ly two values: 1 for good and
                0 for bad pixels. The flat field in the masked pixels is set to
                1.
    08-Feb-2008 NP, added explicit parameter for the minimum flat signal to be
                in normalization.
    25-Feb-2008 NP, Return swath boundaries if requested by the caller
    04-Mar-2008 NP, return uncertainties in the blaze functions
    00-JUL-2018 AW, port to Python
    """

    # if threshold is smaller than 1, assume percentage value is given
    if threshold <= 1:
        threshold = np.percentile(img.flatten(), threshold * 100)

    percent_above_threshold = np.count_nonzero(img > threshold) / img.size
    if percent_above_threshold < 0.1:
        logging.warning(
            "The flat has only %.2f %% pixels with signal above %i",
            percent_above_threshold,
            threshold,
        )
        # TODO ask for confirmation

    # Get background scatter
    scatter = estimate_background_scatter(img, orders, **kwargs)

    im_norm, _, blaze, _ = extract(
        img,
        orders,
        scatter=scatter,
        threshold=threshold,
        extraction_type="normalize",
        tilt=0,
        shear=0,
        **kwargs
    )

    return im_norm, blaze
