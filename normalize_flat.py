import numpy as np
import logging
import pickle

from make_scatter import make_scatter
from extract import extend_orders, fix_column_range, optimal_extraction, extract


def normalize_flat(img, head, orders, threshold=90000, column_range=None, **kwargs):
    """
    Use slit functions to normalize an echelle image of a flat field lamp.
    Inputs:
     im (array(ncol,nrow)) image from which orc and back were determined and
       from which spectrum is to be extracted.
     orc (array(orcdeg,nord)) polynomial coefficients (from FORDS) that describe
       the location of complete orders on the image.
     dxw float scalar, fractional width of the order to be normalized
    
    Outputs:
     blzcof (array(blzdeg,nord)) coefficients of (perhaps broken) polynomial
       fit to extracted spectrum of flat lamp.
    History:
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
    """

    # TODO plots

    percent_above_threshold = np.count_nonzero(img > threshold) / img.size
    if percent_above_threshold < 0.1:
        logging.warning(
            "The flat has only %.2f %% pixels with signal above %i",
            percent_above_threshold,
            threshold,
        )
        # TODO ask for confirmation

    nrow, ncol = img.shape
    nord, opower = orders.shape

    if column_range is None:
        column_range = np.tile([0, ncol], (nord, 0))

    try:
        logging.warning("Loading background from data file, change this")
        _f = open("background.dat", "rb")
        scatter, yscatter = pickle.load(_f)
    except FileNotFoundError:
        scatter, yscatter = make_scatter(
            img, orders, column_range=column_range, subtract=True, **kwargs
        )
        _f = open("background.dat", "wb")
        pickle.dump((scatter, yscatter), _f)

    im_norm, im_ordr, blaze = extract(
        img,
        head,
        orders,
        normalize=True,
        scatter=scatter,
        yscatter=yscatter,
        threshold=threshold,
        column_range=column_range,
        **kwargs
    )

    return im_norm, blaze
