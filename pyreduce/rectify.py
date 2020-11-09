import numpy as np
from tqdm import tqdm

from .extract import correct_for_curvature, fix_parameters
from . import util


def rectify_image(
    img, orders, column_range, extraction_width, order_range, tilt=None, shear=None
):
    nord, _ = orders.shape
    nrow, ncol = img.shape
    x = np.arange(ncol)

    extraction_width, column_range, orders = fix_parameters(
        extraction_width, column_range, orders, nrow, ncol, nord
    )

    nord = order_range[1] - order_range[0]
    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]
    extraction_width = extraction_width[order_range[0] : order_range[1]]

    images = {}
    for i in tqdm(range(nord), desc="Order"):
        x_left_lim = column_range[i, 0]
        x_right_lim = column_range[i, 1]

        # Rectify the image, i.e. remove the shape of the order
        # Then the center of the order is within one pixel variations
        ycen = np.polyval(orders[i], x).astype(int)
        yb, yt = ycen - extraction_width[i, 0], ycen + extraction_width[i, 1]
        height = extraction_width[i, 0] + extraction_width[i, 1] + 1
        index = util.make_index(yb, yt, x_left_lim, x_right_lim)
        img_order = img[index]

        # Correct for tilt and shear
        # For each row of the rectified order, interpolate onto the shifted row
        # Masked pixels are set to 0, similar to the summation
        if tilt is not None and shear is not None:
            img_order = correct_for_curvature(
                img_order,
                tilt[i, x_left_lim:x_right_lim],
                shear[i, x_left_lim:x_right_lim],
                extraction_width[i],
            )
        images[order_range[0] + i] = img_order

    return images
