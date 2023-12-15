# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from . import util
from .extract import correct_for_curvature, fix_parameters


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
        images[i] = img_order

    return images, column_range, extraction_width


def merge_images(images, wave, column_range, extraction_width):
    x_total = sum(img.shape[1] for img in images.values())
    y_max = max(*[img.shape[0] for img in images.values()])
    y_mid = y_max // 2

    combined_img = np.zeros((y_max, x_total))
    wavelength = np.zeros(x_total)

    idx = 0
    x_low = 0
    for iord0, iord1 in zip(range(len(wave) - 1), range(1, len(wave))):

        img0 = images[iord0]
        img1 = images[iord1]

        xwd0, xwd1 = extraction_width[iord0], extraction_width[iord1]
        y0_low = y_mid - xwd0[0]
        y0_high = y_mid + xwd0[1] + 1

        y1_low = y_mid - xwd1[0]
        y1_high = y_mid + xwd1[1] + 1

        # Calculate Overlap
        cr0, cr1 = column_range[iord0], column_range[iord1]
        w0 = wave[iord0][cr0[0] : cr0[1]]
        w1 = wave[iord1][cr1[0] : cr1[1]]

        i0 = np.ma.where((w0 >= np.ma.min(w1)) & (w0 <= np.ma.max(w1)))
        i1 = np.ma.where((w1 >= np.ma.min(w0)) & (w1 <= np.ma.max(w0)))

        if i0[0].size > 0 and i1[0].size > 0:
            # The non overlapping part is just the image
            x_high = i0[0].min()
            combined_img[y0_low:y0_high, idx : idx + x_high - x_low] = img0[
                :, x_low:x_high
            ]
            wavelength[idx : idx + x_high - x_low] = w0[x_low:x_high]

            # for the overlap region use a common wavelength grid
            n_points = (len(i0[0]) + len(i1[0])) // 2
            w_common = np.geomspace(w0[i0][0], w1[i1][-1], num=n_points)

            img0_common = interp1d(
                w0[i0], img0[:, i0[0]], kind="linear", fill_value="extrapolate"
            )(w_common)
            img1_common = interp1d(
                w1[i1], img1[:, i1[0]], kind="linear", fill_value="extrapolate"
            )(w_common)

            # And then simply take the average between the two
            counter_common = np.zeros((y_max, n_points), dtype=int)
            img_common = np.zeros((y_max, n_points))
            img_common[y0_low:y0_high] += img0_common
            counter_common[y0_low:y0_high] += 1
            img_common[y1_low:y1_high] += img1_common
            counter_common[y1_low:y1_high] += 1
            counter_common[counter_common == 0] = 1
            img_common /= counter_common

            combined_img[:, idx + x_low : idx + x_low + n_points] = img_common
            wavelength[idx + x_low : idx + x_low + n_points] = w_common

            idx += x_low + n_points
            x_low = i1[0].max()
        else:
            x_high = img0.shape[1]
            combined_img[y0_low:y0_high, idx : idx + x_high] = img0
            wavelength[idx : idx + x_high] = w0
            idx += x_high
            x_low = 0

    img0 = images[len(wave) - 1]
    y0 = img0.shape[0]
    y0_low = (y_max - y0) // 2
    y0_high = y0 + y0_low
    cr0 = column_range[iord0]
    w0 = wave[iord0][cr0[0] : cr0[1]]

    x_high = img0.shape[1]
    combined_img[y0_low:y0_high, idx : idx + x_high - x_low] = img0[:, x_low:x_high]
    wavelength[idx : idx + x_high - x_low] = w0[x_low:x_high]

    idx += x_high - x_low
    combined_img = combined_img[:, :idx]
    wavelength = wavelength[:idx]

    return wavelength, combined_img
