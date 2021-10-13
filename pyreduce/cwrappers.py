# -*- coding: utf-8 -*-
"""
Wrapper for REDUCE C functions

This module provides access to the extraction algorithms in the
C libraries and sanitizes the input parameters.

"""
import ctypes
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)

try:
    from .clib._slitfunc_2d import lib as slitfunc_2dlib
    from .clib._slitfunc_bd import ffi
    from .clib._slitfunc_bd import lib as slitfunclib
except ImportError:  # pragma: no cover
    logger.error(
        "C libraries could not be found. Compiling them by running build_extract.py"
    )
    from .clib import build_extract

    build_extract.build()
    del build_extract

    from .clib._slitfunc_2d import ffi
    from .clib._slitfunc_2d import lib as slitfunc_2dlib
    from .clib._slitfunc_bd import lib as slitfunclib


c_double = ctypes.c_double
c_int = ctypes.c_int
c_mask = ctypes.c_ubyte


def slitfunc(img, ycen, lambda_sp=0, lambda_sf=0.1, osample=1):
    """Decompose image into spectrum and slitfunction

    This is for horizontal straight orders only, for curved orders use slitfunc_curved instead

    Parameters
    ----------
    img : array[n, m]
        image to decompose, should just contain a small part of the overall image
    ycen : array[n]
        traces the center of the order along the image, relative to the center of the image?
    lambda_sp : float, optional
        smoothing parameter of the spectrum (the default is 0, which no smoothing)
    lambda_sf : float, optional
        smoothing parameter of the slitfunction (the default is 0.1, which )
    osample : int, optional
        Subpixel ovsersampling factor (the default is 1, which no oversampling)

    Returns
    -------
    sp, sl, model, unc
        spectrum, slitfunction, model, spectrum uncertainties
    """

    # Convert input to expected datatypes
    lambda_sf = float(lambda_sf)
    lambda_sp = float(lambda_sp)
    osample = int(osample)
    img = np.asanyarray(img, dtype=c_double)
    ycen = np.asarray(ycen, dtype=c_double)

    assert img.ndim == 2, "Image must be 2 dimensional"
    assert ycen.ndim == 1, "Ycen must be 1 dimensional"

    assert (
        img.shape[1] == ycen.size
    ), f"Image and Ycen shapes are incompatible, got {img.shape} and {ycen.shape}"

    assert osample > 0, f"Oversample rate must be positive, but got {osample}"
    assert (
        lambda_sf >= 0
    ), f"Slitfunction smoothing must be positive, but got {lambda_sf}"
    assert lambda_sp >= 0, f"Spectrum smoothing must be positive, but got {lambda_sp}"

    # Get some derived values
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1
    ycen = ycen - ycen.astype(c_int)

    # Prepare all arrays
    # Inital guess for slit function and spectrum
    sp = np.ma.sum(img, axis=0)
    requirements = ["C", "A", "W", "O"]
    sp = np.require(sp, dtype=c_double, requirements=requirements)

    sl = np.zeros(ny, dtype=c_double)

    mask = ~np.ma.getmaskarray(img)
    mask = np.require(mask, dtype=c_int, requirements=requirements)

    img = np.ma.getdata(img)
    img = np.require(img, dtype=c_double, requirements=requirements)

    pix_unc = np.zeros_like(img)
    pix_unc = np.require(pix_unc, dtype=c_double, requirements=requirements)

    ycen = np.require(ycen, dtype=c_double, requirements=requirements)
    model = np.zeros((nrows, ncols), dtype=c_double)
    unc = np.zeros(ncols, dtype=c_double)

    # Call the C function
    slitfunclib.slit_func_vert(
        ffi.cast("int", ncols),
        ffi.cast("int", nrows),
        ffi.cast("double *", img.ctypes.data),
        ffi.cast("double *", pix_unc.ctypes.data),
        ffi.cast("int *", mask.ctypes.data),
        ffi.cast("double *", ycen.ctypes.data),
        ffi.cast("int", osample),
        ffi.cast("double", lambda_sp),
        ffi.cast("double", lambda_sf),
        ffi.cast("double *", sp.ctypes.data),
        ffi.cast("double *", sl.ctypes.data),
        ffi.cast("double *", model.ctypes.data),
        ffi.cast("double *", unc.ctypes.data),
    )
    mask = ~mask.astype(bool)

    return sp, sl, model, unc, mask


def slitfunc_curved(
    img, ycen, tilt, shear, lambda_sp, lambda_sf, osample, yrange, maxiter=20, gain=1
):
    """Decompose an image into a spectrum and a slitfunction, image may be curved

    Parameters
    ----------
    img : array[n, m]
        input image
    ycen : array[n]
        traces the center of the order
    tilt : array[n]
        tilt (1st order curvature) of the order along the image, set to 0 if order straight
    shear : array[n]
        shear (2nd order curvature) of the order along the image, set to 0 if order straight
    osample : int
        Subpixel ovsersampling factor (the default is 1, no oversampling)
    lambda_sp : float
        smoothing factor spectrum (the default is 0, no smoothing)
    lambda_sl : float
        smoothing factor slitfunction (the default is 0.1, small smoothing)
    yrange : array[2]
        number of pixels below and above the central line that have been cut out
    maxiter : int, optional
        maximumim number of iterations, by default 20
    gain : float, optional
        gain of the image, by default 1

    Returns
    -------
    sp, sl, model, unc
        spectrum, slitfunction, model, spectrum uncertainties
    """

    # Convert datatypes to expected values
    lambda_sf = float(lambda_sf)
    lambda_sp = float(lambda_sp)
    osample = int(osample)
    maxiter = int(maxiter)
    img = np.asanyarray(img, dtype=c_double)
    ycen = np.asarray(ycen, dtype=c_double)
    yrange = np.asarray(yrange, dtype=int)

    assert img.ndim == 2, "Image must be 2 dimensional"
    assert ycen.ndim == 1, "Ycen must be 1 dimensional"
    assert maxiter > 0, "Maximum iterations must be positive"

    if np.isscalar(tilt):
        tilt = np.full(img.shape[1], tilt, dtype=c_double)
    else:
        tilt = np.asarray(tilt, dtype=c_double)
    if np.isscalar(shear):
        shear = np.full(img.shape[1], shear, dtype=c_double)
    else:
        shear = np.asarray(shear, dtype=c_double)

    assert (
        img.shape[1] == ycen.size
    ), "Image and Ycen shapes are incompatible, got {} and {}".format(
        img.shape, ycen.shape
    )
    assert (
        img.shape[1] == tilt.size
    ), "Image and Tilt shapes are incompatible, got {} and {}".format(
        img.shape, tilt.shape
    )
    assert (
        img.shape[1] == shear.size
    ), "Image and Shear shapes are incompatible, got {} and {}".format(
        img.shape,
        shear.shape,
    )

    assert osample > 0, f"Oversample rate must be positive, but got {osample}"
    assert (
        lambda_sf >= 0
    ), f"Slitfunction smoothing must be positive, but got {lambda_sf}"
    assert lambda_sp >= 0, f"Spectrum smoothing must be positive, but got {lambda_sp}"

    # assert np.ma.all(np.isfinite(img)), "All values in the image must be finite"
    assert np.all(np.isfinite(ycen)), "All values in ycen must be finite"
    assert np.all(np.isfinite(tilt)), "All values in tilt must be finite"
    assert np.all(np.isfinite(shear)), "All values in shear must be finite"

    assert yrange.ndim == 1, "Yrange must be 1 dimensional"
    assert yrange.size == 2, "Yrange must have 2 elements"
    assert (
        yrange[0] + yrange[1] + 1 == img.shape[0]
    ), "Yrange must cover the whole image"
    assert yrange[0] >= 0, "Yrange must be positive"
    assert yrange[1] >= 0, "Yrange must be positive"

    # Retrieve some derived values
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    ycen_offset = ycen.astype(c_int)
    ycen_int = ycen - ycen_offset
    y_lower_lim = int(yrange[0])

    mask = np.ma.getmaskarray(img)
    img = np.ma.getdata(img)
    mask2 = ~np.isfinite(img)
    img[mask2] = 0
    mask |= ~np.isfinite(img)

    # sp should never be all zero (thats a horrible guess) and leads to all nans
    # This is a simplified run of the algorithm without oversampling or curvature
    # But strong smoothing
    # To remove the most egregious outliers, which would ruin the fit
    sp = np.sum(img, axis=0)
    median_filter(sp, 5, output=sp)
    sl = np.median(img, axis=1)
    sl /= np.sum(sl)

    model = sl[:, None] * sp[None, :]
    diff = model - img
    mask[np.abs(diff) > 10 * diff.std()] = True

    sp = np.sum(img, axis=0)

    mask = np.where(mask, c_int(0), c_int(1))
    # Determine the shot noise
    # by converting electrons to photonsm via the gain
    pix_unc = np.nan_to_num(np.abs(img), copy=False)
    pix_unc *= gain
    np.sqrt(pix_unc, out=pix_unc)
    pix_unc[pix_unc < 1] = 1

    psf_curve = np.zeros((ncols, 3), dtype=c_double)
    psf_curve[:, 1] = tilt
    psf_curve[:, 2] = shear

    # Initialize arrays and ensure the correct datatype for C
    requirements = ["C", "A", "W", "O"]
    sp = np.require(sp, dtype=c_double, requirements=requirements)
    mask = np.require(mask, dtype=c_mask, requirements=requirements)
    img = np.require(img, dtype=c_double, requirements=requirements)
    pix_unc = np.require(pix_unc, dtype=c_double, requirements=requirements)
    ycen_int = np.require(ycen_int, dtype=c_double, requirements=requirements)
    ycen_offset = np.require(ycen_offset, dtype=c_int, requirements=requirements)

    # This memory could be reused between swaths
    sl = np.zeros(ny, dtype=c_double)
    model = np.zeros((nrows, ncols), dtype=c_double)
    unc = np.zeros(ncols, dtype=c_double)

    # Info contains the folowing: sucess, cost, status, iteration, delta_x
    info = np.zeros(5, dtype=c_double)

    col = np.sum(mask, axis=0) == 0
    if np.any(col):
        mask[mask.shape[0] // 2, col] = 1
    # assert not np.any(np.sum(mask, axis=0) == 0), "At least one mask column is all 0."

    # Call the C function
    slitfunc_2dlib.slit_func_curved(
        ffi.cast("int", ncols),
        ffi.cast("int", nrows),
        ffi.cast("int", ny),
        ffi.cast("double *", img.ctypes.data),
        ffi.cast("double *", pix_unc.ctypes.data),
        ffi.cast("unsigned char *", mask.ctypes.data),
        ffi.cast("double *", ycen_int.ctypes.data),
        ffi.cast("int *", ycen_offset.ctypes.data),
        ffi.cast("int", y_lower_lim),
        ffi.cast("int", osample),
        ffi.cast("double", lambda_sp),
        ffi.cast("double", lambda_sf),
        ffi.cast("int", maxiter),
        ffi.cast("double *", psf_curve.ctypes.data),
        ffi.cast("double *", sp.ctypes.data),
        ffi.cast("double *", sl.ctypes.data),
        ffi.cast("double *", model.ctypes.data),
        ffi.cast("double *", unc.ctypes.data),
        ffi.cast("double *", info.ctypes.data),
    )

    if np.any(np.isnan(sp)):
        logger.error("NaNs in the spectrum")

    # The decomposition failed
    if info[0] == 0:
        status = info[2]
        if status == 0:
            msg = "I dont't know what happened"
        elif status == -1:
            msg = f"Did not finish convergence after maxiter ({maxiter}) iterations"
        elif status == -2:
            msg = "Curvature is larger than the swath. Check the curvature!"
        else:
            msg = f"Check the C code, for status = {status}"
        logger.error(msg)
        # raise RuntimeError(msg)

    mask = mask == 0

    return sp, sl, model, unc, mask, info
