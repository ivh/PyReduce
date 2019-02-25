"""
Wrapper for REDUCE C functions
   locate_cluster
"""

import logging

import numpy as np
import matplotlib.pyplot as plt

try:
    from .clib._slitfunc_bd import lib as slitfunclib
    from .clib._slitfunc_2d import lib as slitfunc_2dlib

    # from .clib._cluster import lib as clusterlib
    from .clib._slitfunc_bd import ffi
except ImportError:
    raise ImportError("Use setup.py to compile the C libraries")


c_double = np.ctypeslib.ctypes.c_double
c_int = np.ctypeslib.ctypes.c_int


# def find_clusters(img, min_cluster=4, filter_size=10, noise=1.0):
#     """Wrapper for locate_clusters and cluster, which find clustered pixels

#     Parameters
#     ----------
#     img : array[nrow, ncol]
#         order definition image
#     min_cluster : int, optional
#         minimum size of cluster (default: 4)
#     filter_size : int, optional
#         size of the mean(?) filter (default: 10)
#     noise : float, optional
#         how much noise to filter out (default: 1.0)

#     Returns
#     -------
#     y : array[n](int)
#         y coordinates of pixels in clusters
#     x : array[n](int)
#         x coordinates of pixels in clusters
#     clusters : array[n](int)
#         cluster id of pixels in clusters
#     nclusters : int
#         number of clusters
#     """

#     nY, nX = img.shape
#     nmax = nY * nX - np.ma.count_masked(img)

#     min_cluster = int(min_cluster)
#     filter_size = int(filter_size)
#     noise = float(noise)

#     mask = ~np.ma.getmaskarray(img).astype(int)
#     mask = np.require(mask, dtype=c_int, requirements=["C", "A", "W", "O"])

#     img = np.ma.getdata(img)
#     img = np.require(img, dtype=c_int, requirements=["C", "A", "W", "O"])

#     x = np.zeros(nmax, dtype=c_int)
#     y = np.zeros(nmax, dtype=c_int)

#     # Find all pixels above the threshold
#     n = clusterlib.locate_clusters(
#         ffi.cast("int", nX),
#         ffi.cast("int", nY),
#         ffi.cast("int", filter_size),
#         ffi.cast("int *", img.ctypes.data),
#         ffi.cast("int", nmax),
#         ffi.cast("int *", x.ctypes.data),
#         ffi.cast("int *", y.ctypes.data),
#         ffi.cast("float", noise),
#         ffi.cast("int *", mask.ctypes.data),
#     )

#     # remove unnecessary memory
#     x = x[:n]
#     y = y[:n]

#     # Not sure its necessay but the numbering is nicer if we do this
#     sort = np.argsort(y)
#     y = np.require(y[sort], dtype=c_int, requirements=["C", "A", "W", "O"])
#     x = np.require(x[sort], dtype=c_int, requirements=["C", "A", "W", "O"])

#     clusters = np.zeros(n, dtype=c_int)

#     # Group the pixels into clusters
#     nclus = clusterlib.cluster(
#         ffi.cast("int *", x.ctypes.data),
#         ffi.cast("int *", y.ctypes.data),
#         ffi.cast("int", n),
#         ffi.cast("int", nX),
#         ffi.cast("int", nY),
#         ffi.cast("int", min_cluster),
#         ffi.cast("int *", clusters.ctypes.data),
#     )

#     # transpose output
#     return y, x, clusters, nclus


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

    # Get dimensions
    lambda_sf = float(lambda_sf)
    lambda_sp = float(lambda_sp)
    osample = int(osample)

    img = np.asanyarray(img)
    ycen = np.asanyarray(ycen)

    if not np.issubdtype(img.dtype, np.number):
        raise TypeError(
            "Input image must be a numeric type, but got %s" % str(img.dtype)
        )

    if not np.issubdtype(ycen.dtype, np.number):
        raise TypeError("Ycen must be a numeric type, but got %s" % str(ycen.dtype))

    if img.shape[0] != ycen.size:
        raise ValueError(
            "Image and Ycen shapes are incompatible, got %s and %s"
            % (img.shape, ycen.shape)
        )

    if osample <= 0:
        raise ValueError("Oversample rate must be positive, but got %i" % osample)
    if lambda_sf < 0:
        raise ValueError(
            "Slitfunction smoothing must be positive, but got %f" % lambda_sf
        )
    if lambda_sp < 0:
        raise ValueError("Spectrum smoothing must be positive, but got %f" % lambda_sp)

    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    # Inital guess for slit function and spectrum
    sp = np.ma.sum(img, axis=0)
    sp = np.require(sp, dtype=c_double, requirements=["C", "A", "W", "O"])

    sl = np.zeros(ny, dtype=c_double)

    mask = ~np.ma.getmaskarray(img)
    mask = np.require(mask, dtype=c_int, requirements=["C", "A", "W", "O"])

    img = np.ma.getdata(img)
    img = np.require(img, dtype=c_double, requirements=["C", "A", "W", "O"])

    pix_unc = np.zeros_like(img)
    pix_unc = np.require(pix_unc, dtype=c_double, requirements=["C", "A", "W", "O"])

    ycen = np.require(ycen, dtype=c_double, requirements=["C", "A", "W", "O"])
    model = np.zeros((nrows, ncols), dtype=c_double)
    unc = np.zeros(ncols, dtype=c_double)

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


def slitfunc_curved(img, ycen, shear, lambda_sp=0, lambda_sf=0.1, osample=1):
    """Decompose an image into a spectrum and a slitfunction, image may be curved

    Parameters
    ----------
    img : array[n, m]
        input image
    ycen : array[n]
        traces the center of the order
    shear : array[n]
        tilt of the order along the image ???, set to 0 if order straight
    osample : int, optional
        Subpixel ovsersampling factor (the default is 1, which no oversampling)
    lambda_sp : float, optional
        smoothing factor spectrum (the default is 0, which no smoothing)
    lambda_sl : float, optional
        smoothing factor slitfunction (the default is 0.1, which small)

    Returns
    -------
    sp, sl, model, unc
        spectrum, slitfunction, model, spectrum uncertainties
    """

    lambda_sf = float(lambda_sf)
    lambda_sp = float(lambda_sp)
    osample = int(osample)

    img = np.asanyarray(img)
    ycen = np.asanyarray(ycen)
    shear = np.asanyarray(shear)
    if np.isscalar(shear):
        shear = np.full(img.shape[0], shear, dtype=c_double)

    if not np.issubdtype(img.dtype, np.number):
        raise TypeError(
            "Input image must be a numeric type, but got %s" % str(img.dtype)
        )
    if not np.issubdtype(ycen.dtype, np.number):
        raise TypeError("Ycen must be a numeric type, but got %s" % str(ycen.dtype))
    if not np.issubdtype(shear.dtype, np.number):
        raise TypeError("Shear must be a numeric type, but got %s" % str(shear.dtype))

    if img.shape[0] != ycen.size:
        raise ValueError(
            "Image and Ycen shapes are incompatible, got %s and %s"
            % (img.shape, ycen.shape)
        )
    if img.shape[0] != shear.size:
        raise ValueError(
            "Image and Shear shapes are incompatible, got %s and %s"
            % (img.shape, shear.shape)
        )

    if osample <= 0:
        raise ValueError("Oversample rate must be positive, but got %i" % osample)
    if lambda_sf < 0:
        raise ValueError(
            "Slitfunction smoothing must be positive, but got %f" % lambda_sf
        )
    if lambda_sp < 0:
        raise ValueError("Spectrum smoothing must be positive, but got %f" % lambda_sp)

    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    y_lower_lim = nrows // 2 - np.min(ycen).astype(int)
    y_lower_lim = int(y_lower_lim)

    sl = np.zeros(ny, dtype=c_double)

    # Inital guess for spectrum
    sp = np.sum(img, axis=0)
    sp = np.require(sp, dtype=c_double, requirements=["C", "A", "W", "O"])

    mask = ~np.ma.getmaskarray(img)
    mask = np.require(mask, dtype=c_int, requirements=["C", "A", "W", "O"])

    img = np.ma.getdata(img)
    img = np.require(img, dtype=c_double, requirements=["C", "A", "W", "O"])

    pix_unc = np.zeros_like(img)
    pix_unc = np.require(pix_unc, dtype=c_double, requirements=["C", "A", "W", "O"])

    ycen = np.require(ycen, dtype=c_double, requirements=["C", "A", "W", "O"])
    ycen_offset = np.require(ycen, dtype=c_int, requirements=["C", "A", "W", "O"])

    shear = np.require(shear, dtype=c_double, requirements=["C", "A", "W", "O"])

    model = np.zeros((nrows, ncols), dtype=c_double)
    unc = np.zeros(ncols, dtype=c_double)

    slitfunc_2dlib.slit_func_curved(
        ffi.cast("int", ncols),
        ffi.cast("int", nrows),
        ffi.cast("double *", img.ctypes.data),
        ffi.cast("double *", pix_unc.ctypes.data),
        ffi.cast("int *", mask.ctypes.data),
        ffi.cast("double *", ycen.ctypes.data),
        ffi.cast("int *", ycen_offset.ctypes.data),
        ffi.cast("double *", shear.ctypes.data),
        ffi.cast("int", y_lower_lim),
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
