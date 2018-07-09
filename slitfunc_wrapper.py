import numpy as np
import logging
from scipy.interpolate import interp1d
from scipy.ndimage.filters import median_filter, gaussian_filter1d

# TODO DEBUG
# import clib.build_extract
# clib.build_extract.build()

import clib._slitfunc_bd.lib as slitfunclib
import clib._slitfunc_2d.lib as slitfunc_2dlib
from clib._cluster import ffi

c_double = np.ctypeslib.ctypes.c_double
c_int = np.ctypeslib.ctypes.c_int


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

    if osample != 1:
        logging.warning("WARNING: Oversampling may be wrong !!!")

    if lambda_sp != 0:
        logging.warning("THIS WILL PROBABLY NOT WORK")

    original = img

    # Get dimensions
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    # Inital guess for slit function and spectrum
    # Just sum up the image along one side and normalize
    #sl = np.sum(img, axis=1)
    #sl = sl / np.sum(sl)

    sp = np.sum(img, axis=0) / (img.size - np.ma.count_masked(img)) 
    if lambda_sp != 0:
        sp = gaussian_filter1d(sp, lambda_sp)

    # Stretch sl by oversampling factor
    #old_points = np.linspace(0, ny-1, nrows, endpoint=True)
    #sl = interp1d(old_points, sl, kind=2)(np.arange(ny))
    sl = np.zeros(ny, dtype=float)

    if hasattr(img, "mask"):
        mask = (~img.mask).astype(c_int).flatten()
        mask = np.ascontiguousarray(mask)
        cmask = ffi.cast("int *", mask.ctypes.data)
    else:
        mask = np.ones(nrows * ncols, dtype=c_int)
        cmask = ffi.cast("int *", mask.ctypes.data)

    sl = sl.astype(c_double)
    csl = ffi.cast("double *", sl.ctypes.data)

    sp = sp.astype(c_double)
    csp = ffi.cast("double *", sp.ctypes.data)

    img = img.flatten().astype(c_double)
    img = np.ascontiguousarray(img)
    cimg = ffi.cast("double *", img.ctypes.data)

    ycen = ycen.astype(c_double)
    ycen = np.ascontiguousarray(ycen)
    cycen = ffi.cast("double *", ycen.ctypes.data)

    model = np.zeros((nrows, ncols), dtype=c_double)
    cmodel = ffi.cast("double *", model.ctypes.data)

    unc = np.zeros(ncols, dtype=c_double)
    cunc = ffi.cast("double *", unc.ctypes.data)

    slitfunclib.slit_func_vert(
        ncols,
        nrows,
        cimg,
        cmask,
        cycen,
        osample,
        lambda_sp,
        lambda_sf,
        csp,
        csl,
        cmodel,
        cunc,
    )

    original.mask = ~mask.astype(bool)

    return sp, sl, model, unc


def slitfunc_curved(img, ycen, shear, osample=1, lambda_sp=0, lambda_sl=0.1):
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

    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    y_lower_lim = int(min(ycen))  # TODO

    # Inital guess for slit function and spectrum
    # Just sum up the image along one side and normalize
    # TODO: rotate image before summation?
    sl = np.sum(img, axis=1)
    sl = sl / np.sum(sl)  # slit function

    sp = np.sum(img * sl[:, None], axis=0)
    sp = sp / np.sum(sp) * np.sum(img)

    # Stretch sl by oversampling factor
    old_points = np.linspace(0, (nrows + 1) * osample, nrows, endpoint=True)
    sl = interp1d(old_points, sl, kind=2)(np.arange(ny))

    sl = sl.astype(c_double)
    csl = ffi.cast("double *", sl.ctypes.data)

    sp = sp.astype(c_double)
    csp = ffi.cast("double *", sp.ctypes.data)

    if np.ma.is_masked(img):
        mask = (~img.mask).astype(c_int).flatten()
        mask = np.ascontiguousarray(mask)
        cmask = ffi.cast("int *", mask.ctypes.data)
    else:
        mask = np.ones(nrows * ncols, dtype=c_int)
        cmask = ffi.cast("int *", mask.ctypes.data)

    img = img.flatten().astype(c_double)
    img = np.ascontiguousarray(img)
    cimg = ffi.cast("double *", img.ctypes.data)

    ycen = ycen.astype(c_double)
    cycen = ffi.cast("double *", ycen.ctypes.data)

    ycen_offset = np.copy(ycen).astype(c_int)
    cycen_offset = ffi.cast("int *", ycen_offset.ctypes.data)

    shear = shear.astype(c_double)
    cshear = ffi.cast("double *", shear.ctypes.data)

    model = np.zeros((nrows, ncols), dtype=c_double)
    cmodel = ffi.cast("double *", model.ctypes.data)

    unc = np.zeros(ncols, dtype=c_double)
    cunc = ffi.cast("double *", unc.ctypes.data)

    slitfunc_2dlib.slit_func_curved(
        ncols,
        nrows,
        cimg,
        cmask,
        cycen,
        cycen_offset,
        cshear,
        y_lower_lim,
        osample,
        lambda_sp,
        lambda_sl,
        csp,
        csl,
        cmodel,
        cunc,
    )

    return sp, sl, model, unc


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import readsav

    sav = readsav("./Test/test.dat")
    img = sav["im"]
    ycen = sav["ycen"]
    shear = np.zeros(img.shape[1])

    sp, sl, model, unc = slitfunc_curved(img, ycen, shear, osample=3)

    plt.subplot(211)
    plt.plot(sp)
    plt.title("Spectrum")

    plt.subplot(212)
    plt.plot(sl)
    plt.title("Slitfunction")
    plt.show()
