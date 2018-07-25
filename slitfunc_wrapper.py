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

    # Get dimensions
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    # Inital guess for slit function and spectrum
    sp = np.sum(img, axis=0) / (img.size - np.ma.count_masked(img))
    sp = np.require(sp, dtype=c_double, requirements=["C", "A", "W", "O"])
    csp = ffi.cast("double *", sp.ctypes.data)

    mask = ~np.ma.getmaskarray(img)
    mask = np.require(mask, dtype=c_int, requirements=["C", "A", "W", "O"])
    cmask = ffi.cast("int *", mask.ctypes.data)

    sl = np.zeros(ny, dtype=c_double)
    csl = ffi.cast("double *", sl.ctypes.data)

    img = np.require(np.ma.getdata(img), dtype=c_double, requirements=["C", "A", "W", "O"])
    cimg = ffi.cast("double *", img.ctypes.data)

    ycen = np.require(ycen, dtype=c_double, requirements=["C", "A", "W", "O"])
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
    mask = ~mask.astype(bool)

    return sp, sl, model, unc, mask


def slitfunc_curved(img, ycen, shear, osample=1, lambda_sp=0, lambda_sf=0.1):
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

    y_lower_lim = nrows // 2 - np.min(ycen).astype(int)  # TODO
    y_lower_lim = int(y_lower_lim)

    sl = np.zeros(ny, dtype=c_double)
    csl = ffi.cast("double *", sl.ctypes.data)

    # Inital guess for spectrum
    sp = np.sum(img, axis=0) / (img.size - np.ma.count_masked(img))
    sp = np.require(sp, dtype=c_double, requirements=["C", "A", "W", "O"])
    csp = ffi.cast("double *", sp.ctypes.data)

    mask = ~np.ma.getmaskarray(img)
    mask = np.require(mask, dtype=c_int, requirements=["C", "A", "W", "O"])
    cmask = ffi.cast("int *", mask.ctypes.data)

    img = np.require(np.ma.getdata(img), dtype=c_double, requirements=["C", "A", "W", "O"])
    cimg = ffi.cast("double *", img.ctypes.data)

    ycen = np.require(ycen, dtype=c_double, requirements=["C", "A", "W", "O"])
    cycen = ffi.cast("double *", ycen.ctypes.data)

    ycen_offset = np.require(ycen, dtype=c_int, requirements=["C", "A", "W", "O"])
    cycen_offset = ffi.cast("int *", ycen_offset.ctypes.data)

    shear = np.require(shear, dtype=c_double, requirements=["C", "A", "W", "O"])
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
        lambda_sf,
        csp,
        csl,
        cmodel,
        cunc,
    )
    mask = ~mask.astype(bool)

    return sp, sl, model, unc, mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import readsav
    from scipy.signal import gaussian

    spec = 10 + 2 * np.sin(
        np.linspace(0, 40 * np.pi, 100)
    )
    slitf = gaussian(40, 2)[:, None]
    img = spec[None, :] * slitf
    ycen = np.zeros(50)

    # sav = readsav("./Test/test.dat")
    # img = sav["im"]
    # ycen = sav["ycen"]

    shear = np.full(50, 0.2)
    # for i in range(img.shape[1]):
    #     img[:, i] = np.roll(img[:, i], 2-i//8)
    for i in range(img.shape[0]):
        img[i] = np.roll(img[i], -i // 5)
    img = img[10:-10, :50]

    sp, sl, model, unc, mask = slitfunc_curved(img, ycen, shear)

    plt.subplot(211)
    plt.imshow(img)
    plt.title("Observation")

    plt.subplot(212)
    plt.imshow(model)
    plt.title("Model")
    plt.show()

    plt.subplot(211)
    plt.plot(sp)
    plt.title("Spectrum")

    plt.subplot(212)
    plt.plot(sl)
    plt.title("Slitfunction")
    plt.show()
