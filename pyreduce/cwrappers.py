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
    from .clib._cluster import lib as clusterlib
    from .clib._slitfunc_bd import ffi
except ImportError:
    raise ImportError("Use setup.py to compile the C libraries")


c_double = np.ctypeslib.ctypes.c_double
c_int = np.ctypeslib.ctypes.c_int


def find_clusters(img, min_cluster=4, filter_size=10, noise=1.0):
    """Wrapper for locate_clusters and cluster, which find clustered pixels

    Parameters
    ----------
    img : array[nrow, ncol]
        order definition image
    min_cluster : int, optional
        minimum size of cluster (default: 4)
    filter_size : int, optional
        size of the mean(?) filter (default: 10)
    noise : float, optional
        how much noise to filter out (default: 1.0)

    Returns
    -------
    y : array[n](int)
        y coordinates of pixels in clusters
    x : array[n](int)
        x coordinates of pixels in clusters
    clusters : array[n](int)
        cluster id of pixels in clusters
    nclusters : int
        number of clusters
    """

    nY, nX = img.shape
    nmax = nY * nX - np.ma.count_masked(img)

    min_cluster = int(min_cluster)
    filter_size = int(filter_size)
    noise = float(noise)

    mask = ~np.ma.getmaskarray(img).astype(int)
    mask = np.require(mask, dtype=c_int, requirements=["C", "A", "W", "O"])

    img = np.ma.getdata(img)
    img = np.require(img, dtype=c_int, requirements=["C", "A", "W", "O"])

    x = np.zeros(nmax, dtype=c_int)
    y = np.zeros(nmax, dtype=c_int)

    # Find all pixels above the threshold
    n = clusterlib.locate_clusters(
        ffi.cast("int", nX),
        ffi.cast("int", nY),
        ffi.cast("int", filter_size),
        ffi.cast("int *", img.ctypes.data),
        ffi.cast("int", nmax),
        ffi.cast("int *", x.ctypes.data),
        ffi.cast("int *", y.ctypes.data),
        ffi.cast("float", noise),
        ffi.cast("int *", mask.ctypes.data),
    )

    # remove unnecessary memory
    x = x[:n]
    y = y[:n]

    # Not sure its necessay but the numbering is nicer if we do this
    sort = np.argsort(y)
    y = np.require(y[sort], dtype=c_int, requirements=["C", "A", "W", "O"])
    x = np.require(x[sort], dtype=c_int, requirements=["C", "A", "W", "O"])

    clusters = np.zeros(n, dtype=c_int)

    # Group the pixels into clusters
    nclus = clusterlib.cluster(
        ffi.cast("int *", x.ctypes.data),
        ffi.cast("int *", y.ctypes.data),
        ffi.cast("int", n),
        ffi.cast("int", nX),
        ffi.cast("int", nY),
        ffi.cast("int", min_cluster),
        ffi.cast("int *", clusters.ctypes.data),
    )

    # transpose output
    return y, x, clusters, nclus


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

    ycen = np.require(ycen, dtype=c_double, requirements=["C", "A", "W", "O"])
    model = np.zeros((nrows, ncols), dtype=c_double)
    unc = np.zeros(ncols, dtype=c_double)

    slitfunclib.slit_func_vert(
        ffi.cast("int", ncols),
        ffi.cast("int", nrows),
        ffi.cast("double *", img.ctypes.data),
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

    lambda_sf = float(lambda_sf)
    lambda_sp = float(lambda_sp)
    osample = int(osample)

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

    ycen = np.require(ycen, dtype=c_double, requirements=["C", "A", "W", "O"])
    ycen_offset = np.require(ycen, dtype=c_int, requirements=["C", "A", "W", "O"])

    if np.isscalar(shear):
        shear = np.full(ncols, shear, dtype=c_double)
    shear = np.require(shear, dtype=c_double, requirements=["C", "A", "W", "O"])

    model = np.zeros((nrows, ncols), dtype=c_double)
    unc = np.zeros(ncols, dtype=c_double)

    slitfunc_2dlib.slit_func_curved(
        ffi.cast("int", ncols),
        ffi.cast("int", nrows),
        ffi.cast("double *", img.ctypes.data),
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.signal import gaussian

    # Clusters test
    img = np.zeros((101, 103), dtype="i") + 10

    img[11:22, :] = 100
    img[80:90, 80:90] = 1

    x, y, clusters, nclus = find_clusters(img, filter_size=20)
    cluster_img = np.zeros_like(img)

    cluster_img[x, y] = 255

    # print(nclus, len(x), x, y, clusters)

    plt.subplot(121)
    plt.imshow(img)
    plt.title("Input")

    plt.subplot(122)
    plt.imshow(cluster_img)
    plt.title("Clusters")

    plt.show()

    # Slitfunc test

    spec = 10 + 2 * np.sin(np.linspace(0, 40 * np.pi, 100))
    slitf = gaussian(40, 2)[:, None]
    img = spec[None, :] * slitf
    ycen = np.zeros(50)

    shear = np.full(50, 0.2)
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
