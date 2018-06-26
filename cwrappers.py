"""
Wrapper for REDUCE C functions 
includes:
   slit_func
   locate_cluster
"""


import numpy as np
from scipy.ndimage.filters import median_filter, gaussian_filter1d
from scipy.interpolate import interp1d

# TODO DEBUG
import clib.build_slitfunc

clib.build_slitfunc.build()

import clib._cluster.lib as clusterlib
import clib._slitfunc_bd.lib as slitfunclib
import clib._slitfunc_2d.lib as slitfunc_2dlib
from clib._cluster import ffi

c_double = np.ctypeslib.ctypes.c_double
c_int = np.ctypeslib.ctypes.c_int


def slitfunc(img, ycen, lambda_sp=0, lambda_sl=0.1, osample=1):
    """
    In:
    int ncols,                        Swath width in pixels
    int nrows,                        Extraction slit height in pixels
    int ny,                           Size of the slit function array: ny=osample(nrows+1)+1
    double im[nrows][ncols],          Image to be decomposed
    byte mask[nrows][ncols],          Initial and final mask for the swath
    double ycen[ncols],               Order centre line offset from pixel row boundary
    int osample,                      Subpixel ovsersampling factor
    double lambda_sP,                 Smoothing parameter for the spectrum, coiuld be zero
    double lambda_sL,                 Smoothing parameter for the slit function, usually >0
    Out:
    double sP[ncols],                 Spectrum resulting from decomposition
    double sL[ny],                    Slit function resulting from decomposition
    double model[nrows][ncols],       Model constructed from sp and sf
    double unc[ncols],                Spectrum uncertainties
    double omega[ny][nrows][ncols]    Work array telling what fraction of subpixel iy falls into pixel {x,y}.
    double sP_old[ncols],             Work array to control the convergence
    double Aij[],                     Various LAPACK arrays (ny*ny)
    double bj[],                      ny
    double Adiag[],                   Array for solving the tridiagonal SLE for sP (ncols*3)
    double E[])                       RHS (ncols)
    """

    # Get dimensions
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    # Inital guess for slit function and spectrum
    # Just sum up the image along one side and normalize
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

    img = img.flatten().astype(c_double)
    img = np.ascontiguousarray(img)
    cimg = ffi.cast("double *", img.ctypes.data)

    if np.ma.is_masked(img):
        mask = (~img.mask).astype(c_int).flatten()
        cmask = ffi.cast("int *", mask.ctypes.data)
    else:
        mask = np.ones(nrows * ncols, dtype=c_int)
        cmask = ffi.cast("int *", mask.ctypes.data)

    ycen = ycen.astype(c_double)
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
        lambda_sl,
        csp,
        csl,
        cmodel,
        cunc,
    )

    return sp, sl, model, unc


def slitfunc_curved(img, ycen, shear, osample=1, lambda_sp=0, lambda_sl=0.1, **kwargs):
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    y_lower_lim = int(min(ycen))  # TODO

    # Inital guess for slit function and spectrum
    # Just sum up the image along one side and normalize
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


def find_clusters(img, min_cluster=4, filter_size=10, noise=1.0):
    img = img.T  # transpose input TODO: why?

    img = img.astype("i")
    nX, nY = img.shape
    nmax = np.inner(*img.shape) - np.ma.count_masked(img)
    x = np.zeros(nmax, dtype="i")
    y = np.zeros(nmax, dtype="i")

    cimg = ffi.cast("int *", img.ctypes.data)
    cx = ffi.cast("int *", x.ctypes.data)
    cy = ffi.cast("int *", y.ctypes.data)

    if np.ma.is_masked(img):
        mask = ffi.cast("int *", (~img.mask).astype("i").ctypes.data)
    else:
        mask = ffi.cast("int *", np.ones_like(img).ctypes.data)

    n = clusterlib.locate_clusters(nX, nY, filter_size, cimg, nmax, cx, cy, noise, mask)

    x = x[:n]
    y = y[:n]
    clusters = np.zeros(n)
    cclusters = ffi.cast("int *", clusters.ctypes.data)

    nclus = clusterlib.cluster(cx, cy, n, nX, nY, min_cluster, cclusters)

    # transpose output
    return y, x, clusters, nclus


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import readsav

    sav = readsav("./Test/test.dat")
    img = sav["im"]
    ycen = sav["ycen"]
    shear = np.zeros(img.shape[1])

    sp, sl, model, unc = slitfunc_curved(img, ycen, shear, osample=1)

    plt.subplot(211)
    plt.plot(sp)
    plt.title("Spectrum")

    plt.subplot(212)
    plt.plot(sl)
    plt.title("Slitfunction")
    plt.show()

    # print(sp)

    #
    # img = np.zeros((101, 103), dtype='i') + 10

    # img[11:22, :] = 100
    # img[80:90, 80:90] = 1

    # x, y, clusters, nclus = find_clusters(img, filter_size=20)
    # cluster_img = np.zeros_like(img)

    # cluster_img[x, y] = 255

    # #print(nclus, len(x), x, y, clusters)

    # plt.subplot(121)
    # plt.imshow(img)
    # plt.title("Input")

    # plt.subplot(122)
    # plt.imshow(cluster_img)
    # plt.title("Clusters")

    # plt.show()
