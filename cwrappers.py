import numpy as np

import clib.build_slitfunc
clib.build_slitfunc.build()

import clib._cluster.lib as clusterlib
import clib._slitfunc_bd.lib as slitfunclib
from clib._cluster import ffi


def slitfunc(img, ycen, lambda_sp=0, lambda_sl=0.1, osample=1, noise=0):
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
    #img = np.ascontiguousarray(img.T)
    img = img.astype(np.ctypeslib.ctypes.c_double)
    ncols, nrows = img.shape
    ny = osample * (nrows + 1) + 1

    cimg = ffi.cast("double **", img.ctypes.data)
    if np.ma.is_masked(img):
        mask = ffi.cast("unsigned char **", (~img.mask).astype(int).ctypes.data)
    else:
        mask = ffi.cast("unsigned char **", np.ones(img.shape, dtype=int).ctypes.data)

    ycen = ycen.astype(np.ctypeslib.ctypes.c_double)
    cycen = ffi.cast("double *", ycen.ctypes.data)

    sp = np.zeros(ncols, dtype=np.ctypeslib.ctypes.c_double)
    csp = ffi.cast("double *", sp.ctypes.data)

    sl = np.zeros(ny, dtype=np.ctypeslib.ctypes.c_double)
    csl = ffi.cast("double *", sl.ctypes.data)

    model = np.zeros((nrows, ncols), dtype=np.ctypeslib.ctypes.c_double)
    cmodel = ffi.cast("double **", model.ctypes.data)

    unc = np.zeros(ncols, dtype=np.ctypeslib.ctypes.c_double)
    cunc = ffi.cast("double *", unc.ctypes.data)

    omega = np.zeros((ny, nrows, ncols), dtype=np.ctypeslib.ctypes.c_double)
    comega = ffi.cast("double ***", omega.ctypes.data)

    sp_old = np.zeros((ny, nrows, ncols), dtype=np.ctypeslib.ctypes.c_double)
    csp_old = ffi.cast("double *", sp_old.ctypes.data)

    aij = np.zeros((ny * ny), dtype=np.ctypeslib.ctypes.c_double)
    caij = ffi.cast("double *", aij.ctypes.data)

    bj = np.zeros(ny, dtype=np.ctypeslib.ctypes.c_double)
    cbj = ffi.cast("double *", bj.ctypes.data)

    a_diag = np.zeros(3 * ncols, dtype=np.ctypeslib.ctypes.c_double)
    ca_diag = ffi.cast("double *", a_diag.ctypes.data)

    e = np.zeros(ncols, dtype=np.ctypeslib.ctypes.c_double)
    ce = ffi.cast("double *", e.ctypes.data)

    slitfunclib.slit_func_vert(ncols, nrows, ny, cimg,
                               mask, cycen, osample, lambda_sp, lambda_sl, csp, csl, cmodel, cunc, comega, csp_old, caij, cbj, ca_diag, ce)

    return sp, sl, model  # , unc, omega, sP_old, Aij, nj, Adiag, E


def find_clusters(img, min_cluster=4, filter_size=10, noise=1.0):
    img = img.T  # transpose input

    img = img.astype('i')
    nX, nY = img.shape
    nmax = np.inner(*img.shape) - np.ma.count_masked(img)
    x = np.zeros(nmax, dtype='i')
    y = np.zeros(nmax, dtype='i')

    cimg = ffi.cast("int *", img.ctypes.data)
    cx = ffi.cast("int *", x.ctypes.data)
    cy = ffi.cast("int *", y.ctypes.data)

    if np.ma.is_masked(img):
        mask = ffi.cast("int *", (~img.mask).astype('i').ctypes.data)
    else:
        mask = ffi.cast("int *", np.ones_like(img).ctypes.data)

    n = clusterlib.locate_clusters(
        nX, nY, filter_size, cimg, nmax, cx, cy, noise, mask)

    x = x[:n]
    y = y[:n]
    clusters = np.zeros(n)
    cclusters = ffi.cast("int *", clusters.ctypes.data)

    nclus = clusterlib.cluster(cx, cy, n, nX, nY, min_cluster, cclusters)

    # transpose output
    return y, x, clusters, nclus


if __name__ == "__main__":
    img = np.full((110, 90), 1., dtype=float)
    ycen = np.full(90, 10, dtype=float)

    img[:, 10] = 20

    sp, sl, model = slitfunc(img, ycen)

    print(sp)

    # import matplotlib.pyplot as plt
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
