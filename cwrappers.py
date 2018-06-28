"""
Wrapper for REDUCE C functions 
includes:
   slit_func
   locate_cluster
"""

import numpy as np

# TODO DEBUG
# import clib.build_cluster
# clib.build_cluster.build()

from clib._cluster import ffi
import clib._cluster.lib as clusterlib

c_double = np.ctypeslib.ctypes.c_double
c_int = np.ctypeslib.ctypes.c_int

def find_clusters(img, min_cluster=4, filter_size=10, noise=1.0):
    # img = img.T  # transpose input TODO: why?

    if np.ma.is_masked(img):
        mask = (~img.mask).astype(c_int).flatten()
        mask = np.ascontiguousarray(mask)
        cmask = ffi.cast("int *", mask.ctypes.data)
    else:
        mask = np.ones(img.size, dtype=c_int)
        cmask = ffi.cast("int *", mask.ctypes.data)

    nY, nX = img.shape
    nmax = np.inner(*img.shape) - np.ma.count_masked(img)
    x = np.zeros(nmax, dtype=c_int)
    y = np.zeros(nmax, dtype=c_int)

    img = img.astype(c_int).flatten()
    img = np.ascontiguousarray(img.data)
    cimg = ffi.cast("int *", img.ctypes.data)
    cx = ffi.cast("int *", x.ctypes.data)
    cy = ffi.cast("int *", y.ctypes.data)

    n = clusterlib.locate_clusters(
        nX, nY, filter_size, cimg, nmax, cx, cy, noise, cmask
    )

    x = x[:n]
    y = y[:n]
    clusters = np.zeros(n)
    cclusters = ffi.cast("int *", clusters.ctypes.data)

    nclus = clusterlib.cluster(cx, cy, n, nX, nY, min_cluster, cclusters)

    # transpose output
    return y, x, clusters, nclus


if __name__ == "__main__":
    import matplotlib.pyplot as plt

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
