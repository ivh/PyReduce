import numpy as np

import clib._cluster.lib as clusterlib
from clib._cluster import ffi

def find_clusters(img, min_cluster=4, filter_size=10, noise=1.0):
    img = img.astype('i')
    nX, nY = img.shape
    nmax = np.inner(*img.shape) - np.ma.count_masked(img)
    x = np.zeros(nmax, dtype='i')
    y = np.zeros(nmax, dtype='i')

    cimg = ffi.cast("int *", img.ctypes.data)
    cx = ffi.cast("int *", x.ctypes.data)
    cy = ffi.cast("int *", y.ctypes.data)

    if np.ma.is_masked(img):
        mask = ffi.cast("int *", (img.mask==False).astype('i').ctypes.data)
    else:
        mask = ffi.cast("int *", np.ones_like(img).ctypes.data)

    n = clusterlib.locate_clusters(nX, nY, filter_size, cimg, nmax, cx, cy, noise, mask)


    x=x[:n]
    y=y[:n]
    clusters = np.zeros(n)
    cclusters = ffi.cast("int *", clusters.ctypes.data)

    nclus = clusterlib.cluster(cx, cy, n, nX, nY, min_cluster, cclusters)

    return x, y, clusters, nclus

if __name__ == "__main__":
    img = np.zeros((100, 100), dtype='i') + 10

    img[:,11:22] = 100
    img[80:90,80:90] = 1

    x, y, clusters, nclus = find_clusters(img)
    print( nclus, len(x), x, y, clusters)
