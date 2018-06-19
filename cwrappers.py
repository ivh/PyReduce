import numpy as np

import clib._cluster.lib as clusterlib
from clib._cluster import ffi

def color_clusters(img, thresh=1):
    img = img.astype('i')
    x,y = np.indices(img.shape)
    n = np.inner(*img.shape)
    colors = np.zeros_like(img)

    cimg = ffi.cast("int *", img.ctypes.data)
    ccolors = ffi.cast("int *", colors.ctypes.data)
    cx = ffi.cast("int *", x.ctypes.data)
    cy = ffi.cast("int *", y.ctypes.data)


    ncols = clusterlib.cluster(cx, cy, n, n, n, thresh, ccolors)

    return colors, ncols

if __name__ == "__main__":
    img = np.zeros((10, 10), dtype='i')

    img[1:2,1:3] = 1
    img[7:9,5:6] = 1

    colors, ncols = color_clusters(img)

    print( ncols, colors)
