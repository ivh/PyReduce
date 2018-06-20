#!/usr/bin/env python3

import os
from cffi import FFI


def build(**kwargs):
    """Build the cluster C library"""
    ffibuilder = FFI()

    CWD = os.path.dirname(__file__)

    with open(os.path.join(CWD, "cluster.h")) as f:
        ffibuilder.cdef(f.read(), override=True)

    with open(os.path.join(CWD, 'cluster.c'), 'r') as f:
        ffibuilder.set_source("clib._cluster",
                              f.read(),
                              # libraries=["c"],
                              #sources=[os.path.join(CWD, "cluster.c")],
                              # library_dirs=["."]
                              # include_dirs=[os.path.join()]
                              )
    ffibuilder.compile(**kwargs)


if __name__ == "__main__":
    build(verbose=True)
