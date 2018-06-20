#!/usr/bin/env python3

import os
from cffi import FFI

ffibuilder = FFI()

CWD = os.path.dirname(__file__)

with open(os.path.join(CWD, "extract.h")) as f:
    ffibuilder.cdef(f.read(), override=True)

with open(os.path.join(CWD, 'cluster.c'),'r') as f:
    ffibuilder.set_source("clib._cluster",
                          f.read(),
                          #libraries=["c"],
                          #sources=[os.path.join(CWD, "cluster.c")],
                          #library_dirs=["."]
                          #include_dirs=[os.path.join()]
                          )

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)