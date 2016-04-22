#!/usr/bin/env python3

import numpy as np
from cffi import FFI

ffi = FFI()
ffi.cdef("void copy(float *in, float *out, int len);")
C = ffi.dlopen("cfuncs.so")

def cast_double(a):
    return ffi.cast("float *", a.ctypes.data)

a = 42*np.ones(16, dtype=np.float32)
b = np.zeros_like(a)

C.copy(cast_double(a), cast_double(b), len(a))
print(b)
