#!/usr/bin/env python3

import numpy as np
import pylab as p
from cffi import FFI

ffi = FFI()
ffi.cdef("void copy(float *in, float *out, int len); void medfilt(float *in, float *out, int len, int hw);")
C = ffi.dlopen("cfuncs.so")

def cast_double(a):
    return ffi.cast("float *", a.ctypes.data)

a = np.random.rand(45).astype(np.float32) + np.arange(45)/10
b = np.zeros_like(a)

C.medfilt(cast_double(a), cast_double(b), len(a), 5)
print(b)
p.plot(a,'ko')
p.plot(b,'-r')
p.show()
