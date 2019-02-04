#!/usr/bin/env python3

import os
import logging

from cffi import FFI


def build(**kwargs):
    """Build the C slitfunc library"""
    logging.info("Building required C libraries, this might take a few seconds")
    CWD = os.path.dirname(__file__)
    ModulePath = os.path.dirname(CWD)
    print(ModulePath)

    ffibuilder = FFI()
    with open(os.path.join(CWD, "slit_func_bd.h")) as f:
        ffibuilder.cdef(f.read(), override=True)
    with open(os.path.join(CWD, "slit_func_bd.c"), "r") as f:
        ffibuilder.set_source("clib._slitfunc_bd", f.read(), include_dirs=[CWD])
    ffibuilder.compile(tmpdir=ModulePath, verbose=True, **kwargs)

    ffibuilder = FFI()
    with open(os.path.join(CWD, "slit_func_2d_xi_zeta_bd.h")) as f:
        ffibuilder.cdef(f.read(), override=True)
    with open(os.path.join(CWD, "slit_func_2d_xi_zeta_bd.c"), "r") as f:
        ffibuilder.set_source("clib._slitfunc_2d", f.read(), include_dirs=[CWD])
    ffibuilder.compile(tmpdir=ModulePath, verbose=True, **kwargs)


if __name__ == "__main__":
    build()
