#!/usr/bin/env python3

import os
import logging

from cffi import FFI


CWD = os.path.dirname(__file__)
print("Include dir: ", CWD)

ffibuilder_vertical = FFI()
with open(os.path.join(CWD, "slit_func_bd.h")) as f:
    ffibuilder_vertical.cdef(f.read())
with open(os.path.join(CWD, "slit_func_bd.c"), "r") as f:
    ffibuilder_vertical.set_source(
        "pyreduce.clib._slitfunc_bd",
        f.read(),
        include_dirs=[CWD],
        depends=["slit_func_bd.h"],
    )

ffibuilder_curved = FFI()
with open(os.path.join(CWD, "slit_func_2d_xi_zeta_bd.h")) as f:
    ffibuilder_curved.cdef(f.read())
with open(os.path.join(CWD, "slit_func_2d_xi_zeta_bd.c"), "r") as f:
    ffibuilder_curved.set_source(
        "pyreduce.clib._slitfunc_2d",
        f.read(),
        include_dirs=[CWD],
        depends=["slit_func_2d_xi_zeta_bd.h"],
    )


def build(**kwargs):
    """Build the C slitfunc library"""
    logging.info("Building required C libraries, this might take a few seconds")
    ffibuilder_vertical.compile(verbose=True)
    ffibuilder_curved.compile(verbose=True)


if __name__ == "__main__":
    build()
