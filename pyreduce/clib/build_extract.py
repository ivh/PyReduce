#!/usr/bin/env python3

import os
import logging

from cffi import FFI


CWD = os.path.dirname(__file__)
CWD = os.path.abspath(CWD)
release_path = os.path.join(CWD, "Release")

print("Include dir: ", CWD)
print("Release dir: ", release_path)


ffibuilder_vertical = FFI()
with open(os.path.join(CWD, "slit_func_bd.h")) as f:
    ffibuilder_vertical.cdef(f.read())
with open(os.path.join(CWD, "slit_func_bd.c"), "r") as f:
    ffibuilder_vertical.set_source(
        "_slitfunc_bd",
        f.read(),
        include_dirs=[CWD, release_path],
        depends=["slit_func_bd.h"],
    )

ffibuilder_curved = FFI()
with open(os.path.join(CWD, "slit_func_2d_xi_zeta_bd.h")) as f:
    ffibuilder_curved.cdef(f.read())
with open(os.path.join(CWD, "slit_func_2d_xi_zeta_bd.c"), "r") as f:
    ffibuilder_curved.set_source(
        "_slitfunc_2d",
        f.read(),
        include_dirs=[CWD, release_path],
        depends=["slit_func_2d_xi_zeta_bd.h"],
    )


def build(**kwargs):
    """Build the C slitfunc library"""
    logging.info("Building required C libraries, this might take a few seconds")
    
    old_cwd = os.getcwd()
    path = os.path.abspath(CWD)
    os.chdir(path)
    
    ffibuilder_vertical.compile(verbose=True, debug=False)
    ffibuilder_curved.compile(verbose=True, debug=False)

    os.chdir(old_cwd)

if __name__ == "__main__":
    build()
