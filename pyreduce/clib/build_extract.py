#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builds the C library that contains the extraction algorithm

This module prepares and builds the C libary containing the curved
(and vertical) extraction algorithm using CFFI.
It also prepares the ffibuilder objects for setup.py,
so that the library is compiled on installation.


The user can also call the Module as a script to compile the
C libraries again.

Attributes
----------
ffi_builder_vertical : FFI
    CFFI Builder for the vertical extraction algorithm
ffi_builder_curved : FFI
    CFFI Builder for the curved extraction algorithm
"""

import logging
import os

from cffi import FFI

logger = logging.getLogger(__name__)


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


def build():
    """Builds the C slitfunc library"""
    logger.info("Building required C libraries, this might take a few seconds")

    old_cwd = os.getcwd()
    path = os.path.abspath(CWD)
    os.chdir(path)

    ffibuilder_vertical.compile(verbose=True, debug=False)
    ffibuilder_curved.compile(verbose=True, debug=False)

    os.chdir(old_cwd)


if __name__ == "__main__":  # pragma: no cover
    build()
