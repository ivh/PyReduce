#!/usr/bin/env python3
"""Build script for CFFI C extensions (development only).

Usage:
    uv run reduce-build    # compile extensions
    uv run reduce-clean    # remove compiled files

This compiles the C extraction libraries in-place for development.
For production, extensions are built automatically during wheel creation
via hatch_build.py.
"""

import glob
import os

from cffi import FFI

CWD = os.path.dirname(__file__)
CWD = os.path.abspath(CWD)
release_path = os.path.join(CWD, "Release")


def clean():
    """Remove compiled extension files."""
    patterns = ["*.so", "*.o", "*.pyd"]
    removed = []
    for pattern in patterns:
        for f in glob.glob(os.path.join(CWD, pattern)):
            os.remove(f)
            removed.append(os.path.basename(f))
    if removed:
        print(f"Removed: {', '.join(removed)}")
    else:
        print("Nothing to clean.")


def build():
    """Build the C slitfunc libraries in-place."""
    print("Building CFFI extensions for development...")
    print(f"  Source dir: {CWD}")

    old_cwd = os.getcwd()
    os.chdir(CWD)

    try:
        # Vertical extraction
        ffibuilder_vertical = FFI()
        with open("slit_func_bd.h") as f:
            ffibuilder_vertical.cdef(f.read())
        with open("slit_func_bd.c") as f:
            ffibuilder_vertical.set_source(
                "_slitfunc_bd",
                f.read(),
                include_dirs=[CWD, release_path],
                depends=["slit_func_bd.h"],
            )
        ffibuilder_vertical.compile(verbose=True)
        print("[OK] _slitfunc_bd")

        # Curved extraction
        ffibuilder_curved = FFI()
        with open("slit_func_2d_xi_zeta_bd.h") as f:
            ffibuilder_curved.cdef(f.read())
        with open("slit_func_2d_xi_zeta_bd.c") as f:
            ffibuilder_curved.set_source(
                "_slitfunc_2d",
                f.read(),
                include_dirs=[CWD, release_path],
                depends=["slit_func_2d_xi_zeta_bd.h"],
            )
        ffibuilder_curved.compile(verbose=True)
        print("[OK] _slitfunc_2d")

        print("Done.")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    build()
