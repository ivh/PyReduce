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
    """Build the slitdec C library in-place."""
    print("Building CFFI extensions for development...")
    print(f"  Source dir: {CWD}")

    old_cwd = os.getcwd()
    os.chdir(CWD)

    try:
        # Slit decomposition (slitdec.c, copied from charslit)
        ffibuilder_slitdec = FFI()
        with open("slitdec.h") as f:
            ffibuilder_slitdec.cdef(f.read())
        with open("slitdec.c") as f:
            ffibuilder_slitdec.set_source(
                "_slitdec",
                f.read(),
                include_dirs=[CWD],
                depends=["slitdec.h"],
            )
        ffibuilder_slitdec.compile(verbose=True)
        print("[OK] _slitdec")

        print("Done.")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    build()
