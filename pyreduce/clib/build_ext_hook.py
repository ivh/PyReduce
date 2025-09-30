"""
Build hook for compiling CFFI C extensions during package installation.

This module integrates CFFI C extension building with setuptools by providing
a custom build_py command that triggers CFFI compilation before the Python
files are built.
"""

import os
import sys

from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Custom build_py command that compiles CFFI extensions first."""

    def run(self):
        """Execute the build, compiling CFFI extensions first."""
        # Import and build the CFFI extensions
        try:
            from pyreduce.clib.build_extract import (
                ffibuilder_curved,
                ffibuilder_vertical,
            )

            print("Building CFFI extensions...")
            print("Building vertical extraction extension...")
            ffibuilder_vertical.compile(verbose=True)

            print("Building curved extraction extension...")
            ffibuilder_curved.compile(verbose=True)

            print("CFFI extensions built successfully")

        except Exception as e:
            print(f"Error building CFFI extensions: {e}", file=sys.stderr)
            raise

        # Continue with the normal build
        super().run()