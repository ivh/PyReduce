"""
Hatch build hook for compiling CFFI extensions.

This hook runs during the build process to compile the CFFI C extensions
before the wheel is created.
"""

import os
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build hook that compiles CFFI extensions."""

    def initialize(self, version, build_data):
        """
        Initialize the build hook and compile CFFI extensions.

        This method is called by hatchling during the build process.
        """
        # Mark wheel as platform-specific (not py3-none-any)
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        print("=" * 60)
        print("Building CFFI extensions for PyReduce")
        print("=" * 60)

        # Get paths
        project_root = Path(self.root)
        clib_dir = project_root / "pyreduce" / "clib"

        # Add project root to sys.path so we can import
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            # Import the CFFI FFI builder
            from cffi import FFI

            # Build vertical extraction
            print("\n[1/2] Building vertical extraction extension...")
            ffibuilder_vertical = FFI()

            with open(clib_dir / "slit_func_bd.h") as f:
                ffibuilder_vertical.cdef(f.read())

            with open(clib_dir / "slit_func_bd.c") as f:
                ffibuilder_vertical.set_source(
                    "_slitfunc_bd",
                    f.read(),
                    include_dirs=[str(clib_dir), str(clib_dir / "Release")],
                    depends=["slit_func_bd.h"],
                )

            old_cwd = os.getcwd()
            try:
                os.chdir(clib_dir)
                ffibuilder_vertical.compile(verbose=True)
                print("[OK] Vertical extraction extension built successfully\n")
            finally:
                os.chdir(old_cwd)

            # Build curved extraction
            print("[2/2] Building curved extraction extension...")
            ffibuilder_curved = FFI()

            with open(clib_dir / "slit_func_2d_xi_zeta_bd.h") as f:
                ffibuilder_curved.cdef(f.read())

            with open(clib_dir / "slit_func_2d_xi_zeta_bd.c") as f:
                ffibuilder_curved.set_source(
                    "_slitfunc_2d",
                    f.read(),
                    include_dirs=[str(clib_dir), str(clib_dir / "Release")],
                    depends=["slit_func_2d_xi_zeta_bd.h"],
                )

            old_cwd = os.getcwd()
            try:
                os.chdir(clib_dir)
                ffibuilder_curved.compile(verbose=True)
                print("[OK] Curved extraction extension built successfully\n")
            finally:
                os.chdir(old_cwd)

            print("=" * 60)
            print("All CFFI extensions built successfully!")
            print("=" * 60)

        except Exception as e:
            print(f"ERROR: Failed to build CFFI extensions: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            raise
