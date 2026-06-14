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

            # Build slit decomposition (slitdec.c, copied from charslit)
            print("\n[1/1] Building slit decomposition extension...")
            ffibuilder_slitdec = FFI()

            with open(clib_dir / "slitdec.h") as f:
                ffibuilder_slitdec.cdef(f.read())

            with open(clib_dir / "slitdec.c") as f:
                ffibuilder_slitdec.set_source(
                    "_slitdec",
                    f.read(),
                    include_dirs=[str(clib_dir)],
                    depends=["slitdec.h"],
                )

            old_cwd = os.getcwd()
            try:
                os.chdir(clib_dir)
                ffibuilder_slitdec.compile(verbose=True)
                print("[OK] slit decomposition extension built successfully\n")
            finally:
                os.chdir(old_cwd)

            print("=" * 60)
            print("CFFI extension built successfully!")
            print("=" * 60)

        except Exception as e:
            print(f"ERROR: Failed to build CFFI extensions: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            raise
