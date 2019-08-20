# Define Version
from ._version import get_versions
__version__ = get_versions()["version"]
del get_versions

# Start logging
from . import util
util.start_logging(None)

# Make sure that clibraries exists and are compilled
import os
dirname = os.path.dirname(__file__)
fname = os.path.join(dirname, "clib", "_slitfunc_2d.c")

if not os.path.exists(fname):
    from .clib import build_extract
    build_extract.build()
    del build_extract
del os

# Load externally available modules
from . import reduce, datasets, instruments, util, configuration
