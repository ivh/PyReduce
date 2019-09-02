# Define Version
from ._version import get_versions
__version__ = get_versions()["version"]
del get_versions

# Start logging
from . import util
util.start_logging(None)

# Load externally available modules
from . import reduce, datasets, instruments, util, configuration
