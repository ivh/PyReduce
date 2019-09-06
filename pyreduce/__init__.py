# Define Version
from ._version import get_versions
__version__ = get_versions()["version"]
del get_versions

import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    mpl.use("QT5Agg")
    plt.style.use("seaborn-paper")
except:
    pass
del mpl
del plt

# Start logging
from . import util
util.start_logging(None)

# Load externally available modules
from . import reduce, datasets, instruments, util, configuration
