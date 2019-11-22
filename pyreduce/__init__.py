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

# add logger to console
import logging
import colorlog
import tqdm

# We need to use this to have logging messages handle properly with the progressbar
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console = TqdmLoggingHandler()
console.setLevel(logging.INFO)
console.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s - %(message)s")
)
logger.addHandler(console)


del logging
del colorlog

# Load externally available modules
from . import reduce, datasets, instruments, util, configuration
