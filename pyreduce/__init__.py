try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # for Python<3.8
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pyreduce-astro")
except PackageNotFoundError:
    __version__ = "unknown"


# add logger to console
import logging

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
logging.captureWarnings(True)

console = TqdmLoggingHandler()
console.setLevel(logging.INFO)

try:
    import colorlog

    console.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(levelname)s - %(message)s")
    )
    del colorlog
except ImportError:
    console.setFormatter("%(levelname)s - %(message)s")
    print("Install colorlog for colored logging output")

logger.addHandler(console)

del logging
# do not del tqdm, it is needed in the Log Handler


# Lazy loading for faster imports
def __getattr__(name):
    """Lazy load submodules on first access."""
    if name in ("configuration", "datasets", "reduce", "util"):
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
