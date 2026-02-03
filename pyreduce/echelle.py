"""Deprecated: use pyreduce.spectra instead.

This module is provided for backward compatibility only.
All functionality has been moved to pyreduce.spectra.
"""

import warnings

warnings.warn(
    "pyreduce.echelle is deprecated and will be removed in a future release. "
    "Use pyreduce.spectra instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from spectra for backward compatibility
from pyreduce.spectra import *  # noqa: F401, F403, E402
from pyreduce.spectra import Spectra as Echelle  # noqa: F401, E402
