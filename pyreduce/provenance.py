"""Provenance stamping for PyReduce output files."""

from __future__ import annotations

import logging
import os
import subprocess
from functools import lru_cache

import astropy.io.fits as fits

from . import __version__

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_git_revision() -> str | None:
    """Git description of the PyReduce source tree.

    Returns e.g. "v0.9b1-6-gef7c19d-dirty" when running from a git
    checkout ("-dirty" marks uncommitted changes), or None for installed
    packages.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    # Only trust git when the package sits directly in its own checkout;
    # otherwise git would report whatever repo happens to contain us
    if not os.path.isdir(os.path.join(os.path.dirname(pkg_dir), ".git")):
        return None
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=pkg_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("Could not determine git revision: %s", e)
        return None
    return result.stdout.strip() or None


def add_provenance(header: fits.Header | None = None) -> fits.Header:
    """Stamp PyReduce version and git revision into a FITS header."""
    if header is None:
        header = fits.Header()
    header["HIERARCH PR_version"] = __version__
    revision = get_git_revision()
    if revision is not None:
        header["HIERARCH PR_githash"] = (revision, "PyReduce git revision")
    return header
