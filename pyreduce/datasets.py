"""
Download example datasets for PyReduce.

Downloads tarballs from the PyReduce server and extracts them
to $REDUCE_DATA (or ~/REDUCE_DATA by default).
"""

import hashlib
import logging
import os
import tarfile
import urllib.request
from os.path import isfile, join

logger = logging.getLogger(__name__)

SERVER = "https://sme.astro.uu.se/pyreduce/"

# Map instrument name to (tarball_name, sha256_checksum)
DATASETS = {
    "UVES": (
        "UVES",
        "1b44274b10e05b62e1d2f37b13495298359f91acaa5e7ee88214740a238bf4ab",
    ),
    "HARPS": (
        "HARPS",
        "c913b7c911de16ed6fc6be1525e315c9b041837bea7a8c7b909eb73b76a80406",
    ),
    "LICK_APF": (
        "LICK_APF",
        "6695064122f69143d5ec548364c93d4ff0f9f5a59c7187eacf7439a697cb523a",
    ),
    "MCDONALD": (
        "MCDONALD",
        "43b060fdb20fa2b1e77d7a391b9a6d2ab6bc7207f80f825d5182324994b4074e",
    ),
    "JWST_MIRI": (
        "JWST_MIRI",
        "569ec6b0a3b5d46fcdbf73c33332eca57aaea29c5ca306f7d685325fb3e5f451",
    ),
    "JWST_NIRISS": (
        "NIRISS",
        "f5a1a2894970471c27e7cbd73aed6027f72cd0df0900ea000119cacb852d72d3",
    ),
    "KECK_NIRSPEC": (
        "NIRSPEC",
        "7005bf1dc6953093866ab8a359d66aa9be8c26fa43d351dcb9c4c0aab1f80b61",
    ),
    "XSHOOTER": (
        "XSHOOTER",
        "4a3a86a50163b4d2136703f953f5e91e0867f8971f726642aba9b96bdb5f551e",
    ),
}


def get_data_dir():
    """Get the default data directory.

    Returns $REDUCE_DATA if set, otherwise ~/REDUCE_DATA
    """
    return os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))


def _download_with_progress(url, dest):
    """Download a file with progress indicator."""

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent}%)", end="")
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            print(f"\r  {mb_downloaded:.1f} MB", end="")

    urllib.request.urlretrieve(url, dest, reporthook)
    print()  # newline after progress


def _verify_checksum(filepath, expected_sha256):
    """Verify SHA256 checksum of a file."""
    if expected_sha256 is None:
        return True
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_sha256:
        logger.error("Checksum mismatch for %s", filepath)
        logger.error("  Expected: %s", expected_sha256)
        logger.error("  Got:      %s", actual)
        return False
    return True


def get_dataset(name, local_dir=None):
    """Download and extract a dataset.

    Parameters
    ----------
    name : str
        Name of the dataset (e.g., "UVES", "HARPS")
    local_dir : str, optional
        Directory to save data (default: $REDUCE_DATA or ~/REDUCE_DATA)

    Returns
    -------
    str
        Directory where the data was saved
    """
    if name not in DATASETS:
        available = ", ".join(sorted(DATASETS.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    tarball_name, checksum = DATASETS[name]

    if local_dir is None:
        local_dir = get_data_dir()

    fname = f"{tarball_name}.tar.gz"
    data_dir = join(local_dir, name)
    filepath = join(data_dir, fname)

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.isfile(filepath):
        url = SERVER + fname
        logger.info("Downloading %s from %s", name, url)
        logger.info("Saving to %s", data_dir)
        _download_with_progress(url, filepath)

        if not _verify_checksum(filepath, checksum):
            os.remove(filepath)
            raise RuntimeError(f"Checksum verification failed for {name}")
    else:
        logger.info("Using existing dataset %s", name)

    # Extract
    with tarfile.open(filepath) as tar:
        raw_dir = join(data_dir, "raw")
        members = [m for m in tar if not isfile(join(raw_dir, m.name))]
        if members:
            logger.info("Extracting %d files", len(members))
            tar.extractall(path=raw_dir, members=members)

    return data_dir


# Convenience functions for each instrument
def UVES(local_dir=None):
    """Download UVES example dataset (target: HD132205)."""
    return get_dataset("UVES", local_dir)


def HARPS(local_dir=None):
    """Download HARPS example dataset (target: HD109200)."""
    return get_dataset("HARPS", local_dir)


def LICK_APF(local_dir=None):
    """Download Lick APF example dataset (target: KIC05005618)."""
    return get_dataset("LICK_APF", local_dir)


def MCDONALD(local_dir=None):
    """Download McDonald Observatory example dataset."""
    return get_dataset("MCDONALD", local_dir)


def JWST_MIRI(local_dir=None):
    """Download JWST/MIRI example dataset (simulated with MIRIsim)."""
    return get_dataset("JWST_MIRI", local_dir)


def JWST_NIRISS(local_dir=None):
    """Download JWST/NIRISS example dataset (simulated with awesimsoss)."""
    return get_dataset("JWST_NIRISS", local_dir)


def KECK_NIRSPEC(local_dir=None):
    """Download Keck/NIRSPEC example dataset (target: GJ1214)."""
    return get_dataset("KECK_NIRSPEC", local_dir)


def XSHOOTER(local_dir=None):
    """Download XSHOOTER example dataset (target: Ux-Ori)."""
    return get_dataset("XSHOOTER", local_dir)
