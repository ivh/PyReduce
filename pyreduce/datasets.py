# -*- coding: utf-8 -*-
"""
Provides example datasets for the examples

This requires the server to be up and running,
if data needs to be downloaded
"""
import logging
import os
import tarfile
from os.path import dirname, isfile, join

import wget

logger = logging.getLogger(__name__)


def load_data_from_server(filename, directory):
    server = r"http://sme.astro.uu.se/pyreduce/"
    url = server + filename
    directory = join(directory, filename)
    wget.download(url, out=directory)


def get_dataset(name, local_dir=None):
    """Load a dataset

    Note
    ----
    This method will not override existing files with the same
    name, even if they have a different content. Therefore
    if the files were changed for any reason, the user has to
    manually delete them from the disk before using this method.

    Parameters
    ----------
    name : str
        Name of the dataset
    local_dir : str, optional
        directory to save data at (default: "./")

    Returns
    -------
    dataset_dir : str
        directory where the data was saved
    """

    if local_dir is None:
        local_dir = dirname(__file__)
        local_dir = join(local_dir, "../")

    # load data if necessary
    fname = f"{name}.tar.gz"
    data_dir = join(local_dir, "datasets", name)
    filename = join(data_dir, fname)

    os.makedirs(data_dir, exist_ok=True)
    if not os.path.isfile(filename):
        logger.info("Downloading dataset %s", name)
        logger.info("Data is stored at %s", data_dir)
        load_data_from_server(fname, data_dir)
    else:
        logger.info("Using existing dataset %s", name)

    # Extract the downloaded .tar.gz file
    with tarfile.open(filename) as file:
        raw_dir = join(data_dir, "raw")
        names = [f for f in file if not isfile(join(raw_dir, f.name))]
        if len(names) != 0:
            logger.info("Extracting data from tarball")
            file.extractall(path=raw_dir, members=names)

    return data_dir


def UVES(local_dir=None):  # pragma: no cover
    """Load an example dataset
    instrument: UVES
    target: HD132205

    Parameters
    ----------
    local_dir : str, optional
        directory to save data at (default: "./")

    Returns
    -------
    dataset_dir : str
        directory where the data was saved
    """

    return get_dataset("UVES", local_dir)


def HARPS(local_dir=None):  # pragma: no cover
    """Load an example dataset
    instrument: HARPS
    target: HD109200

    Parameters
    ----------
    local_dir : str, optional
        directory to save data at (default: "./")

    Returns
    -------
    dataset_dir : str
        directory where the data was saved
    """

    return get_dataset("HARPS", local_dir)


def LICK_APF(local_dir=None):  # pragma: no cover
    """Load an example dataset
    instrument: LICK_APF
    target: KIC05005618

    Parameters
    ----------
    local_dir : str, optional
        directory to save data at (default: "./")

    Returns
    -------
    dataset_dir : str
        directory where the data was saved
    """

    return get_dataset("APF", local_dir)


def MCDONALD(local_dir=None):  # pragma: no cover
    """Load an example dataset
    instrument: JWST_MIRI
    target: ?

    Data simulated with MIRIsim

    Parameters
    ----------
    local_dir : str, optional
        directory to save data at (default: "./")

    Returns
    -------
    dataset_dir : str
        directory where the data was saved
    """

    return get_dataset("MCDONALD", local_dir)


def JWST_MIRI(local_dir=None):  # pragma: no cover
    """Load an example dataset
    instrument: JWST_MIRI
    target: ?

    Data simulated with MIRIsim

    Parameters
    ----------
    local_dir : str, optional
        directory to save data at (default: "./")

    Returns
    -------
    dataset_dir : str
        directory where the data was saved
    """

    return get_dataset("MIRI", local_dir)


def JWST_NIRISS(local_dir=None):  # pragma: no cover
    """Load an example dataset
    instrument: JWST_NIRISS
    target: ?

    Data simulated with awesimsoss

    Parameters
    ----------
    local_dir : str, optional
        directory to save data at (default: "./")

    Returns
    -------
    dataset_dir : str
        directory where the data was saved
    """

    return get_dataset("NIRISS", local_dir)


def KECK_NIRSPEC(local_dir=None):  # pragma: no cover
    """Load an example dataset
    instrument: KECK_NIRSPEC
    target: GJ1214

    Parameters
    ----------
    local_dir : str, optional
        directory to save data at (default: "./")

    Returns
    -------
    dataset_dir : str
        directory where the data was saved
    """

    return get_dataset("NIRSPEC", local_dir)


def XSHOOTER(local_dir=None):  # pragma: no cover
    """Load an example dataset
    instrument: XSHOOTER
    target: Ux-Ori

    Parameters
    ----------
    local_dir : str, optional
        directory to save data at (default: "./")

    Returns
    -------
    dataset_dir : str
        directory where the data was saved
    """

    return get_dataset("XSHOOTER", local_dir)
