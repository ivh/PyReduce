# -*- coding: utf-8 -*-
"""
Interface for all instrument specific information
The actual info is contained in the instruments/{name}.py modules/classes, which are all subclasses of "common"
"""

import importlib

from .common import Instrument


def load_instrument(instrument) -> Instrument:
    """Load an python instrument module

    Parameters
    ----------
    instrument : str
        name of the instrument

    Returns
    -------
    instrument : Instrument
        Instance of the {instrument} class
    """

    # TODO: Loading arbitrary modules is probably bad style
    # from instruments import uves, harps
    # instruments = {"uves": uves.UVES, "harps": harps.HARPS}
    # instrument = instruments[instrument.lower()]
    # instrument = instrument()
    if instrument is None:
        instrument = "common"

    fname = ".instruments.%s" % instrument.lower()
    lib = importlib.import_module(fname, package="pyreduce")
    instrument = getattr(lib, instrument.upper())
    instrument = instrument()

    return instrument


def get_instrument_info(instrument):
    """Load instrument specific information

    Parameters
    ----------
    instrument : str
        Name of the instrument

    Returns
    -------
    dict{str:obj}
        Dictionary with information
    """

    instrument = load_instrument(instrument)
    return instrument.info


def sort_files(input_dir, target, night, instrument, mode, **kwargs):
    """Sort a list of files into different categories and discard files that are not used

    Parameters
    ----------
    input_dir : str
        directory containing all files (with tags for target, night, and instrument)
    target : str
        observation target name, as found in the files
    night : str
        observation night of interest, as found in the files
    instrument : str
        instrument name
    mode : str
        instrument mode, if applicable (e.g. red/blue for HARPS)

    Returns
    -------
    biaslist : list(str)
        list of bias files
    flatlist : list(str)
        list of flat field files
    wavelist : list(str)
        list of wavelength calibration files
    orderlist : list(str)
        list of order definition files (for order tracing)
    speclist : list(str)
        list of science files, i.e. observations
    """

    instrument = load_instrument(instrument)
    return instrument.sort_files(input_dir, target, night, mode, **kwargs)


def get_supported_modes(instrument):
    instrument = load_instrument(instrument)
    return instrument.get_supported_modes()


def modeinfo(header, instrument, mode, **kwargs):
    """Add instrument specific information to a header/dict

    Parameters
    ----------
    header : fits.header, dict
        header to add information to
    instrument : str
        instrument name
    mode : str
        instrument mode (e.g. red/blue for HARPS)

    Returns
    -------
    header
        header with added information
    """

    instrument = load_instrument(instrument)
    header = instrument.add_header_info(header, mode, **kwargs)
    return header


def get_wavecal_filename(header, instrument, mode, **kwargs):
    """Get the filename of the pre-existing wavelength solution for the current settings

    Parameters
    ----------
    header : fits.header, dict
        header of the wavelength calibration file
    instrument : str
        instrument name
    mode : str
        instrument mode (e.g. red/blue for HARPS)

    Returns
    -------
    filename : str
        wavelength solution file
    """

    instrument = load_instrument(instrument)
    return instrument.get_wavecal_filename(header, mode, **kwargs)
