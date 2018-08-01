import importlib

from . import uves, harps

def load_instrument(instrument):
    """ load the instrument module """

    instruments = {"uves": uves.UVES, "harps": harps.HARPS}
    instrument = instruments[instrument.lower()]
    instrument = instrument()

    # fname = "instruments.%s" % instrument.lower()
    # lib = importlib.import_module(fname)
    # instrument = getattr(lib, instrument.upper())
    # instrument = instrument()

    return instrument


def get_instrument_info(instrument):
    instrument = load_instrument(instrument)
    return instrument.load_info()


def sort_files(files, target, night, instrument, mode, **kwargs):
    instrument = load_instrument(instrument)
    return instrument.sort_files(files, target, night, mode, **kwargs)


def modeinfo(header, instrument, mode, **kwargs):
    instrument = load_instrument(instrument)
    header = instrument.add_header_info(header, mode, **kwargs)
    return header


def get_wavecal_filename(header, instrument, mode, **kwargs):
    instrument = load_instrument(instrument)
    return instrument.get_wavecal_filename(header, mode, **kwargs)

