import importlib

def load_instrument(instrument):
    """ load the instrument module """

    fname = "instruments.%s" % instrument.lower()
    lib = importlib.import_module(fname)
    instrument = getattr(lib, instrument.upper())
    instrument = instrument()

    return instrument

def get_instrument_info(instrument):
    instrument = load_instrument(instrument)
    return instrument.load_info()

def sort_files(files, target, instrument, mode):
    instrument = load_instrument(instrument)
    return instrument.sort_files(files, target, mode)

def modeinfo(header, instrument, mode):
    instrument = load_instrument(instrument)
    header = instrument.add_header_info(header, mode)
    return header
