from os.path import dirname, join, basename
from glob import glob

import pytest

from pyreduce.instruments import instrument_info
from pyreduce.instruments import common


supported_instruments = glob(join(dirname(__file__), "../pyreduce/instruments/*.json"))
supported_instruments = [basename(f)[:-5] for f in supported_instruments]
supported_instruments = [f for f in supported_instruments if f not in ["common"]]

@pytest.fixture(params=supported_instruments)
def supported_instrument(request):
    return request.param

@pytest.fixture
def supported_modes(supported_instrument):
    return instrument_info.get_supported_modes(supported_instrument)

def test_load_common():
    instr = instrument_info.load_instrument(None)
    assert isinstance(instr, common.instrument)
    assert isinstance(instr, common.COMMON)

def test_load_instrument(supported_instrument):
    instr = instrument_info.load_instrument(supported_instrument)
    assert isinstance(instr, common.instrument)


def test_get_instrument_info(supported_instrument):
    info = instrument_info.get_instrument_info(supported_instrument)
    assert isinstance(info, dict)


def test_modeinfo(supported_instrument, supported_modes):
    for mode in supported_modes:
        header = instrument_info.modeinfo({}, supported_instrument, mode)
        assert isinstance(header, dict)


def test_sort_files():
    files, nights = instrument_info.sort_files(".", "", "", None, "")

    assert isinstance(files, list)
    assert isinstance(nights, list)


def test_get_wavecal_name():
    wname = instrument_info.get_wavecal_filename({}, None, "")
    assert isinstance(wname, str)
