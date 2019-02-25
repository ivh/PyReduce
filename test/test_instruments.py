import pytest

from pyreduce.instruments import instrument_info
from pyreduce.instruments import common


def test_load_instrument():
    instr = instrument_info.load_instrument(None)

    assert isinstance(instr, common.instrument)
    assert isinstance(instr, common.COMMON)


def test_get_instrument_info():
    info = instrument_info.get_instrument_info(None)

    assert isinstance(info, dict)


def test_modeinfo():
    header = instrument_info.modeinfo({}, None, "")

    assert isinstance(header, dict)


def test_sort_files():
    files, nights = instrument_info.sort_files(".", "", "", None, "")

    assert isinstance(files, list)
    assert isinstance(nights, list)


def test_get_wavecal_name():
    wname = instrument_info.get_wavecal_filename({}, None, "")
    assert isinstance(wname, str)
