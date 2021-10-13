# -*- coding: utf-8 -*-
from glob import glob
from os.path import basename, dirname, exists, join

import pytest

from pyreduce.configuration import get_configuration_for_instrument
from pyreduce.instruments import common, instrument_info

supported_instruments = glob(join(dirname(__file__), "../pyreduce/instruments/*.json"))
supported_instruments = [basename(f)[:-5] for f in supported_instruments]
supported_instruments = [
    f for f in supported_instruments if f not in ["common", "instrument_schema"]
]


@pytest.fixture(params=supported_instruments)
def supported_instrument(request):
    return request.param


@pytest.fixture
def supported_modes(supported_instrument):
    return instrument_info.get_supported_modes(supported_instrument)


@pytest.fixture
def config(supported_instrument):
    return get_configuration_for_instrument(supported_instrument)


def test_load_common():
    instr = instrument_info.load_instrument(None)
    assert isinstance(instr, common.Instrument)
    assert isinstance(instr, common.COMMON)


def test_load_instrument(supported_instrument):
    instr = instrument_info.load_instrument(supported_instrument)
    assert isinstance(instr, common.Instrument)


def test_get_instrument_info(supported_instrument):
    info = instrument_info.get_instrument_info(supported_instrument)
    assert isinstance(info, dict)


def test_modeinfo(supported_instrument, supported_modes):
    # Standard FITS header keywords
    required_keywords = ["e_instrument", "e_telescope", "e_exptime", "e_jd"]
    # PyReduce keywords
    required_keywords += [
        "e_xlo",
        "e_xhi",
        "e_ylo",
        "e_yhi",
        "e_gain",
        "e_readn",
        "e_sky",
        "e_drk",
        "e_backg",
        "e_imtype",
        "e_ctg",
        "e_ra",
        "e_dec",
        "e_obslon",
        "e_obslat",
        "e_obsalt",
    ]
    for mode in supported_modes:
        header = instrument_info.modeinfo({}, supported_instrument, mode)
        assert isinstance(header, dict)
        for key in required_keywords:
            assert key in header.keys()


def test_sort_files(supported_instrument, supported_modes, config):
    for mode in supported_modes:
        files = instrument_info.sort_files(
            ".", "", "", supported_instrument, mode, **config["instrument"]
        )
        print(files)
        assert isinstance(files, list)

        # TODO Test contents of the lists
        # Maybe create debug files for each instrument and then make sure only the correct ones are in each list?
        # That would require a function for each instrument that creates the appropiate files
        # if len(files) != 0:
        # f = files[0][list(files[0].keys())[0]]
        # assert "bias" in f.keys()
        # assert "flat" in f.keys()
        # assert "wavecal" in f.keys()
        # assert "curvature" in f.keys()
        # assert "orders" in f.keys()
        # assert "science" in f.keys()


@pytest.mark.skip(
    reason="No wavelength calibration files for most instruments present at the moment"
)
def test_get_wavecal_name(supported_instrument, supported_modes):
    for mode in supported_modes:
        wname = instrument_info.get_wavecal_filename({}, supported_instrument, mode)
        assert isinstance(wname, str)
        assert exists(wname)
