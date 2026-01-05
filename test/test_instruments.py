from glob import glob
from os.path import basename, dirname, exists, join

import pytest
import yaml

from pyreduce.configuration import get_configuration_for_instrument
from pyreduce.instruments import common, instrument_info
from pyreduce.instruments.models import validate_instrument_config

supported_instruments = glob(join(dirname(__file__), "../pyreduce/instruments/*.yaml"))
supported_instruments = [basename(f)[:-5] for f in supported_instruments]
supported_instruments = [f for f in supported_instruments if f not in ["common"]]


@pytest.fixture(params=supported_instruments)
def supported_instrument(request):
    return request.param


@pytest.fixture
def supported_channels(supported_instrument):
    return instrument_info.get_supported_channels(supported_instrument)


@pytest.fixture
def config(supported_instrument):
    return get_configuration_for_instrument(supported_instrument)


@pytest.mark.unit
def test_instrument_yaml_pydantic_valid(supported_instrument):
    """Validate instrument YAML files with Pydantic."""
    instrument_path = join(
        dirname(__file__), f"../pyreduce/instruments/{supported_instrument}.yaml"
    )
    with open(instrument_path) as f:
        instrument_data = yaml.safe_load(f)

    # This will raise ValidationError if invalid
    config = validate_instrument_config(instrument_data)
    assert config.instrument is not None


@pytest.mark.unit
def test_load_common():
    instr = instrument_info.load_instrument(None)
    assert isinstance(instr, common.Instrument)
    assert isinstance(instr, common.COMMON)


@pytest.mark.unit
def test_load_instrument(supported_instrument):
    instr = instrument_info.load_instrument(supported_instrument)
    assert isinstance(instr, common.Instrument)


@pytest.mark.unit
def test_get_instrument_info(supported_instrument):
    info = instrument_info.get_instrument_info(supported_instrument)
    assert isinstance(info, dict)


@pytest.mark.unit
def test_channelinfo(supported_instrument, supported_channels):
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
    for channel in supported_channels:
        header = instrument_info.channelinfo({}, supported_instrument, channel)
        assert isinstance(header, dict)
        for key in required_keywords:
            assert key in header.keys()


@pytest.mark.unit
def test_sort_files(supported_instrument, supported_channels, config):
    for channel in supported_channels:
        files = instrument_info.sort_files(
            ".", "", "", supported_instrument, channel, **config["instrument"]
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


