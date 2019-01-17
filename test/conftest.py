import os
from os.path import dirname, join

import json
import pytest
from pyreduce import datasets, util, instruments

# TODO Add more datasets
@pytest.fixture(params=[("UVES", "HD132205")], ids=["UVES_HD132205"])
def dataset(request):
    return request.param


@pytest.fixture
def instrument(dataset):
    return dataset[0]


@pytest.fixture
def target(dataset):
    return dataset[1]


@pytest.fixture
def night(dataset):
    instr, target = dataset
    if target == "HD132205":
        return "2010-04-02"


@pytest.fixture
def mode(dataset):
    instrument, target = dataset
    if instrument == "UVES":
        return "middle"


@pytest.fixture
def info(instrument):
    i = instruments.instrument_info.get_instrument_info(instrument)
    return i


@pytest.fixture
def extension(info, mode):
    imode = util.find_first_index(info["modes"], mode)
    ext = info["extension"][imode]
    return ext


@pytest.fixture
def data(dataset):
    instrument, target = dataset
    folder = dirname(__file__)
    if instrument == "UVES" and target == "HD132205":
        folder = datasets.UVES_HD132205(folder)
    return folder


@pytest.fixture
def config(dataset):
    instrument, target = dataset
    folder = dirname(__file__)
    filename = join(folder, "settings", f"settings_{instrument.upper()}.json")

    with open(filename) as f:
        conf = json.load(f)

    return conf


@pytest.fixture
def settings(config):
    setti = util.read_config()
    setti.update(config)
    return setti


@pytest.fixture
def files(data, instrument, target, night, mode, settings):
    input_dir = join(data, "datasets", target)
    files, _ = instruments.instrument_info.sort_files(
        input_dir, target, night, instrument, mode, **settings
    )
    files = files[0][list(files[0].keys())[0]]
    return files

