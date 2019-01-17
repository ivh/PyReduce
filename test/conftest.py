import os
from os.path import dirname, join

import numpy as np
from astropy.io import fits

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
def input_dir(data, target):
    return join(data, "datasets", target)


@pytest.fixture
def output_dir(data, settings):
    return join(data, settings["reduce.output_dir"])


@pytest.fixture
def files(input_dir, instrument, target, night, mode, settings):
    files, _ = instruments.instrument_info.sort_files(
        input_dir, target, night, instrument, mode, **settings
    )
    files = files[0][list(files[0].keys())[0]]
    return files


@pytest.fixture
def mask(instrument, mode):
    mask_dir = os.path.dirname(__file__)
    mask_dir = os.path.join(mask_dir, "../pyreduce", "masks")
    mask_file = join(mask_dir, "mask_%s_%s.fits.gz" % (instrument.lower(), mode))

    mask, _ = util.load_fits(mask_file, instrument, mode, extension=0)
    mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks
    return mask


@pytest.fixture
def prefix(instrument, mode):
    prefix = "%s_%s" % (instrument.lower(), mode.lower())
    return prefix


@pytest.fixture
def flat(prefix, output_dir, mask):
    flat_file = join(output_dir, f"{prefix}.flat.fits")
    flat = fits.open(flat_file)[0]
    flat, fhead = flat.data, flat.header
    flat = np.ma.masked_array(flat, mask=mask)
    return flat, fhead


@pytest.fixture
def bias(prefix, output_dir, mask):
    bias_file = join(output_dir, f"{prefix}.bias.fits")
    bias = fits.open(bias_file)[0]
    bias, bhead = bias.data, bias.header
    bias = np.ma.masked_array(bias, mask=mask)
    return bias, bhead
