import pytest
import os
from astropy.io import fits
import datetime as dt
import dateutil

from pyreduce.instruments import uves, common, instrument_info


@pytest.fixture
def header(instrument, input_dir):
    if instrument != "UVES":
        return None

    files = [
        f
        for f in os.listdir(input_dir)
        if (f.endswith(".fits") or f.endswith(".fits.gz"))
    ]
    hdu = fits.open(os.path.join(input_dir, files[0]))
    header = hdu[0].header
    return header


def test_load_instrument():
    instr = instrument_info.load_instrument("UVES")

    assert isinstance(instr, common.instrument)
    assert isinstance(instr, uves.UVES)


def test_load_info():
    info = uves.UVES().load_info()
    assert isinstance(info, dict)


def test_modeinfo(header, mode):
    if header is None:
        return

    new = uves.UVES().add_header_info(header, mode)

    assert isinstance(new, fits.Header)
    assert "e_xlo" in new
    assert "e_xhi" in new
    assert "e_ylo" in new
    assert "e_yhi" in new
    assert "e_gain" in new
    assert "e_readn" in new
    assert "e_exptim" in new
    assert "e_sky" in new
    assert "e_drk" in new
    assert "e_backg" in new
    assert "e_imtype" in new
    assert "e_ctg" in new
    assert "e_ra" in new
    assert "e_dec" in new
    assert "e_jd" in new
    assert "e_obslon" in new
    assert "e_obslat" in new
    assert "e_obsalt" in new
    assert "e_pol" in new


def test_wavecalfile(header, mode):
    if header is None:
        return

    wname = uves.UVES().get_wavecal_filename(header, mode)
    assert isinstance(wname, str)
    assert os.path.exists(wname)


def test_sort_files(instrument, target, night, mode, input_dir):
    if instrument != "UVES":
        return None

    files, nights = uves.UVES().sort_files(input_dir, target, night, mode)

    assert isinstance(files, list)
    assert isinstance(files[0], dict)
    assert isinstance(files[0][list(files[0].keys())[0]], dict)
    assert isinstance(nights, list)
    assert isinstance(nights[0], dt.date)

    assert len(files) == len(nights)

    f = files[0][list(files[0].keys())[0]]
    assert "bias" in f.keys()
    assert "flat" in f.keys()
    assert "wavecal" in f.keys()
    assert "curvature" in f.keys()
    assert "orders" in f.keys()
    assert "science" in f.keys()

    assert len(f["bias"]) != 0
    assert len(f["flat"]) != 0
    assert len(f["wavecal"]) != 0
    assert len(f["curvature"]) != 0
    assert len(f["orders"]) != 0
    assert len(f["science"]) != 0

