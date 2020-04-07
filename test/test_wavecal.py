import pytest
import numpy as np

from pyreduce import util
from pyreduce.extract import extract
from pyreduce.wavelength_calibration import WavelengthCalibration

from pyreduce import instruments


def test_wavecal(files, instr, instrument, mode, mask, orders, settings, order_range):
    if len(files["wavecal"]) == 0:
        pytest.skip(f"No wavecal files found for instrument {instrument}")

    orders, column_range = orders
    files = files["wavecal"][0]
    orig, thead = instr.load_fits(files, mode, mask=mask)
    thead["obase"] = (0, "base order number")

    # Extract wavecal spectrum
    thar, _, _, _ = extract(
        orig,
        orders,
        gain=thead["e_gain"],
        readnoise=thead["e_readn"],
        dark=thead["e_drk"],
        extraction_type="arc",
        column_range=column_range,
        order_range=order_range,
        extraction_width=settings["wavecal"]["extraction_width"],
        plot=False,
    )

    assert isinstance(thar, np.ndarray)
    assert thar.ndim == 2
    assert thar.shape[0] == order_range[1] - order_range[0]
    assert thar.shape[1] == orig.shape[1]
    assert np.issubdtype(thar.dtype, np.floating)

    # assert np.min(thar) == 0
    # assert np.max(thar) == 1

    reference = instr.get_wavecal_filename(thead, mode, **settings["instrument"])
    reference = np.load(reference, allow_pickle=True)
    linelist = reference["cs_lines"]

    module = WavelengthCalibration(
        plot=False,
        manual=False,
        threshold=settings["wavecal"]["threshold"],
        degree=settings["wavecal"]["degree"],
    )
    wave, solution = module.execute(thar, linelist)

    assert isinstance(wave, np.ndarray)
    assert wave.ndim == 2
    assert wave.shape[0] == order_range[1] - order_range[0]
    assert wave.shape[1] == orig.shape[1]
    assert np.issubdtype(wave.dtype, np.floating)
