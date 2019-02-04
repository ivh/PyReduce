import pytest
import numpy as np
from scipy.io import readsav

from pyreduce import util
from pyreduce.extract import extract
from pyreduce.wavelength_calibration import wavecal

from pyreduce import instruments


def test_wavecal(files, instrument, mode, extension, mask, orders, settings):
    orders, column_range = orders
    files = files["wave"][0]
    orig, thead = util.load_fits(files, instrument, mode, extension, mask=mask)
    thead["obase"] = (0, "base order number")

    # Extract wavecal spectrum
    thar, _ = extract(
        orig,
        orders,
        gain=thead["e_gain"],
        readnoise=thead["e_readn"],
        dark=thead["e_drk"],
        extraction_type="arc",
        column_range=column_range,
        extraction_width=settings["wavecal.extraction_width"],
        osample=settings["wavecal.oversampling"],
        plot=False,
    )

    assert isinstance(thar, np.ndarray)
    assert thar.ndim == 2
    assert thar.shape[0] == orders.shape[0]
    assert thar.shape[1] == orig.shape[1]
    assert np.issubdtype(thar.dtype, np.floating)

    # assert np.min(thar) == 0
    # assert np.max(thar) == 1

    reference = instruments.instrument_info.get_wavecal_filename(
        thead, instrument, mode
    )
    reference = readsav(reference)
    cs_lines = reference["cs_lines"]
    wave = wavecal(thar, cs_lines, plot=False, manual=False)

    assert isinstance(wave, np.ndarray)
    assert wave.ndim == 2
    assert wave.shape[0] == orders.shape[0]
    assert wave.shape[1] == orig.shape[1]
    assert np.issubdtype(wave.dtype, np.floating)
