import pytest
import numpy as np

import os
from os.path import join

from pyreduce import util


@pytest.fixture
def mask_dir():
    mask_dir = os.path.dirname(__file__)
    mask_dir = os.path.join(mask_dir, "../pyreduce", "masks")
    return mask_dir


def test_load_mask(instrument, mode, mask_dir):
    mask_file = join(mask_dir, "mask_%s_%s.fits.gz" % (instrument.lower(), mode.lower()))
    mask, _ = instrument.load_fits(mask_file, "MASK")
    mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks

    assert isinstance(mask, np.ndarray)
    assert not np.all(mask)
