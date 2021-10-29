# -*- coding: utf-8 -*-
import os
from os.path import join

import numpy as np
import pytest

from pyreduce import util


@pytest.fixture
def mask_dir():
    mask_dir = os.path.dirname(__file__)
    mask_dir = os.path.join(mask_dir, "../pyreduce", "masks")
    return mask_dir


def test_load_mask(instr, mode, mask_dir):
    # mask_file = join(
    #     mask_dir, "mask_{}_{}.fits.gz".format(instr.name.lower(), mode.lower())
    # )
    mask_file = instr.get_mask_filename(mode=mode)
    mask, _ = instr.load_fits(mask_file, mode, extension=0)
    mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks

    assert isinstance(mask, np.ndarray)
    assert not np.all(mask)
