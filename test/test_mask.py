import os

import numpy as np
import pytest

pytestmark = [pytest.mark.instrument, pytest.mark.downloads]


@pytest.fixture
def mask_dir():
    mask_dir = os.path.dirname(__file__)
    mask_dir = os.path.join(mask_dir, "../pyreduce", "masks")
    return mask_dir


def test_load_mask(instr, arm, mask_dir):
    # mask_file = join(
    #     mask_dir, "mask_{}_{}.fits.gz".format(instr.name.lower(), arm.lower())
    # )
    mask_file = instr.get_mask_filename(arm=arm)
    mask, _ = instr.load_fits(mask_file, arm, extension=0)
    mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks

    assert isinstance(mask, np.ndarray)
    assert not np.all(mask)
