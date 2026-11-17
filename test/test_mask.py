import numpy as np
import pytest

pytestmark = [pytest.mark.instrument, pytest.mark.downloads]


def test_load_mask(instr, channel):
    mask_file = instr.get_mask_filename(channel=channel)
    mask, _ = instr.load_fits(mask_file, channel, extension=0)
    mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks

    assert isinstance(mask, np.ndarray)
    assert not np.all(mask)
