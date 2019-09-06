import pytest
import numpy as np

from pyreduce.util import load_fits
from pyreduce.combine_frames import combine_frames
from pyreduce.make_shear import Curvature as CurvatureModule


def test_shear(files, wave, orders, instrument, mode, extension, mask, order_range):
    if len(files["curvature"]) == 0:
        pytest.skip(f"No curvature files found for instrument {instrument}")

    _, extracted = wave
    orders, column_range = orders
    files = files["curvature"]

    original, chead = combine_frames(files, instrument, mode, extension, mask=mask)

    module = CurvatureModule(
        orders,
        column_range=column_range,
        order_range=order_range,
        plot=False,
        mode="1D",
        curv_degree=2,
    )
    tilt, shear = module.execute(extracted, original)

    assert isinstance(tilt, np.ndarray)
    assert tilt.ndim == 2
    assert tilt.shape[0] == order_range[1] - order_range[0]
    assert tilt.shape[1] == extracted.shape[1]

    assert isinstance(shear, np.ndarray)
    assert shear.ndim == 2
    assert shear.shape[0] == order_range[1] - order_range[0]
    assert shear.shape[1] == extracted.shape[1]

    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]

    module = CurvatureModule(
        orders, column_range=column_range, plot=False, curv_degree=1, mode="2D"
    )
    tilt, shear = module.execute(extracted, original)

    assert isinstance(tilt, np.ndarray)
    assert tilt.ndim == 2
    assert tilt.shape[0] == order_range[1] - order_range[0]
    assert tilt.shape[1] == extracted.shape[1]

    assert isinstance(shear, np.ndarray)
    assert shear.ndim == 2
    assert shear.shape[0] == order_range[1] - order_range[0]
    assert shear.shape[1] == extracted.shape[1]


def test_shear_exception(
    files, wave, orders, instrument, mode, extension, mask, order_range
):
    if len(files["curvature"]) == 0:
        pytest.skip(f"No curvature files found for instrument {instrument}")

    _, extracted = wave
    orders, column_range = orders
    files = files["curvature"]
    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]

    original, chead = combine_frames(files, instrument, mode, extension, mask=mask)
    original = np.copy(original)

    # Wrong curv_degree input
    with pytest.raises(ValueError):
        module = CurvatureModule(
            orders, column_range=column_range, plot=False, curv_degree=3
        )
        tilt, shear = module.execute(extracted, original)

    # Wrong mode
    with pytest.raises(ValueError):
        module = CurvatureModule(
            orders, column_range=column_range, plot=False, mode="3D"
        )
        tilt, shear = module.execute(extracted, original)


def test_shear_nopeaks(
    files, wave, orders, instrument, mode, extension, mask, order_range
):
    if len(files["curvature"]) == 0:
        pytest.skip(f"No curvature files found for instrument {instrument}")

    _, extracted = wave
    orders, column_range = orders
    files = files["curvature"]
    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]

    original, chead = combine_frames(files, instrument, mode, extension, mask=mask)
    original = np.copy(original)

    # Reject all possible peaks
    module = CurvatureModule(
        orders, column_range=column_range, plot=False, max_iter=None, sigma_cutoff=0
    )
    tilt, shear = module.execute(extracted, original)


def test_shear_zero(
    files, wave, orders, instrument, mode, extension, mask, order_range
):
    if len(files["curvature"]) == 0:
        pytest.skip(f"No curvature files found for instrument {instrument}")

    _, extracted = wave
    orders, column_range = orders
    files = files["curvature"]
    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]

    files = files[0]
    original, thead = load_fits(files, instrument, mode, extension, mask=mask)

    original = np.zeros_like(original)
    extracted = np.zeros_like(extracted)

    # Reject all possible peaks
    module = CurvatureModule(
        orders, column_range=column_range, plot=False, max_iter=None, sigma_cutoff=0
    )
    tilt, shear = module.execute(extracted, original)
