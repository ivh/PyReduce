# -*- coding: utf-8 -*-
import numpy as np
import pytest

from pyreduce.combine_frames import combine_frames
from pyreduce.extract import extract
from pyreduce.make_shear import Curvature as CurvatureModule


@pytest.fixture
def original(files, instrument, mode, mask):
    if len(files["curvature"]) == 0:
        return None, None

    files = files["curvature"]
    original, chead = combine_frames(files, instrument, mode, mask=mask)

    return original, chead


@pytest.fixture
def extracted(original, orders, order_range, settings):
    original, chead = original
    orders, column_range = orders
    settings = settings["curvature"]

    if original is None:
        return None

    extracted, _, _, _ = extract(
        original,
        orders,
        gain=chead["e_gain"],
        readnoise=chead["e_readn"],
        dark=chead["e_drk"],
        extraction_type="arc",
        column_range=column_range,
        order_range=order_range,
        plot=False,
        extraction_width=settings["extraction_width"],
    )
    return extracted


def test_shear(original, extracted, orders, order_range, settings):
    original, chead = original
    orders, column_range = orders
    settings = settings["curvature"]

    if extracted is None:
        pytest.skip("No curvature files")

    module = CurvatureModule(
        orders,
        column_range=column_range,
        order_range=order_range,
        extraction_width=settings["extraction_width"],
        window_width=settings["window_width"],
        peak_threshold=settings["peak_threshold"],
        peak_width=settings["peak_width"],
        fit_degree=settings["degree"],
        sigma_cutoff=settings["curvature_cutoff"],
        peak_function=settings["peak_function"],
        mode="1D",
        curv_degree=2,
        plot=False,
        plot_title=None,
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

    # Reduce the number of orders this way
    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]

    module = CurvatureModule(
        orders,
        column_range=column_range,
        extraction_width=settings["extraction_width"],
        window_width=settings["window_width"],
        peak_threshold=settings["peak_threshold"],
        peak_width=settings["peak_width"],
        fit_degree=settings["degree"],
        sigma_cutoff=settings["curvature_cutoff"],
        peak_function=settings["peak_function"],
        mode="2D",
        curv_degree=1,
        plot=False,
        plot_title=None,
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


def test_shear_exception(original, extracted, orders, order_range):
    original, chead = original
    orders, column_range = orders

    if extracted is None:
        pytest.skip("No curvature files")

    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]

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


def test_shear_zero(original, extracted, orders, order_range):
    original, chead = original
    orders, column_range = orders

    if extracted is None:
        pytest.skip("No curvature files")
    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]

    original = np.zeros_like(original)
    extracted = np.zeros_like(extracted)

    # Reject all possible peaks
    module = CurvatureModule(
        orders, column_range=column_range, plot=False, sigma_cutoff=0
    )
    tilt, shear = module.execute(extracted, original)
