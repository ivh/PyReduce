import numpy as np
import pytest

from pyreduce.combine_frames import combine_frames
from pyreduce.slit_curve import Curvature as CurvatureModule

pytestmark = [pytest.mark.instrument, pytest.mark.downloads]


@pytest.fixture
def original(files, instrument, channel, mask):
    if len(files["curvature"]) == 0:
        return None, None

    files = files["curvature"]
    original, chead = combine_frames(files, instrument, channel, mask=mask)

    return original, chead


@pytest.mark.slow
def test_curvature(original, orders, order_range, settings):
    original, chead = original
    orders, column_range = orders
    settings = settings["curvature"]

    if original is None:
        pytest.skip("No curvature files")

    module = CurvatureModule(
        orders,
        column_range=column_range,
        order_range=order_range,
        extraction_height=settings["extraction_height"],
        curve_height=settings.get("curve_height", 0.5),
        window_width=settings["window_width"],
        peak_threshold=settings["peak_threshold"],
        peak_width=settings["peak_width"],
        fit_degree=settings["degree"],
        sigma_cutoff=settings["curvature_cutoff"],
        peak_function=settings["peak_function"],
        mode="1D",
        curve_degree=2,
        plot=False,
        plot_title=None,
    )
    p1, p2 = module.execute(original)

    assert isinstance(p1, np.ndarray)
    assert p1.ndim == 2
    assert p1.shape[0] == order_range[1] - order_range[0]
    assert p1.shape[1] == original.shape[1]

    assert isinstance(p2, np.ndarray)
    assert p2.ndim == 2
    assert p2.shape[0] == order_range[1] - order_range[0]
    assert p2.shape[1] == original.shape[1]

    # Reduce the number of orders this way
    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]

    module = CurvatureModule(
        orders,
        column_range=column_range,
        extraction_height=settings["extraction_height"],
        curve_height=settings.get("curve_height", 0.5),
        window_width=settings["window_width"],
        peak_threshold=settings["peak_threshold"],
        peak_width=settings["peak_width"],
        fit_degree=settings["degree"],
        sigma_cutoff=settings["curvature_cutoff"],
        peak_function=settings["peak_function"],
        mode="2D",
        curve_degree=1,
        plot=False,
        plot_title=None,
    )
    p1, p2 = module.execute(original)

    assert isinstance(p1, np.ndarray)
    assert p1.ndim == 2
    assert p1.shape[0] == order_range[1] - order_range[0]
    assert p1.shape[1] == original.shape[1]

    assert isinstance(p2, np.ndarray)
    assert p2.ndim == 2
    assert p2.shape[0] == order_range[1] - order_range[0]
    assert p2.shape[1] == original.shape[1]


@pytest.mark.slow
def test_curvature_exception(original, orders, order_range):
    original, chead = original
    orders, column_range = orders

    if original is None:
        pytest.skip("No curvature files")

    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]

    original = np.copy(original)

    # Wrong curve_degree input
    with pytest.raises(ValueError):
        module = CurvatureModule(
            orders, column_range=column_range, plot=False, curve_degree=3
        )
        p1, p2 = module.execute(original)

    # Wrong mode
    with pytest.raises(ValueError):
        module = CurvatureModule(
            orders, column_range=column_range, plot=False, mode="3D"
        )
        p1, p2 = module.execute(original)


@pytest.mark.slow
def test_curvature_zero(original, orders, order_range):
    original, chead = original
    orders, column_range = orders

    if original is None:
        pytest.skip("No curvature files")
    orders = orders[order_range[0] : order_range[1]]
    column_range = column_range[order_range[0] : order_range[1]]

    original = np.zeros_like(original)

    # With zero image, should produce zero curvature
    module = CurvatureModule(
        orders, column_range=column_range, plot=False, sigma_cutoff=0
    )
    p1, p2 = module.execute(original)
