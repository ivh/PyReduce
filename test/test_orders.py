import numpy as np
import pytest

from pyreduce.combine_frames import combine_frames
from pyreduce.trace import trace

pytestmark = [pytest.mark.instrument, pytest.mark.downloads]


@pytest.mark.slow
def test_orders(instr, instrument, channel, files, settings, mask):
    if len(files["trace"]) == 0:
        pytest.skip(f"No order definition files found for instrument {instrument}")

    order_img, _ = combine_frames(files["trace"], instrument, channel, mask=mask)
    settings = settings["trace"]

    orders, column_range = trace(
        order_img,
        min_cluster=settings["min_cluster"],
        min_width=settings["min_width"],
        filter_x=settings.get("filter_x", 0),
        filter_y=settings["filter_y"],
        noise=settings["noise"],
        degree=settings["degree"],
        degree_before_merge=settings["degree_before_merge"],
        regularization=settings["regularization"],
        closing_shape=settings["closing_shape"],
        opening_shape=settings.get("opening_shape", None),
        border_width=settings["border_width"],
        manual=False,
        auto_merge_threshold=settings["auto_merge_threshold"],
        merge_min_threshold=settings["merge_min_threshold"],
        sigma=settings["split_sigma"],
        plot=False,
    )

    assert isinstance(orders, np.ndarray)
    assert np.issubdtype(orders.dtype, np.floating)
    assert orders.shape[1] == settings["degree"] + 1

    assert isinstance(column_range, np.ndarray)
    assert np.issubdtype(column_range.dtype, np.integer)
    assert column_range.shape[1] == 2
    assert np.all(column_range >= 0)
    assert np.all(column_range <= order_img.shape[1])

    assert orders.shape[0] == column_range.shape[0]


def test_simple():
    img = np.full((100, 100), 1)
    img[45:56, :] = 100

    orders, column_range = trace(
        img, manual=False, degree=1, plot=False, border_width=0
    )

    assert orders.shape[0] == 1
    assert np.allclose(orders[0], [0, 50])

    assert column_range.shape[0] == 1
    assert column_range[0, 0] == 0
    assert column_range[0, 1] == 100


def test_per_side_border_width():
    """Test per-side border_width as [top, bottom, left, right]."""
    img = np.full((100, 100), 1)
    # Two traces: one at top, one in middle
    img[10:15, :] = 100  # near top edge
    img[45:56, :] = 100  # middle

    # With symmetric border_width=20, top trace should be excluded
    orders, _ = trace(img, manual=False, degree=1, plot=False, border_width=20)
    assert orders.shape[0] == 1  # only middle trace

    # With per-side [5, 0, 0, 0], top trace still excluded (y=10-15 masked by top=5? no wait)
    # Actually top=5 masks rows 0-5, trace at 10-15 should survive
    # Let's use top=20 to exclude trace at y=10-15
    orders, _ = trace(
        img, manual=False, degree=1, plot=False, border_width=[20, 0, 0, 0]
    )
    assert orders.shape[0] == 1  # top trace excluded

    # With per-side [0, 0, 0, 0], both traces should be found
    orders, _ = trace(
        img, manual=False, degree=1, plot=False, border_width=[0, 0, 0, 0]
    )
    assert orders.shape[0] == 2  # both traces found


def test_parameters():
    img = np.full((100, 100), 1)
    img[45:56, :] = 100

    with pytest.raises(TypeError):
        trace(None)
    with pytest.raises(TypeError):
        trace(img, min_cluster="bla")
    with pytest.raises(TypeError):
        trace(img, filter_y="bla")
    with pytest.raises(ValueError):
        trace(img, filter_y=0)
    with pytest.raises(TypeError):
        trace(img, noise="bla")
    with pytest.raises(TypeError):
        trace(img, border_width="bla")
    with pytest.raises(ValueError):
        trace(img, border_width=-1)
    with pytest.raises(ValueError):
        trace(img, border_width=[10, 10, 10])  # wrong length
    with pytest.raises(ValueError):
        trace(img, border_width=[10, -1, 10, 10])  # negative value
    with pytest.raises(TypeError):
        trace(img, degree="bla")
    with pytest.raises(ValueError):
        trace(img, degree=-1)
