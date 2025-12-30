import numpy as np
import pytest

from pyreduce.combine_frames import combine_frames
from pyreduce.trace import trace

pytestmark = [pytest.mark.instrument, pytest.mark.downloads]


@pytest.mark.slow
def test_orders(instr, instrument, channel, files, settings, mask):
    if len(files["orders"]) == 0:
        pytest.skip(f"No order definition files found for instrument {instrument}")

    order_img, _ = combine_frames(files["orders"], instrument, channel, mask=mask)
    settings = settings["orders"]

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
    with pytest.raises(TypeError):
        trace(img, degree="bla")
    with pytest.raises(ValueError):
        trace(img, degree=-1)
