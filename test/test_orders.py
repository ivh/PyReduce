import pytest

import numpy as np

from pyreduce import util
from pyreduce.trace_orders import mark_orders


def test_orders(instrument, mode, extension, files, settings, mask):
    """
    Test the order tracing for each test dataset

    Parameters
    ----------
    instrument : str
        Instrument Name
    mode : str
        Current Mode
    extension : int
        fits extension for that mode
    files : dict(str:str)
        dict of files, sorted by type
    settings : dict(str:obj)
        configuration settings of this run
    mask : array(bool) of size (n,)
        Bad pixel mask
    """

    files = files["order"][0]
    order_img, _ = util.load_fits(files, instrument, mode, extension, mask=mask)

    orders, column_range = mark_orders(
        order_img,
        min_cluster=settings["orders.min_cluster"],
        filter_size=settings["orders.filter_size"],
        noise=settings["orders.noise"],
        opower=settings["orders.fit_degree"],
        border_width=settings["orders.border_width"],
        manual=False,
        plot=False,
    )

    assert isinstance(orders, np.ndarray)
    assert np.issubdtype(orders.dtype, np.float)
    assert orders.shape[1] == settings["orders.fit_degree"] + 1

    assert isinstance(column_range, np.ndarray)
    assert np.issubdtype(column_range.dtype, np.integer)
    assert column_range.shape[1] == 2
    assert np.all(column_range >= 0)
    assert np.all(column_range <= order_img.shape[1])

    assert orders.shape[0] == column_range.shape[0]
