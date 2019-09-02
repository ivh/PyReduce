import pytest
import numpy as np

from pyreduce import util


def test_extend_orders():
    # Test normal case
    orders = np.array([[0.1, 5], [0.1, 7]])
    extended = util.extend_orders(orders, 10)

    assert np.array_equal(orders, extended[1:-1])
    assert np.array_equal(extended[0], [0.1, 3])
    assert np.array_equal(extended[-1], [0.1, 9])

    # Test just one order
    orders = np.array([0.1, 5], ndmin=2)
    extended = util.extend_orders(orders, 10)

    assert np.array_equal(orders, extended[1:-1])
    assert np.array_equal(extended[0], [0, 0])
    assert np.array_equal(extended[-1], [0, 10])


def test_fix_column_range():
    # Some orders will be shortened
    nrow, ncol = 50, 1000
    orders = np.array([[0.2, 3], [0.2, 5], [0.2, 7], [0.2, 9]])
    ew = np.array([[10, 10], [10, 10], [10, 10], [10, 10]])
    cr = np.array([[0, 1000], [0, 1000], [0, 1000], [0, 1000]])

    fixed = util.fix_column_range(cr, orders, ew, nrow, ncol)

    assert np.array_equal(fixed[1], [25, 175])
    assert np.array_equal(fixed[2], [15, 165])
    assert np.array_equal(fixed[0], fixed[1])
    assert np.array_equal(fixed[-1], fixed[-1])

    # Nothing should change here
    orders = np.array([[20], [20], [20]])
    ew = np.array([[10, 10], [10, 10], [10, 10]])
    cr = np.array([[0, 1000], [0, 1000], [0, 1000]])

    fixed = util.fix_column_range(np.copy(cr), orders, ew, nrow, ncol)
    assert np.array_equal(fixed, cr)
