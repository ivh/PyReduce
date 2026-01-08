"""Unit tests for trace.py"""

import numpy as np
import pytest

from pyreduce import trace


class TestWhittakerSmooth:
    """Tests for whittaker_smooth function."""

    @pytest.mark.unit
    def test_1d_smoothing(self):
        """Test 1D Whittaker smoothing preserves shape."""
        y = np.random.randn(100)
        result = trace.whittaker_smooth(y, lam=10)

        assert result.shape == y.shape
        assert np.issubdtype(result.dtype, np.floating)

    @pytest.mark.unit
    def test_1d_smoothing_reduces_noise(self):
        """Test that smoothing reduces noise amplitude."""
        np.random.seed(42)
        noise = np.random.randn(100) * 0.5
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        y = signal + noise

        smoothed = trace.whittaker_smooth(y, lam=100)

        # Smoothed signal should have smaller residuals to true signal
        resid_noisy = np.std(y - signal)
        resid_smooth = np.std(smoothed - signal)
        assert resid_smooth < resid_noisy

    @pytest.mark.unit
    def test_2d_smoothing_along_axis(self):
        """Test 2D smoothing along specified axis."""
        y = np.random.randn(10, 100)
        result = trace.whittaker_smooth(y, lam=10, axis=1)

        assert result.shape == y.shape

    @pytest.mark.unit
    def test_higher_lambda_smoother(self):
        """Higher lambda should produce smoother output."""
        np.random.seed(42)
        y = np.random.randn(100)

        smooth_low = trace.whittaker_smooth(y, lam=1)
        smooth_high = trace.whittaker_smooth(y, lam=1000)

        # Higher lambda = smoother = smaller second derivative
        d2_low = np.diff(smooth_low, n=2)
        d2_high = np.diff(smooth_high, n=2)

        assert np.std(d2_high) < np.std(d2_low)


class TestFit:
    """Tests for polynomial fitting functions."""

    @pytest.mark.unit
    def test_fit_linear(self):
        """Test fitting a linear polynomial.

        Note: trace.fit(x, y, deg) fits y -> x, i.e. y is the independent variable.
        """
        y = np.array([0, 1, 2, 3, 4])  # independent variable (column position)
        x = 2 * y + 1  # dependent variable (row position): x = 2*y + 1

        coef = trace.fit(x, y, deg=1)

        assert len(coef) == 2
        # polyval(coef, y) should give x, so coef[0]*y + coef[1] = 2*y + 1
        assert coef[0] == pytest.approx(2, rel=0.01)  # slope
        assert coef[1] == pytest.approx(1, rel=0.01)  # intercept

    @pytest.mark.unit
    def test_fit_quadratic(self):
        """Test fitting a quadratic polynomial."""
        y = np.array([0, 1, 2, 3, 4, 5])  # column position
        x = y**2 + 2 * y + 1  # row position: x = y^2 + 2y + 1

        coef = trace.fit(x, y, deg=2)

        assert len(coef) == 3
        assert coef[0] == pytest.approx(1, rel=0.01)  # y^2 term
        assert coef[1] == pytest.approx(2, rel=0.01)  # y term
        assert coef[2] == pytest.approx(1, rel=0.01)  # constant

    @pytest.mark.unit
    def test_best_fit_selects_appropriate_degree(self):
        """Test that best_fit uses AIC to select degree."""
        x = np.arange(20)
        y = 2 * x + 1 + np.random.randn(20) * 0.1

        coef = trace.best_fit(x, y)

        # For linear data with small noise, should select degree 1
        assert len(coef) <= 3


class TestDetermineOverlapRating:
    """Tests for determine_overlap_rating function."""

    @pytest.mark.unit
    def test_overlapping_clusters(self):
        """Test rating for overlapping clusters."""
        # Two horizontal clusters that overlap
        xi = np.full(100, 50.0)  # y=50
        yi = np.arange(100, 200)  # x from 100 to 200

        xj = np.full(100, 52.0)  # y=52 (close to cluster i)
        yj = np.arange(150, 250)  # x from 150 to 250 (overlaps with i)

        overlap, region = trace.determine_overlap_rating(
            xi, yi, xj, yj, mean_cluster_thickness=10, nrow=100, ncol=300
        )

        assert overlap > 0  # Should have some overlap

    @pytest.mark.unit
    def test_non_overlapping_clusters(self):
        """Test rating for non-overlapping clusters."""
        # Two horizontal clusters that don't overlap
        xi = np.full(100, 50.0)  # y=50
        yi = np.arange(0, 100)  # x from 0 to 100

        xj = np.full(100, 150.0)  # y=150 (far from cluster i)
        yj = np.arange(200, 300)  # x from 200 to 300 (no overlap)

        overlap, region = trace.determine_overlap_rating(
            xi, yi, xj, yj, mean_cluster_thickness=10, nrow=200, ncol=400
        )

        assert overlap < 0.5  # Should have low overlap


class TestCalculateMeanClusterThickness:
    """Tests for calculate_mean_cluster_thickness function."""

    @pytest.mark.unit
    def test_single_cluster(self):
        """Test thickness calculation for single cluster."""
        # Single horizontal cluster with thickness ~10
        x = {1: np.repeat(np.arange(50, 60), 100)}  # 10 rows
        y = {1: np.tile(np.arange(100), 10)}  # 100 columns

        thickness = trace.calculate_mean_cluster_thickness(x, y)

        # Should be around 9 (max - min = 59 - 50 = 9)
        assert 5 < thickness < 20

    @pytest.mark.unit
    def test_empty_clusters(self):
        """Test with no valid clusters."""
        x = {}
        y = {}

        thickness = trace.calculate_mean_cluster_thickness(x, y)

        assert thickness == 10  # Default value


class TestMergeTraces:
    """Tests for merge_traces function."""

    @pytest.mark.unit
    def test_merge_empty_traces(self):
        """Test merging empty trace lists."""
        traces_a = np.array([])
        cr_a = np.array([])
        traces_b = np.array([])
        cr_b = np.array([])

        result = trace.merge_traces(traces_a, cr_a, traces_b, cr_b)

        assert result == ({}, {}, {})

    @pytest.mark.unit
    def test_merge_single_set(self):
        """Test merging when only one set has traces."""
        # Single trace: y = 100 (constant)
        # [0, 0, 100] in polyval order means y = 100
        traces_a = np.array([[0.0, 0.0, 100.0]])
        cr_a = np.array([[0, 1000]])
        traces_b = np.array([]).reshape(0, 3)
        cr_b = np.array([]).reshape(0, 2)

        t_by_o, cr_by_o, fib_by_o = trace.merge_traces(
            traces_a, cr_a, traces_b, cr_b, ncols=1000
        )

        assert 0 in t_by_o
        assert len(t_by_o[0]) == 1

    @pytest.mark.unit
    def test_merge_two_sets(self):
        """Test merging two trace sets."""
        # Even fibers - [0, 0, y] in polyval order means y = constant
        traces_a = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 200.0]])
        cr_a = np.array([[0, 1000], [0, 1000]])

        # Odd fibers
        traces_b = np.array([[0.0, 0.0, 150.0]])
        cr_b = np.array([[0, 1000]])

        t_by_o, cr_by_o, fib_by_o = trace.merge_traces(
            traces_a, cr_a, traces_b, cr_b, ncols=1000
        )

        # All should be in order 0 (no order centers provided)
        assert 0 in t_by_o
        assert len(t_by_o[0]) == 3  # All 3 traces

    @pytest.mark.unit
    def test_merge_with_order_centers(self):
        """Test merging with order assignment based on centers."""
        # Two traces at y=100 and y=300 (constant polynomials)
        # Polynomial coefficients in numpy order: [high_deg, ..., const]
        # So [0, 0, 100] means y = 0*x^2 + 0*x + 100 = 100
        traces_a = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 300.0]])
        cr_a = np.array([[0, 1000], [0, 1000]])
        traces_b = np.array([]).reshape(0, 3)
        cr_b = np.array([]).reshape(0, 2)

        # Order centers at 100 and 300
        order_centers = [100, 300]
        order_numbers = [10, 20]  # Use distinct numbers

        t_by_o, cr_by_o, fib_by_o = trace.merge_traces(
            traces_a,
            cr_a,
            traces_b,
            cr_b,
            order_centers=order_centers,
            order_numbers=order_numbers,
            ncols=1000,
        )

        assert 10 in t_by_o
        assert 20 in t_by_o
        assert len(t_by_o[10]) == 1
        assert len(t_by_o[20]) == 1


class TestGroupAndRefit:
    """Tests for group_and_refit function."""

    @pytest.mark.unit
    def test_group_and_refit_basic(self):
        """Test basic grouping and refitting."""
        # Two traces in order 0 at y=100 and y=110
        # [0, 0, 100] in polyval order means y = 100
        traces_by_order = {0: np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 110.0]])}
        cr_by_order = {0: np.array([[10, 990], [10, 990]])}
        fiber_ids_by_order = {0: np.array([0, 1])}

        groups = {"A": (0, 2)}

        logical_traces, logical_cr, fiber_counts = trace.group_and_refit(
            traces_by_order, cr_by_order, fiber_ids_by_order, groups, degree=2
        )

        assert "A" in logical_traces
        assert len(logical_traces["A"]) == 1  # One order
        assert logical_traces["A"][0].shape == (3,)  # degree 2 + 1 coeffs

        assert 0 in fiber_counts["A"]
        assert fiber_counts["A"][0] == 2  # 2 fibers in group A

    @pytest.mark.unit
    def test_group_and_refit_multiple_groups(self):
        """Test with multiple fiber groups."""
        traces_by_order = {
            0: np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 110.0], [0.0, 0.0, 200.0]])
        }
        cr_by_order = {0: np.array([[10, 990], [10, 990], [10, 990]])}
        fiber_ids_by_order = {0: np.array([0, 1, 5])}

        groups = {"A": (0, 2), "B": (5, 6)}

        logical_traces, logical_cr, fiber_counts = trace.group_and_refit(
            traces_by_order, cr_by_order, fiber_ids_by_order, groups, degree=2
        )

        assert "A" in logical_traces
        assert "B" in logical_traces
        assert fiber_counts["A"][0] == 2
        assert fiber_counts["B"][0] == 1

    @pytest.mark.unit
    def test_group_and_refit_missing_group(self):
        """Test with a group that has no traces."""
        traces_by_order = {0: np.array([[0.0, 0.0, 100.0]])}
        cr_by_order = {0: np.array([[10, 990]])}
        fiber_ids_by_order = {0: np.array([0])}

        groups = {"A": (0, 1), "B": (5, 6)}  # Group B has no traces

        logical_traces, logical_cr, fiber_counts = trace.group_and_refit(
            traces_by_order, cr_by_order, fiber_ids_by_order, groups, degree=2
        )

        assert fiber_counts["B"][0] == 0
        assert np.all(np.isnan(logical_traces["B"][0]))
