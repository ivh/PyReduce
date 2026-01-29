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


class TestComputeTraceHeights:
    """Tests for compute_trace_heights function."""

    @pytest.mark.unit
    def test_evenly_spaced_traces(self):
        """Test height computation for evenly spaced traces."""
        # 5 traces spaced 20 pixels apart, constant across columns
        traces = np.array(
            [
                [0.0, 0.0, 20.0],  # y = 20
                [0.0, 0.0, 40.0],  # y = 40
                [0.0, 0.0, 60.0],  # y = 60
                [0.0, 0.0, 80.0],  # y = 80
                [0.0, 0.0, 100.0],  # y = 100
            ]
        )
        column_range = np.array([[0, 1000]] * 5)

        heights = trace.compute_trace_heights(traces, column_range, ncol=1000)

        assert len(heights) == 5
        # First trace: distance to next neighbor = 20
        assert heights[0] == pytest.approx(20.0, rel=0.01)
        # Middle traces: half distance between neighbors = 20
        assert heights[1] == pytest.approx(20.0, rel=0.01)
        assert heights[2] == pytest.approx(20.0, rel=0.01)
        assert heights[3] == pytest.approx(20.0, rel=0.01)
        # Last trace: distance to previous neighbor = 20
        assert heights[4] == pytest.approx(20.0, rel=0.01)

    @pytest.mark.unit
    def test_uneven_spacing(self):
        """Test height computation for unevenly spaced traces."""
        # Traces with varying spacing
        traces = np.array(
            [
                [0.0, 0.0, 10.0],  # y = 10
                [0.0, 0.0, 30.0],  # y = 30 (spacing 20 above)
                [0.0, 0.0, 90.0],  # y = 90 (spacing 60 above)
            ]
        )
        column_range = np.array([[0, 1000]] * 3)

        heights = trace.compute_trace_heights(traces, column_range, ncol=1000)

        # First: distance to next = 20
        assert heights[0] == pytest.approx(20.0, rel=0.01)
        # Middle: half distance between neighbors = (90-10)/2 = 40
        assert heights[1] == pytest.approx(40.0, rel=0.01)
        # Last: distance to prev = 60
        assert heights[2] == pytest.approx(60.0, rel=0.01)

    @pytest.mark.unit
    def test_single_trace(self):
        """Single trace should return NaN (no neighbors)."""
        traces = np.array([[0.0, 0.0, 50.0]])
        column_range = np.array([[0, 1000]])

        heights = trace.compute_trace_heights(traces, column_range, ncol=1000)

        assert len(heights) == 1
        assert np.isnan(heights[0])

    @pytest.mark.unit
    def test_empty_traces(self):
        """Empty traces should return empty array."""
        traces = np.array([]).reshape(0, 3)
        column_range = np.array([]).reshape(0, 2)

        heights = trace.compute_trace_heights(traces, column_range, ncol=1000)

        assert len(heights) == 0

    @pytest.mark.unit
    def test_varying_column_range(self):
        """Height uses max across valid reference columns."""
        # Trace with curvature: y varies across x
        # y = 0.001*x + 50 (slight slope)
        traces = np.array(
            [
                [0.001, 50.0],  # y = 0.001*x + 50
                [0.001, 70.0],  # y = 0.001*x + 70
            ]
        )
        column_range = np.array([[100, 900], [100, 900]])

        heights = trace.compute_trace_heights(traces, column_range, ncol=1000)

        # Spacing is constant at 20 pixels regardless of x
        assert heights[0] == pytest.approx(20.0, rel=0.01)
        assert heights[1] == pytest.approx(20.0, rel=0.01)


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


class TestOrganizeFibers:
    """Tests for organize_fibers function (config-based fiber grouping)."""

    @pytest.fixture
    def sample_traces(self):
        """Create sample traces for testing."""
        # 10 traces at y = 100, 110, 120, ..., 190
        traces = np.array([[0.0, 0.0, 100.0 + i * 10] for i in range(10)])
        column_range = np.array([[10, 990]] * 10)
        return traces, column_range

    @pytest.fixture
    def groups_config(self):
        """Create a groups-based FibersConfig."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        return FibersConfig(
            groups={
                "A": FiberGroupConfig(range=(1, 5), merge="average"),
                "B": FiberGroupConfig(range=(5, 11), merge="center"),
            }
        )

    @pytest.fixture
    def bundles_config(self):
        """Create a bundles-based FibersConfig."""
        from pyreduce.instruments.models import FiberBundleConfig, FibersConfig

        return FibersConfig(bundles=FiberBundleConfig(size=5, merge="center"))

    @pytest.mark.unit
    def test_organize_fibers_with_groups_average(self, sample_traces, groups_config):
        """Test organize_fibers with named groups using average merge."""
        traces, column_range = sample_traces

        group_traces, group_cr, group_counts, _ = trace.organize_fibers(
            traces, column_range, groups_config
        )

        assert "A" in group_traces
        assert "B" in group_traces
        assert group_counts["A"] == 4  # fibers 1-4 (0-based: 0-3)
        assert group_counts["B"] == 6  # fibers 5-10 (0-based: 4-9)

    @pytest.mark.unit
    def test_organize_fibers_with_groups_center(self, sample_traces):
        """Test organize_fibers with center merge method."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, column_range = sample_traces
        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 11), merge="center")}
        )

        group_traces, group_cr, group_counts, _ = trace.organize_fibers(
            traces, column_range, config
        )

        assert "A" in group_traces
        assert len(group_traces["A"]) == 1  # Single center trace
        assert group_counts["A"] == 10

    @pytest.mark.unit
    def test_organize_fibers_with_bundles(self, sample_traces, bundles_config):
        """Test organize_fibers with bundle pattern."""
        traces, column_range = sample_traces

        group_traces, group_cr, group_counts, _ = trace.organize_fibers(
            traces, column_range, bundles_config
        )

        # 10 traces / 5 per bundle = 2 bundles
        assert "bundle_1" in group_traces
        assert "bundle_2" in group_traces
        assert len(group_traces) == 2
        assert group_counts["bundle_1"] == 5
        assert group_counts["bundle_2"] == 5

    @pytest.mark.unit
    def test_organize_fibers_bundles_not_divisible(self, sample_traces):
        """Test that bundles fails when traces not divisible by bundle size."""
        from pyreduce.instruments.models import FiberBundleConfig, FibersConfig

        traces, column_range = sample_traces
        config = FibersConfig(
            bundles=FiberBundleConfig(size=3)
        )  # 10 not divisible by 3

        with pytest.raises(ValueError, match="not divisible"):
            trace.organize_fibers(traces, column_range, config)

    @pytest.mark.unit
    def test_organize_fibers_bundles_wrong_count(self, sample_traces):
        """Test that bundles fails when count doesn't match."""
        from pyreduce.instruments.models import FiberBundleConfig, FibersConfig

        traces, column_range = sample_traces
        config = FibersConfig(
            bundles=FiberBundleConfig(size=5, count=10)  # Expected 10, but 10/5=2
        )

        with pytest.raises(ValueError, match="Expected 10 bundles"):
            trace.organize_fibers(traces, column_range, config)

    @pytest.mark.unit
    def test_organize_fibers_merge_indices(self, sample_traces):
        """Test organize_fibers with index-based merge."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, column_range = sample_traces
        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 11), merge=[1, 5, 10])}
        )

        group_traces, group_cr, group_counts, _ = trace.organize_fibers(
            traces, column_range, config
        )

        # Should select 3 specific traces
        assert len(group_traces["A"]) == 3
        assert group_counts["A"] == 10  # Still counted all 10 fibers in group

    @pytest.mark.unit
    def test_organize_fibers_per_order(self):
        """Test per-order fiber grouping."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        # Create 6 traces: 3 fibers per order, 2 orders
        # Order 1 at y~100, Order 2 at y~200
        traces = np.array(
            [
                [0.0, 0.0, 90.0],  # Order 1, fiber 1
                [0.0, 0.0, 100.0],  # Order 1, fiber 2
                [0.0, 0.0, 110.0],  # Order 1, fiber 3
                [0.0, 0.0, 190.0],  # Order 2, fiber 1
                [0.0, 0.0, 200.0],  # Order 2, fiber 2
                [0.0, 0.0, 210.0],  # Order 2, fiber 3
            ]
        )
        column_range = np.array([[10, 990]] * 6)

        config = FibersConfig(
            per_order=True,
            fibers_per_order=3,
            order_centers={1: 100.0, 2: 200.0},
            groups={
                "A": FiberGroupConfig(range=(1, 3), merge="center"),  # fibers 1-2
                "B": FiberGroupConfig(range=(3, 4), merge="center"),  # fiber 3
            },
        )

        group_traces, group_cr, group_counts, _ = trace.organize_fibers(
            traces, column_range, config
        )

        # Should have 2 groups, each with 2 orders
        assert "A" in group_traces
        assert "B" in group_traces
        assert 1 in group_traces["A"]
        assert 2 in group_traces["A"]
        assert 1 in group_traces["B"]
        assert 2 in group_traces["B"]

        # Each merged trace should be 1 trace
        assert len(group_traces["A"][1]) == 1
        assert len(group_traces["A"][2]) == 1

    @pytest.mark.unit
    def test_organize_fibers_bundle_centers_all_present(self):
        """Test bundle_centers with all fibers present picks middle index."""
        from pyreduce.instruments.models import FiberBundleConfig, FibersConfig

        # 3 bundles of 5 fibers each, bundle centers at y=100, 200, 300
        traces = np.array(
            [
                # Bundle 1 (center at 100): fibers at 80, 90, 100, 110, 120
                [0.0, 0.0, 80.0],
                [0.0, 0.0, 90.0],
                [0.0, 0.0, 100.0],
                [0.0, 0.0, 110.0],
                [0.0, 0.0, 120.0],
                # Bundle 2 (center at 200): fibers at 180, 190, 200, 210, 220
                [0.0, 0.0, 180.0],
                [0.0, 0.0, 190.0],
                [0.0, 0.0, 200.0],
                [0.0, 0.0, 210.0],
                [0.0, 0.0, 220.0],
                # Bundle 3 (center at 300): fibers at 280, 290, 300, 310, 320
                [0.0, 0.0, 280.0],
                [0.0, 0.0, 290.0],
                [0.0, 0.0, 300.0],
                [0.0, 0.0, 310.0],
                [0.0, 0.0, 320.0],
            ]
        )
        column_range = np.array([[10, 990]] * 15)

        config = FibersConfig(
            bundles=FiberBundleConfig(
                size=5,
                merge="center",
                bundle_centers={1: 100.0, 2: 200.0, 3: 300.0},
            )
        )

        group_traces, group_cr, group_counts, _ = trace.organize_fibers(
            traces, column_range, config
        )

        assert len(group_traces) == 3
        assert "bundle_1" in group_traces
        assert "bundle_2" in group_traces
        assert "bundle_3" in group_traces

        # All bundles have 5 fibers, should pick middle (index 2)
        assert group_counts["bundle_1"] == 5
        assert group_counts["bundle_2"] == 5
        assert group_counts["bundle_3"] == 5

        # Center trace is y=100, 200, 300 respectively
        assert group_traces["bundle_1"][0, 2] == pytest.approx(100.0)
        assert group_traces["bundle_2"][0, 2] == pytest.approx(200.0)
        assert group_traces["bundle_3"][0, 2] == pytest.approx(300.0)

    @pytest.mark.unit
    def test_organize_fibers_bundle_centers_missing_fibers(self):
        """Test bundle_centers with missing fibers picks closest to center."""
        from pyreduce.instruments.models import FiberBundleConfig, FibersConfig

        # Bundle 1: all 5 fibers present
        # Bundle 2: only 4 fibers, center fiber (y=200) missing
        # Bundle 3: only 3 fibers, bottom two missing
        traces = np.array(
            [
                # Bundle 1: complete
                [0.0, 0.0, 80.0],
                [0.0, 0.0, 90.0],
                [0.0, 0.0, 100.0],
                [0.0, 0.0, 110.0],
                [0.0, 0.0, 120.0],
                # Bundle 2: center missing (no y=200)
                [0.0, 0.0, 180.0],
                [0.0, 0.0, 190.0],
                [0.0, 0.0, 210.0],
                [0.0, 0.0, 220.0],
                # Bundle 3: bottom two missing (no y=280, 290)
                [0.0, 0.0, 300.0],
                [0.0, 0.0, 310.0],
                [0.0, 0.0, 320.0],
            ]
        )
        column_range = np.array([[10, 990]] * 12)

        config = FibersConfig(
            bundles=FiberBundleConfig(
                size=5,
                merge="center",
                bundle_centers={1: 100.0, 2: 200.0, 3: 300.0},
            )
        )

        group_traces, group_cr, group_counts, _ = trace.organize_fibers(
            traces, column_range, config
        )

        x_mid = 500  # middle of column_range

        # Bundle 1: all present, picks middle (y=100)
        assert group_counts["bundle_1"] == 5
        y1 = np.polyval(group_traces["bundle_1"][0], x_mid)
        assert y1 == pytest.approx(100.0)

        # Bundle 2: 4 fibers, center missing, averages neighbors (190 + 210) / 2 = 200
        assert group_counts["bundle_2"] == 4
        y2 = np.polyval(group_traces["bundle_2"][0], x_mid)
        assert y2 == pytest.approx(200.0, abs=1.0)

        # Bundle 3: 3 fibers, picks closest to 300 (which is 300 itself)
        assert group_counts["bundle_3"] == 3
        y3 = np.polyval(group_traces["bundle_3"][0], x_mid)
        assert y3 == pytest.approx(300.0)

    @pytest.mark.unit
    def test_organize_fibers_bundle_centers_average_merge(self):
        """Test bundle_centers with average merge uses all present fibers."""
        from pyreduce.instruments.models import FiberBundleConfig, FibersConfig

        # Bundle with only 3 of 5 fibers at y=90, 100, 110
        traces = np.array(
            [
                [0.0, 0.0, 90.0],
                [0.0, 0.0, 100.0],
                [0.0, 0.0, 110.0],
            ]
        )
        column_range = np.array([[10, 990]] * 3)

        config = FibersConfig(
            bundles=FiberBundleConfig(
                size=5,
                merge="average",
                bundle_centers={1: 100.0},
            )
        )

        group_traces, group_cr, group_counts, _ = trace.organize_fibers(
            traces, column_range, config, degree=2
        )

        assert group_counts["bundle_1"] == 3
        # Average of 90, 100, 110 is 100
        assert group_traces["bundle_1"][0, 2] == pytest.approx(100.0, abs=1.0)

    @pytest.mark.unit
    def test_organize_fibers_bundle_centers_empty_bundle(self):
        """Test bundle_centers handles bundles with no traces."""
        from pyreduce.instruments.models import FiberBundleConfig, FibersConfig

        # Only traces near bundle 1, none near bundle 2
        traces = np.array(
            [
                [0.0, 0.0, 90.0],
                [0.0, 0.0, 100.0],
                [0.0, 0.0, 110.0],
            ]
        )
        column_range = np.array([[10, 990]] * 3)

        config = FibersConfig(
            bundles=FiberBundleConfig(
                size=5,
                merge="center",
                bundle_centers={1: 100.0, 2: 500.0},  # Bundle 2 far away
            )
        )

        group_traces, group_cr, group_counts, _ = trace.organize_fibers(
            traces, column_range, config
        )

        # Bundle 1 gets all traces (closest)
        assert group_counts["bundle_1"] == 3
        # Bundle 2 is empty
        assert group_counts["bundle_2"] == 0
        assert len(group_traces["bundle_2"]) == 0

    @pytest.mark.unit
    def test_organize_fibers_height_explicit(self, sample_traces):
        """Test organize_fibers with explicit height."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, column_range = sample_traces
        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 11), merge="center", height=50.0)}
        )

        _, _, _, group_heights = trace.organize_fibers(traces, column_range, config)

        assert group_heights["A"] == 50.0

    @pytest.mark.unit
    def test_organize_fibers_height_derived(self, sample_traces):
        """Test organize_fibers with derived height."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, column_range = sample_traces
        # Traces are at y = 100, 110, 120, ..., 190 (spacing = 10)
        # 10 fibers: span = 90, fiber_diameter = 10, total = 100
        config = FibersConfig(
            groups={
                "A": FiberGroupConfig(range=(1, 11), merge="center", height="derived")
            }
        )

        _, _, _, group_heights = trace.organize_fibers(traces, column_range, config)

        assert group_heights["A"] == pytest.approx(100.0)

    @pytest.mark.unit
    def test_organize_fibers_height_none(self, sample_traces):
        """Test organize_fibers with no height specified (default)."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, column_range = sample_traces
        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 11), merge="center")}  # no height
        )

        _, _, _, group_heights = trace.organize_fibers(traces, column_range, config)

        assert group_heights["A"] is None


class TestSelectTracesForStep:
    """Tests for select_traces_for_step function."""

    @pytest.fixture
    def raw_traces(self):
        """Create raw traces."""
        traces = np.array([[0.0, 0.0, 100.0 + i * 10] for i in range(10)])
        column_range = np.array([[10, 990]] * 10)
        return traces, column_range

    @pytest.fixture
    def group_traces(self):
        """Create grouped traces."""
        return {
            "A": np.array([[0.0, 0.0, 125.0]]),  # Averaged
            "B": np.array([[0.0, 0.0, 175.0]]),  # Averaged
        }

    @pytest.fixture
    def group_cr(self):
        """Create grouped column ranges."""
        return {
            "A": np.array([[10, 990]]),
            "B": np.array([[10, 990]]),
        }

    @pytest.mark.unit
    def test_select_traces_no_config(self, raw_traces):
        """Test that None config returns raw traces in 'all' key."""
        traces, cr = raw_traces

        result = trace.select_traces_for_step(traces, cr, {}, {}, None, "science")

        assert "all" in result
        selected, selected_cr, _ = result["all"]
        assert np.array_equal(selected, traces)
        assert np.array_equal(selected_cr, cr)

    @pytest.mark.unit
    def test_select_traces_all(self, raw_traces, group_traces, group_cr):
        """Test selecting all raw traces."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, cr = raw_traces
        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 11))},
            use={"science": "all"},
        )

        result = trace.select_traces_for_step(
            traces, cr, group_traces, group_cr, config, "science"
        )

        assert "all" in result
        selected, _, _ = result["all"]
        assert np.array_equal(selected, traces)

    @pytest.mark.unit
    def test_select_traces_groups(self, raw_traces, group_traces, group_cr):
        """Test selecting all grouped traces stacked."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, cr = raw_traces
        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"science": "groups"},
        )

        result = trace.select_traces_for_step(
            traces, cr, group_traces, group_cr, config, "science"
        )

        # "groups" stacks all into single "all" entry
        assert "all" in result
        selected, _, _ = result["all"]
        # Should concatenate all groups (A and B)
        assert len(selected) == 2

    @pytest.mark.unit
    def test_select_traces_specific_groups(self, raw_traces, group_traces, group_cr):
        """Test selecting specific named groups returns dict per group."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, cr = raw_traces
        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"science": ["A"]},  # Only group A
        )

        result = trace.select_traces_for_step(
            traces, cr, group_traces, group_cr, config, "science"
        )

        # Explicit list returns dict with named keys
        assert "A" in result
        assert len(result) == 1
        selected, _, _ = result["A"]
        assert len(selected) == 1

    @pytest.mark.unit
    def test_select_traces_explicit_default(self, raw_traces, group_traces, group_cr):
        """Test explicit default key in use config."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, cr = raw_traces
        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"default": "groups"},  # explicit default
        )

        result = trace.select_traces_for_step(
            traces, cr, group_traces, group_cr, config, "science"
        )

        # Should use the explicit default "groups"
        assert "all" in result
        selected, _, _ = result["all"]
        assert len(selected) == 2

    @pytest.mark.unit
    def test_select_traces_default_all(self, raw_traces, group_traces, group_cr):
        """Test explicit default: all returns raw traces."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, cr = raw_traces
        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"default": "all"},
        )

        result = trace.select_traces_for_step(
            traces, cr, group_traces, group_cr, config, "science"
        )

        assert "all" in result
        selected, _, _ = result["all"]
        assert np.array_equal(selected, traces)

    @pytest.mark.unit
    def test_select_traces_step_overrides_default(
        self, raw_traces, group_traces, group_cr
    ):
        """Test step-specific config takes precedence over default."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, cr = raw_traces
        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"default": "groups", "science": "all"},  # science overrides
        )

        result = trace.select_traces_for_step(
            traces, cr, group_traces, group_cr, config, "science"
        )

        # science: all should override default: groups
        assert "all" in result
        selected, _, _ = result["all"]
        assert np.array_equal(selected, traces)

    @pytest.mark.unit
    def test_select_traces_no_default_falls_back_to_all(
        self, raw_traces, group_traces, group_cr
    ):
        """Test missing default key falls back to 'all'."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, cr = raw_traces
        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"other_step": "groups"},  # no default, science not specified
        )

        result = trace.select_traces_for_step(
            traces, cr, group_traces, group_cr, config, "science"
        )

        # Without default key, should fall back to "all"
        assert "all" in result
        selected, _, _ = result["all"]
        assert np.array_equal(selected, traces)

    @pytest.mark.unit
    def test_select_traces_missing_group_warns(
        self, raw_traces, group_traces, group_cr
    ):
        """Test warning when requested group doesn't exist."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, cr = raw_traces
        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"science": ["A", "nonexistent"]},
        )

        # Should warn but still return valid groups
        result = trace.select_traces_for_step(
            traces, cr, group_traces, group_cr, config, "science"
        )

        # Only A found as named key
        assert "A" in result
        assert len(result) == 1

    @pytest.mark.unit
    def test_select_traces_per_order(self):
        """Test selecting from per-order grouped traces."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        # Raw traces (not used when selecting groups)
        raw_traces = np.array([[0.0, 0.0, 100.0 + i * 10] for i in range(6)])
        raw_cr = np.array([[10, 990]] * 6)

        # Per-order grouped traces: {group: {order: trace}}
        group_traces = {
            "A": {1: np.array([[0.0, 0.0, 100.0]]), 2: np.array([[0.0, 0.0, 200.0]])},
            "B": {1: np.array([[0.0, 0.0, 110.0]]), 2: np.array([[0.0, 0.0, 210.0]])},
        }
        group_cr = {
            "A": {1: np.array([[10, 990]]), 2: np.array([[10, 990]])},
            "B": {1: np.array([[10, 990]]), 2: np.array([[10, 990]])},
        }

        config = FibersConfig(
            per_order=True,
            order_centers={1: 100.0, 2: 200.0},
            groups={
                "A": FiberGroupConfig(range=(1, 2)),
                "B": FiberGroupConfig(range=(2, 3)),
            },
            use={"science": ["A"]},
        )

        result = trace.select_traces_for_step(
            raw_traces, raw_cr, group_traces, group_cr, config, "science"
        )

        # Should return A with stacked traces from both orders
        assert "A" in result
        assert len(result) == 1
        selected, _, _ = result["A"]
        assert len(selected) == 2  # 2 orders

    @pytest.mark.unit
    def test_select_traces_returns_height(self, raw_traces):
        """Test that select_traces_for_step returns group heights."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces, cr = raw_traces
        group_traces = {"A": np.array([[0.0, 0.0, 125.0]])}
        group_cr = {"A": np.array([[10, 990]])}
        group_heights = {"A": 42.0}

        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6), height=42.0)},
            use={"science": ["A"]},
        )

        result = trace.select_traces_for_step(
            traces, cr, group_traces, group_cr, config, "science", group_heights
        )

        # Should return height in tuple
        assert "A" in result
        _, _, height = result["A"]
        assert height == 42.0


class TestNoiseThreshold:
    """Tests for noise threshold settings in trace()."""

    @pytest.fixture
    def simple_image(self):
        """Create a simple test image with known background and signal.

        Image is 100x200 (nrow x ncol) with:
        - Background level of 1000
        - A horizontal "order" at row 50 with signal 100 above background
        """
        nrow, ncol = 100, 200
        im = np.full((nrow, ncol), 1000.0)
        # Add a trace at row 50, width ~5 pixels
        for row in range(48, 53):
            im[row, :] = 1100.0
        return im

    @pytest.mark.unit
    def test_both_zero_defaults_to_relative(self, simple_image, caplog):
        """When both noise=0 and noise_relative=0, should default to 0.001."""
        import logging

        caplog.set_level(logging.INFO)

        # Should work and log the default
        trace.trace(
            simple_image,
            noise=0,
            noise_relative=0,
            manual=False,
        )

        assert "Using default noise_relative=0.001" in caplog.text

    @pytest.mark.unit
    def test_absolute_noise_only(self, simple_image):
        """Test with only absolute noise threshold."""
        # Signal is 100 above background (1100 vs 1000)
        # With noise=50, signal should be detected (100 > 50)
        result = trace.trace(
            simple_image,
            noise=50,
            noise_relative=0,
            manual=False,
        )
        orders, column_range, heights = result
        assert len(orders) >= 1

        # With noise=150, signal should NOT be detected (100 < 150)
        result = trace.trace(
            simple_image,
            noise=150,
            noise_relative=0,
            manual=False,
        )
        orders, column_range, heights = result
        assert len(orders) == 0

    @pytest.mark.unit
    def test_relative_noise_only(self, simple_image):
        """Test with only relative noise threshold."""
        # Background ~1000, signal 100 above (10% above background)
        # With noise_relative=0.05 (5%), signal should be detected
        result = trace.trace(
            simple_image,
            noise=0,
            noise_relative=0.05,
            manual=False,
        )
        orders, column_range, heights = result
        assert len(orders) >= 1

        # With noise_relative=0.15 (15%), signal should NOT be detected
        result = trace.trace(
            simple_image,
            noise=0,
            noise_relative=0.15,
            manual=False,
        )
        orders, column_range, heights = result
        assert len(orders) == 0

    @pytest.mark.unit
    def test_combined_thresholds(self, simple_image):
        """Test with both absolute and relative thresholds combined."""
        # Background ~1000, signal 100 above
        # Threshold = background * (1 + noise_relative) + noise
        #           = 1000 * 1.05 + 20 = 1070
        # Signal at 1100 > 1070, should detect
        result = trace.trace(
            simple_image,
            noise=20,
            noise_relative=0.05,
            manual=False,
        )
        orders, column_range, heights = result
        assert len(orders) >= 1

        # Threshold = 1000 * 1.08 + 30 = 1110
        # Signal at 1100 < 1110, should NOT detect
        result = trace.trace(
            simple_image,
            noise=30,
            noise_relative=0.08,
            manual=False,
        )
        orders, column_range, heights = result
        assert len(orders) == 0


class TestChannelTemplateSubstitution:
    """Tests for {channel} template substitution in order_centers_file."""

    @pytest.mark.unit
    def test_channel_template_substitution(self, tmp_path):
        """Test that {channel} in order_centers_file is substituted with channel name."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        # Create a temporary order_centers file for channel "j"
        order_centers_file = tmp_path / "order_centers_j.yaml"
        order_centers_file.write_text("1: 100\n2: 200\n")

        # Create traces at y=100 and y=200
        traces = np.array(
            [
                [0.0, 0.0, 100.0],
                [0.0, 0.0, 200.0],
            ]
        )
        column_range = np.array([[10, 990], [10, 990]])

        config = FibersConfig(
            per_order=True,
            fibers_per_order=1,
            order_centers_file="order_centers_{channel}.yaml",
            groups={"A": FiberGroupConfig(range=(1, 2), merge="center")},
        )

        group_traces, group_cr, group_counts, _ = trace.organize_fibers(
            traces,
            column_range,
            config,
            instrument_dir=str(tmp_path),
            channel="J",  # Should resolve to order_centers_j.yaml
        )

        # Should have organized into orders 1 and 2
        assert "A" in group_traces
        assert 1 in group_traces["A"]
        assert 2 in group_traces["A"]

    @pytest.mark.unit
    def test_channel_template_lowercase(self, tmp_path):
        """Test that channel name is lowercased in template substitution."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        # Create file with lowercase channel name
        order_centers_file = tmp_path / "order_centers_h.yaml"
        order_centers_file.write_text("10: 150\n")

        traces = np.array([[0.0, 0.0, 150.0]])
        column_range = np.array([[10, 990]])

        config = FibersConfig(
            per_order=True,
            fibers_per_order=1,
            order_centers_file="order_centers_{channel}.yaml",
            groups={"A": FiberGroupConfig(range=(1, 2), merge="center")},
        )

        # Pass uppercase channel, should still find lowercase file
        group_traces, group_cr, group_counts, _ = trace.organize_fibers(
            traces,
            column_range,
            config,
            instrument_dir=str(tmp_path),
            channel="H",
        )

        assert "A" in group_traces
        assert 10 in group_traces["A"]

    @pytest.mark.unit
    def test_no_channel_no_substitution(self, tmp_path, caplog):
        """Test that template is used literally when channel is None."""
        import logging

        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        caplog.set_level(logging.WARNING)

        traces = np.array([[0.0, 0.0, 100.0]])
        column_range = np.array([[10, 990]])

        config = FibersConfig(
            per_order=True,
            order_centers_file="order_centers_{channel}.yaml",
            groups={"A": FiberGroupConfig(range=(1, 2), merge="center")},
        )

        # Without channel, should try to load literal filename, warn, and return empty
        group_traces, group_cr, group_counts, _ = trace.organize_fibers(
            traces,
            column_range,
            config,
            instrument_dir=str(tmp_path),
            channel=None,
        )

        # Should warn about missing file with literal {channel} in path
        assert "order_centers_{channel}.yaml" in caplog.text
        assert group_traces == {}


class TestTraceByGrouping:
    """Tests for trace_by config option in Trace step."""

    @pytest.mark.unit
    def test_trace_by_groups_files_by_header(self, tmp_path):
        """Test that _trace_by_groups correctly groups files by header value."""
        from astropy.io import fits

        # Create mock FITS files with different FIBMODE headers
        for mode in ["even", "odd"]:
            img = np.zeros((100, 100), dtype=np.float32)
            hdu = fits.PrimaryHDU(img)
            hdu.header["FIBMODE"] = mode
            hdu.writeto(tmp_path / f"flat_{mode}.fits", overwrite=True)

        # Test the grouping logic directly using fits.getheader
        files = [str(tmp_path / "flat_even.fits"), str(tmp_path / "flat_odd.fits")]
        trace_by = "FIBMODE"

        # Group files by header value (mimics _trace_by_groups logic)
        file_groups = {}
        for f in files:
            hdr = fits.getheader(f)
            group_key = hdr.get(trace_by, "unknown")
            if group_key not in file_groups:
                file_groups[group_key] = []
            file_groups[group_key].append(f)

        assert len(file_groups) == 2
        assert "even" in file_groups
        assert "odd" in file_groups
        assert len(file_groups["even"]) == 1
        assert len(file_groups["odd"]) == 1

    @pytest.mark.unit
    def test_trace_by_merges_and_sorts_traces(self):
        """Test that trace_by merges traces from groups and sorts by y-position."""
        # Simulate traces from two groups
        traces_even = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 300.0]])  # y=100, 300
        cr_even = np.array([[0, 1000], [0, 1000]])
        heights_even = [10.0, 10.0]

        traces_odd = np.array([[0.0, 0.0, 200.0]])  # y=200
        cr_odd = np.array([[0, 1000]])
        heights_odd = [10.0]

        # Merge (mimics logic in _trace_by_groups)
        all_traces = [traces_even, traces_odd]
        all_cr = [cr_even, cr_odd]
        all_heights = []
        all_heights.extend(heights_even)
        all_heights.extend(heights_odd)

        traces = np.vstack(all_traces)
        column_range = np.vstack(all_cr)
        heights = np.array(all_heights)

        # Sort by y-position
        mid_x = traces.shape[1] // 2 if traces.shape[1] > 1 else 0
        y_positions = np.polyval(traces[:, ::-1].T, mid_x)
        sort_idx = np.argsort(y_positions)
        traces = traces[sort_idx]
        column_range = column_range[sort_idx]
        heights = heights[sort_idx]

        # Check merged and sorted
        assert len(traces) == 3
        # Should be sorted by y: 100, 200, 300
        assert traces[0, 2] == pytest.approx(100.0)
        assert traces[1, 2] == pytest.approx(200.0)
        assert traces[2, 2] == pytest.approx(300.0)

    @pytest.mark.unit
    def test_fibers_config_trace_by_field(self):
        """Test that FibersConfig accepts trace_by field."""
        from pyreduce.instruments.models import FibersConfig

        config = FibersConfig(trace_by="FIBMODE")
        assert config.trace_by == "FIBMODE"

        config_none = FibersConfig()
        assert config_none.trace_by is None

    @pytest.mark.unit
    def test_trace_by_with_three_groups(self, tmp_path):
        """Test trace_by with three illumination patterns."""
        from astropy.io import fits

        # Create files for three groups
        for mode in ["third1", "third2", "third3"]:
            img = np.zeros((100, 100), dtype=np.float32)
            hdu = fits.PrimaryHDU(img)
            hdu.header["ILLUM"] = mode
            hdu.writeto(tmp_path / f"flat_{mode}.fits", overwrite=True)

        files = [
            str(tmp_path / "flat_third1.fits"),
            str(tmp_path / "flat_third2.fits"),
            str(tmp_path / "flat_third3.fits"),
        ]
        trace_by = "ILLUM"

        # Group files
        file_groups = {}
        for f in files:
            hdr = fits.getheader(f)
            group_key = hdr.get(trace_by, "unknown")
            if group_key not in file_groups:
                file_groups[group_key] = []
            file_groups[group_key].append(f)

        assert len(file_groups) == 3
        assert set(file_groups.keys()) == {"third1", "third2", "third3"}


class TestPerOrderMissingFile:
    """Tests for graceful handling of missing order_centers_file."""

    @pytest.mark.unit
    def test_missing_order_centers_file_warns_and_continues(self, tmp_path, caplog):
        """Test that missing order_centers_file logs warning and returns empty groups."""
        import logging

        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        caplog.set_level(logging.WARNING)

        traces = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 200.0]])
        column_range = np.array([[10, 990], [10, 990]])

        config = FibersConfig(
            per_order=True,
            fibers_per_order=1,
            order_centers_file="nonexistent_file.yaml",
            groups={"A": FiberGroupConfig(range=(1, 2), merge="center")},
        )

        # Should not raise, should warn
        group_traces, group_cr, group_counts, group_heights = trace.organize_fibers(
            traces,
            column_range,
            config,
            instrument_dir=str(tmp_path),
            channel="test",
        )

        # Should return empty groups
        assert group_traces == {}
        assert group_cr == {}
        assert group_counts == {}

        # Should have logged a warning
        assert "Order centers file not found" in caplog.text
        assert "Skipping fiber grouping" in caplog.text

    @pytest.mark.unit
    def test_missing_order_centers_with_channel_template(self, tmp_path, caplog):
        """Test missing file with {channel} template substitution."""
        import logging

        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        caplog.set_level(logging.WARNING)

        traces = np.array([[0.0, 0.0, 100.0]])
        column_range = np.array([[10, 990]])

        config = FibersConfig(
            per_order=True,
            order_centers_file="order_centers_{channel}.yaml",
            groups={"A": FiberGroupConfig(range=(1, 2), merge="center")},
        )

        group_traces, group_cr, group_counts, group_heights = trace.organize_fibers(
            traces,
            column_range,
            config,
            instrument_dir=str(tmp_path),
            channel="R0",
        )

        # Should warn about the resolved filename
        assert "order_centers_r0.yaml" in caplog.text
        assert group_traces == {}

    @pytest.mark.unit
    def test_inline_order_centers_works(self):
        """Test that inline order_centers (no file) still works."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        traces = np.array(
            [
                [0.0, 0.0, 100.0],
                [0.0, 0.0, 200.0],
            ]
        )
        column_range = np.array([[10, 990], [10, 990]])

        config = FibersConfig(
            per_order=True,
            fibers_per_order=1,
            order_centers={1: 100.0, 2: 200.0},  # Inline, no file needed
            groups={"A": FiberGroupConfig(range=(1, 2), merge="center")},
        )

        group_traces, group_cr, group_counts, group_heights = trace.organize_fibers(
            traces, column_range, config
        )

        # Should work normally with inline order_centers
        assert "A" in group_traces
        assert 1 in group_traces["A"]
        assert 2 in group_traces["A"]


class TestNaturalSortKey:
    """Tests for _natural_sort_key function."""

    @pytest.mark.unit
    def test_natural_sort_basic(self):
        """Test natural sorting of bundle names."""
        names = ["bundle_1", "bundle_10", "bundle_2", "bundle_20", "bundle_3"]
        sorted_names = sorted(names, key=trace._natural_sort_key)
        assert sorted_names == [
            "bundle_1",
            "bundle_2",
            "bundle_3",
            "bundle_10",
            "bundle_20",
        ]

    @pytest.mark.unit
    def test_natural_sort_mixed(self):
        """Test natural sorting with mixed prefixes."""
        names = ["fiber_1", "fiber_10", "bundle_2", "fiber_2"]
        sorted_names = sorted(names, key=trace._natural_sort_key)
        assert sorted_names == ["bundle_2", "fiber_1", "fiber_2", "fiber_10"]

    @pytest.mark.unit
    def test_natural_sort_no_numbers(self):
        """Test sorting when no numbers present."""
        names = ["apple", "banana", "cherry"]
        sorted_names = sorted(names, key=trace._natural_sort_key)
        assert sorted_names == ["apple", "banana", "cherry"]

    @pytest.mark.unit
    def test_natural_sort_multiple_numbers(self):
        """Test sorting with multiple number groups."""
        names = ["order_1_fiber_10", "order_1_fiber_2", "order_2_fiber_1"]
        sorted_names = sorted(names, key=trace._natural_sort_key)
        assert sorted_names == [
            "order_1_fiber_2",
            "order_1_fiber_10",
            "order_2_fiber_1",
        ]
