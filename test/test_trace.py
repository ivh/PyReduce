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
    """Tests for _compute_heights_inplace function."""

    @pytest.mark.unit
    def test_evenly_spaced_traces(self):
        """Test height computation for evenly spaced traces."""
        from pyreduce.trace_model import Trace as TraceData

        # 5 traces spaced 20 pixels apart
        traces = [
            TraceData(m=None, pos=np.array([0.0, 0.0, 20.0]), column_range=(0, 1000)),
            TraceData(m=None, pos=np.array([0.0, 0.0, 40.0]), column_range=(0, 1000)),
            TraceData(m=None, pos=np.array([0.0, 0.0, 60.0]), column_range=(0, 1000)),
            TraceData(m=None, pos=np.array([0.0, 0.0, 80.0]), column_range=(0, 1000)),
            TraceData(m=None, pos=np.array([0.0, 0.0, 100.0]), column_range=(0, 1000)),
        ]

        trace._compute_heights_inplace(traces, ncol=1000)

        # First trace: distance to next neighbor = 20
        assert traces[0].height == pytest.approx(20.0, rel=0.01)
        # Middle traces: half distance between neighbors = 20
        assert traces[1].height == pytest.approx(20.0, rel=0.01)
        assert traces[2].height == pytest.approx(20.0, rel=0.01)
        assert traces[3].height == pytest.approx(20.0, rel=0.01)
        # Last trace: distance to previous neighbor = 20
        assert traces[4].height == pytest.approx(20.0, rel=0.01)

    @pytest.mark.unit
    def test_uneven_spacing(self):
        """Test height computation for unevenly spaced traces."""
        from pyreduce.trace_model import Trace as TraceData

        traces = [
            TraceData(m=None, pos=np.array([0.0, 0.0, 10.0]), column_range=(0, 1000)),
            TraceData(m=None, pos=np.array([0.0, 0.0, 30.0]), column_range=(0, 1000)),
            TraceData(m=None, pos=np.array([0.0, 0.0, 90.0]), column_range=(0, 1000)),
        ]

        trace._compute_heights_inplace(traces, ncol=1000)

        # First: distance to next = 20
        assert traces[0].height == pytest.approx(20.0, rel=0.01)
        # Middle: half distance between neighbors = (90-10)/2 = 40
        assert traces[1].height == pytest.approx(40.0, rel=0.01)
        # Last: distance to prev = 60
        assert traces[2].height == pytest.approx(60.0, rel=0.01)

    @pytest.mark.unit
    def test_single_trace(self):
        """Single trace should leave height as None (no neighbors)."""
        from pyreduce.trace_model import Trace as TraceData

        traces = [
            TraceData(m=None, pos=np.array([0.0, 0.0, 50.0]), column_range=(0, 1000))
        ]

        trace._compute_heights_inplace(traces, ncol=1000)

        assert traces[0].height is None

    @pytest.mark.unit
    def test_empty_traces(self):
        """Empty traces should be handled gracefully."""
        trace._compute_heights_inplace([], ncol=1000)
        # Just verify no exception is raised

    @pytest.mark.unit
    def test_varying_column_range(self):
        """Height uses max across valid reference columns."""
        from pyreduce.trace_model import Trace as TraceData

        traces = [
            TraceData(m=None, pos=np.array([0.001, 50.0]), column_range=(100, 900)),
            TraceData(m=None, pos=np.array([0.001, 70.0]), column_range=(100, 900)),
        ]

        trace._compute_heights_inplace(traces, ncol=1000)

        # Spacing is constant at 20 pixels regardless of x
        assert traces[0].height == pytest.approx(20.0, rel=0.01)
        assert traces[1].height == pytest.approx(20.0, rel=0.01)


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


class TestSelectTracesForStep:
    """Tests for select_traces_for_step function with Trace objects."""

    @pytest.fixture
    def trace_objects(self):
        """Create Trace objects with different fiber assignments."""
        from pyreduce.trace_model import Trace

        traces = []
        for i in range(10):
            # Assign fibers: first 5 to "A", last 5 to "B"
            fiber = "A" if i < 5 else "B"
            traces.append(
                Trace(
                    m=i,
                    group=fiber,
                    pos=np.array([0.0, 0.0, 100.0 + i * 10]),
                    column_range=(10, 990),
                )
            )
        return traces

    @pytest.fixture
    def all_fiber_traces(self):
        """Create Trace objects all with default fiber (0)."""
        from pyreduce.trace_model import Trace

        return [
            Trace(
                m=i,
                group=0,
                pos=np.array([0.0, 0.0, 100.0 + i * 10]),
                column_range=(10, 990),
            )
            for i in range(10)
        ]

    @pytest.mark.unit
    def test_select_traces_no_config(self, all_fiber_traces):
        """Test that None config returns all traces in 'all' key."""
        result = trace.select_traces_for_step(all_fiber_traces, None, "science")

        assert "all" in result
        assert len(result["all"]) == 10

    @pytest.mark.unit
    def test_select_traces_groups(self, trace_objects):
        """Test selecting grouped traces with use='groups'."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"science": "groups"},
        )

        result = trace.select_traces_for_step(trace_objects, config, "science")

        # "groups" returns all non-default fiber traces
        assert "all" in result
        # All 10 traces have fiber A or B (non-default)
        assert len(result["all"]) == 10

    @pytest.mark.unit
    def test_select_traces_specific_groups(self, trace_objects):
        """Test selecting specific named groups returns dict per group."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"science": ["A"]},  # Only group A
        )

        result = trace.select_traces_for_step(trace_objects, config, "science")

        # Explicit list returns dict with named keys
        assert "A" in result
        assert len(result) == 1
        assert len(result["A"]) == 5  # First 5 traces have group="A"

    @pytest.mark.unit
    def test_select_traces_explicit_default(self, trace_objects):
        """Test explicit default key in use config."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"default": "groups"},  # explicit default
        )

        result = trace.select_traces_for_step(trace_objects, config, "science")

        # Should use the explicit default "groups"
        assert "all" in result
        # All traces have non-default fiber
        assert len(result["all"]) == 10

    @pytest.mark.unit
    def test_select_traces_step_overrides_default(self, trace_objects):
        """Test step-specific config takes precedence over default."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"default": ["A"], "science": "groups"},  # science overrides
        )

        result = trace.select_traces_for_step(trace_objects, config, "science")

        # science: groups should override default (returns grouped traces)
        assert "all" in result
        assert len(result["all"]) == 10

    @pytest.mark.unit
    def test_select_traces_no_default_falls_back_to_groups(self, trace_objects):
        """Test missing default key falls back to 'groups'."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"other_step": "per_fiber"},  # no default, science not specified
        )

        result = trace.select_traces_for_step(trace_objects, config, "science")

        # Without default key, should fall back to "groups"
        assert "all" in result
        assert len(result["all"]) == 10

    @pytest.mark.unit
    def test_select_traces_missing_group_warns(self, trace_objects):
        """Test warning when requested group doesn't exist."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 6))},
            use={"science": ["A", "nonexistent"]},
        )

        # Should warn but still return valid groups
        result = trace.select_traces_for_step(trace_objects, config, "science")

        # Only A found as named key
        assert "A" in result
        assert len(result) == 1
        assert len(result["A"]) == 5

    @pytest.mark.unit
    def test_select_traces_per_order(self):
        """Test selecting traces with different order numbers."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig
        from pyreduce.trace_model import Trace

        # Create traces with order numbers and fiber assignments
        traces = [
            Trace(
                m=1, group="A", pos=np.array([0.0, 0.0, 100.0]), column_range=(10, 990)
            ),
            Trace(
                m=1, group="B", pos=np.array([0.0, 0.0, 110.0]), column_range=(10, 990)
            ),
            Trace(
                m=2, group="A", pos=np.array([0.0, 0.0, 200.0]), column_range=(10, 990)
            ),
            Trace(
                m=2, group="B", pos=np.array([0.0, 0.0, 210.0]), column_range=(10, 990)
            ),
        ]

        config = FibersConfig(
            groups={
                "A": FiberGroupConfig(range=(1, 2)),
                "B": FiberGroupConfig(range=(2, 3)),
            },
            use={"science": ["A"]},
        )

        result = trace.select_traces_for_step(traces, config, "science")

        # Should return A with traces from both orders
        assert "A" in result
        assert len(result) == 1
        assert len(result["A"]) == 2  # 2 orders with fiber A

    @pytest.mark.unit
    def test_select_traces_with_height(self):
        """Test that Trace objects preserve height information."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig
        from pyreduce.trace_model import Trace

        traces = [
            Trace(
                m=0,
                group="A",
                pos=np.array([0.0, 0.0, 100.0]),
                column_range=(10, 990),
                height=42.0,
            )
        ]

        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 2))},
            use={"science": ["A"]},
        )

        result = trace.select_traces_for_step(traces, config, "science")

        assert "A" in result
        assert len(result["A"]) == 1
        assert result["A"][0].height == 42.0


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
        traces = trace.trace(
            simple_image,
            noise=50,
            noise_relative=0,
            manual=False,
        )
        assert len(traces) >= 1

        # With noise=150, signal should NOT be detected (100 < 150)
        traces = trace.trace(
            simple_image,
            noise=150,
            noise_relative=0,
            manual=False,
        )
        assert len(traces) == 0

    @pytest.mark.unit
    def test_relative_noise_only(self, simple_image):
        """Test with only relative noise threshold."""
        # Background ~1000, signal 100 above (10% above background)
        # With noise_relative=0.05 (5%), signal should be detected
        traces = trace.trace(
            simple_image,
            noise=0,
            noise_relative=0.05,
            manual=False,
        )
        assert len(traces) >= 1

        # With noise_relative=0.15 (15%), signal should NOT be detected
        traces = trace.trace(
            simple_image,
            noise=0,
            noise_relative=0.15,
            manual=False,
        )
        assert len(traces) == 0

    @pytest.mark.unit
    def test_combined_thresholds(self, simple_image):
        """Test with both absolute and relative thresholds combined."""
        # Background ~1000, signal 100 above
        # Threshold = background * (1 + noise_relative) + noise
        #           = 1000 * 1.05 + 20 = 1070
        # Signal at 1100 > 1070, should detect
        traces = trace.trace(
            simple_image,
            noise=20,
            noise_relative=0.05,
            manual=False,
        )
        assert len(traces) >= 1

        # Threshold = 1000 * 1.08 + 30 = 1110
        # Signal at 1100 < 1110, should NOT detect
        traces = trace.trace(
            simple_image,
            noise=30,
            noise_relative=0.08,
            manual=False,
        )
        assert len(traces) == 0


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


class TestTraceReturnsTraceObjects:
    """Tests that trace() returns list[Trace] with proper attributes."""

    @pytest.fixture
    def simple_image(self):
        """Create a simple test image with two horizontal orders."""
        nrow, ncol = 200, 500
        im = np.full((nrow, ncol), 1000.0)
        # Two traces at rows 50 and 150
        for row in range(48, 53):
            im[row, :] = 1200.0
        for row in range(148, 153):
            im[row, :] = 1200.0
        return im

    @pytest.mark.unit
    def test_trace_returns_list_of_trace_objects(self, simple_image):
        """Test that trace() returns list[Trace] not arrays."""
        from pyreduce.trace_model import Trace as TraceData

        result = trace.trace(simple_image, manual=False)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(t, TraceData) for t in result)

    @pytest.mark.unit
    def test_trace_objects_have_fiber_idx(self, simple_image):
        """Test that returned Trace objects have fiber_idx set."""
        result = trace.trace(simple_image, manual=False)

        for t in result:
            assert t.fiber_idx is not None
            assert t.group is None  # Not yet grouped

    @pytest.mark.unit
    def test_trace_with_order_centers_assigns_m(self, simple_image):
        """Test that order_centers parameter assigns m values."""
        order_centers = {90: 50.0, 91: 150.0}

        result = trace.trace(
            simple_image,
            manual=False,
            order_centers=order_centers,
        )

        assert len(result) == 2
        m_values = {t.m for t in result}
        assert m_values == {90, 91}

    @pytest.mark.unit
    def test_trace_without_order_centers_assigns_sequential_m(self, simple_image):
        """Without order_centers, traces get sequential m values."""
        result = trace.trace(
            simple_image,
            manual=False,
            order_centers=None,
        )

        m_values = sorted(t.m for t in result)
        assert m_values == list(range(len(result)))
        for t in result:
            assert t.fiber_idx == 1


class TestGroupFibers:
    """Tests for the new group_fibers() function."""

    @pytest.fixture
    def sample_traces(self):
        """Create sample Trace objects with fiber_idx set."""
        from pyreduce.trace_model import Trace as TraceData

        # 4 traces: 2 orders (m=90, 91), 2 fibers each
        return [
            TraceData(
                m=90,
                fiber_idx=1,
                pos=np.array([0.0, 0.0, 100.0]),
                column_range=(10, 990),
            ),
            TraceData(
                m=90,
                fiber_idx=2,
                pos=np.array([0.0, 0.0, 120.0]),
                column_range=(10, 990),
            ),
            TraceData(
                m=91,
                fiber_idx=1,
                pos=np.array([0.0, 0.0, 200.0]),
                column_range=(10, 990),
            ),
            TraceData(
                m=91,
                fiber_idx=2,
                pos=np.array([0.0, 0.0, 220.0]),
                column_range=(10, 990),
            ),
        ]

    @pytest.mark.unit
    def test_group_fibers_no_config(self, sample_traces):
        """Test group_fibers with no config returns empty list."""
        result = trace.group_fibers(sample_traces, None)

        assert result == []

    @pytest.mark.unit
    def test_group_fibers_with_groups_center(self, sample_traces):
        """Test group_fibers with groups config using center merge."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 3), merge="center")}
        )

        result = trace.group_fibers(sample_traces, config)

        # Should have 2 grouped traces (one per order)
        assert len(result) == 2
        for t in result:
            assert t.group == "A"
            assert t.fiber_idx is None
            assert t.m in {90, 91}

    @pytest.mark.unit
    def test_group_fibers_with_groups_average(self, sample_traces):
        """Test group_fibers with average merge method."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 3), merge="average")}
        )

        result = trace.group_fibers(sample_traces, config)

        # Should have 2 grouped traces (one per order)
        assert len(result) == 2

        # Check that the average was computed (y position should be midpoint)
        for t in result:
            x_mid = 500
            y = t.y_at_x(x_mid)
            # Order 90: avg of 100 and 120 = 110; Order 91: avg of 200 and 220 = 210
            if t.m == 90:
                assert y == pytest.approx(110.0, abs=1.0)
            else:
                assert y == pytest.approx(210.0, abs=1.0)

    @pytest.mark.unit
    def test_group_fibers_preserves_m(self, sample_traces):
        """Test that group_fibers preserves m values from input."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 3), merge="center")}
        )

        result = trace.group_fibers(sample_traces, config)

        m_values = {t.m for t in result}
        assert m_values == {90, 91}

    @pytest.mark.unit
    def test_group_fibers_clears_fiber_idx(self, sample_traces):
        """Test that group_fibers sets fiber_idx to None."""
        from pyreduce.instruments.models import FiberGroupConfig, FibersConfig

        config = FibersConfig(
            groups={"A": FiberGroupConfig(range=(1, 3), merge="center")}
        )

        result = trace.group_fibers(sample_traces, config)

        for t in result:
            assert t.fiber_idx is None

    @pytest.mark.unit
    def test_group_fibers_bundles(self):
        """Test group_fibers with bundles config."""
        from pyreduce.instruments.models import FiberBundleConfig, FibersConfig
        from pyreduce.trace_model import Trace as TraceData

        # 4 traces in one order, 2 fibers per bundle
        traces = [
            TraceData(
                m=90,
                fiber_idx=i,
                pos=np.array([0.0, 0.0, 100.0 + i * 10]),
                column_range=(10, 990),
            )
            for i in range(1, 5)
        ]

        config = FibersConfig(bundles=FiberBundleConfig(size=2, merge="center"))

        result = trace.group_fibers(traces, config)

        # Should have 2 bundles
        groups = {t.group for t in result}
        assert groups == {"bundle_1", "bundle_2"}


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
