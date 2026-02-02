"""Tests for pyreduce.trace_model module."""

import numpy as np
import pytest
from astropy.io import fits

from pyreduce.trace_model import Trace, load_traces, save_traces

pytestmark = pytest.mark.unit


class TestTraceDataclass:
    """Tests for Trace dataclass methods."""

    def test_y_at_x_linear(self):
        """y_at_x evaluates linear polynomial correctly."""
        # y = 2*x + 100
        trace = Trace(m=1, group="A", pos=np.array([2.0, 100.0]), column_range=(0, 100))
        x = np.array([0, 10, 50])
        y = trace.y_at_x(x)
        np.testing.assert_array_almost_equal(y, [100, 120, 200])

    def test_y_at_x_quadratic(self):
        """y_at_x evaluates quadratic polynomial correctly."""
        # y = x^2 + 2*x + 100
        trace = Trace(
            m=1, group="A", pos=np.array([1.0, 2.0, 100.0]), column_range=(0, 100)
        )
        x = np.array([0, 1, 10])
        y = trace.y_at_x(x)
        np.testing.assert_array_almost_equal(y, [100, 103, 220])

    def test_wlen_returns_none_when_no_wave(self):
        """wlen returns None when wave is not set."""
        trace = Trace(m=1, group="A", pos=np.array([1.0, 100.0]), column_range=(0, 100))
        assert trace.wlen(np.array([0, 50, 100])) is None

    def test_wlen_evaluates_1d_polynomial(self):
        """wlen evaluates 1D wavelength polynomial correctly."""
        # wave = 0.1*x + 5000
        trace = Trace(
            m=1,
            group="A",
            pos=np.array([1.0, 100.0]),
            column_range=(0, 100),
            wave=np.array([0.1, 5000.0]),
        )
        x = np.array([0, 100, 1000])
        wlen = trace.wlen(x)
        np.testing.assert_array_almost_equal(wlen, [5000, 5010, 5100])

    def test_wlen_evaluates_2d_polynomial(self):
        """wlen evaluates 2D wavelength polynomial using trace's m value."""
        # 2D polynomial: wave[i,j] is coeff for x^i * m^j
        # wave = c00 + c10*x + c01*m + c11*x*m
        # Using: wave = 5000 + 0.1*x + 10*m + 0.001*x*m
        wave_2d = np.array(
            [
                [5000.0, 10.0],  # c00, c01
                [0.1, 0.001],  # c10, c11
            ]
        )
        trace = Trace(
            m=85,  # physical order number
            group="A",
            pos=np.array([1.0, 100.0]),
            column_range=(0, 2000),
            wave=wave_2d,
        )
        x = np.array([0, 1000, 2000])
        wlen = trace.wlen(x)
        # At x=0: 5000 + 0 + 10*85 + 0 = 5850
        # At x=1000: 5000 + 100 + 850 + 85 = 6035
        # At x=2000: 5000 + 200 + 850 + 170 = 6220
        expected = [5850.0, 6035.0, 6220.0]
        np.testing.assert_array_almost_equal(wlen, expected)

    def test_wlen_2d_different_m_values(self):
        """Different traces with same 2D poly but different m give different wavelengths."""
        # Same polynomial, different order numbers
        wave_2d = np.array(
            [
                [5000.0, 10.0],
                [0.1, 0.0],
            ]
        )
        trace_m85 = Trace(
            m=85,
            group="A",
            pos=np.array([1.0, 100.0]),
            column_range=(0, 2000),
            wave=wave_2d,
        )
        trace_m90 = Trace(
            m=90,
            group="A",
            pos=np.array([1.0, 200.0]),
            column_range=(0, 2000),
            wave=wave_2d,
        )

        x = np.array([1000])
        wlen_85 = trace_m85.wlen(x)
        wlen_90 = trace_m90.wlen(x)

        # At x=1000, m=85: 5000 + 100 + 850 = 5950
        # At x=1000, m=90: 5000 + 100 + 900 = 6000
        np.testing.assert_array_almost_equal(wlen_85, [5950.0])
        np.testing.assert_array_almost_equal(wlen_90, [6000.0])

    def test_slit_at_x_returns_none_when_no_slit(self):
        """slit_at_x returns None when slit is not set."""
        trace = Trace(m=1, group="A", pos=np.array([1.0, 100.0]), column_range=(0, 100))
        assert trace.slit_at_x(500) is None

    def test_slit_at_x_evaluates_2d_polynomial(self):
        """slit_at_x evaluates 2D slit polynomial correctly."""
        # slit[i, :] are coefficients for y^i term as function of x
        # slit = [[0, 0.001], [0.01, 0]]
        # At x=0: c0=0, c1=0.01 -> offset = 0.01*y
        # At x=1000: c0=1, c1=0.01 -> offset = 1 + 0.01*y
        slit = np.array([[0.001, 0.0], [0.0, 0.01]])  # (deg_y+1, deg_x+1)
        trace = Trace(
            m=1,
            group="A",
            pos=np.array([1.0, 100.0]),
            column_range=(0, 2000),
            slit=slit,
        )

        coeffs_at_0 = trace.slit_at_x(0)
        np.testing.assert_array_almost_equal(coeffs_at_0, [0.0, 0.01])

        coeffs_at_1000 = trace.slit_at_x(1000)
        np.testing.assert_array_almost_equal(coeffs_at_1000, [1.0, 0.01])


class TestSaveLoadRoundtrip:
    """Tests for save_traces/load_traces roundtrip."""

    @pytest.fixture
    def minimal_trace(self):
        """Create a minimal trace with required fields only."""
        return Trace(
            m=5,
            group="A",
            pos=np.array([0.001, -0.5, 512.0]),  # quadratic
            column_range=(100, 1900),
        )

    @pytest.fixture
    def full_trace(self):
        """Create a trace with all fields populated."""
        return Trace(
            m=10,
            group="B",
            pos=np.array([0.0001, 0.002, -0.5, 600.0]),  # cubic
            column_range=(50, 1950),
            height=25.5,
            slit=np.array([[0.001, 0.0, 0.0], [0.0, 0.01, 0.0]]),  # 2x3
            slitdelta=np.array([0.1, 0.05, -0.02, 0.03, -0.01]),
            wave=np.array([0.0001, 0.1, 5000.0]),  # quadratic wavelength
        )

    def test_roundtrip_minimal_trace(self, tmp_path, minimal_trace):
        """Minimal trace survives save/load roundtrip."""
        path = tmp_path / "traces.fits"
        save_traces(path, [minimal_trace])
        loaded, header = load_traces(path)

        assert len(loaded) == 1
        t = loaded[0]
        assert t.m == minimal_trace.m
        assert t.group == minimal_trace.group
        np.testing.assert_array_almost_equal(t.pos, minimal_trace.pos)
        assert t.column_range == minimal_trace.column_range
        assert t.height is None
        assert t.slit is None
        assert t.slitdelta is None
        assert t.wave is None

    def test_roundtrip_full_trace(self, tmp_path, full_trace):
        """Full trace with all fields survives save/load roundtrip."""
        path = tmp_path / "traces.fits"
        save_traces(path, [full_trace])
        loaded, header = load_traces(path)

        assert len(loaded) == 1
        t = loaded[0]
        assert t.m == full_trace.m
        assert t.group == full_trace.group
        np.testing.assert_array_almost_equal(t.pos, full_trace.pos)
        assert t.column_range == full_trace.column_range
        assert t.height == pytest.approx(full_trace.height)
        np.testing.assert_array_almost_equal(t.slit, full_trace.slit)
        np.testing.assert_array_almost_equal(t.slitdelta, full_trace.slitdelta)
        np.testing.assert_array_almost_equal(t.wave, full_trace.wave)

    def test_roundtrip_multiple_traces_different_degrees(self, tmp_path):
        """Multiple traces with different polynomial degrees roundtrip correctly."""
        traces = [
            Trace(m=1, group="A", pos=np.array([1.0, 100.0]), column_range=(0, 1000)),
            Trace(
                m=2,
                group="B",
                pos=np.array([0.001, 2.0, 200.0]),
                column_range=(50, 950),
                wave=np.array([0.1, 5000.0]),
            ),
            Trace(
                m=3,
                group="cal",
                pos=np.array([0.0001, 0.01, 3.0, 300.0]),
                column_range=(100, 900),
                height=30.0,
            ),
        ]

        path = tmp_path / "traces.fits"
        save_traces(path, traces)
        loaded, header = load_traces(path)

        assert len(loaded) == 3
        for orig, load in zip(traces, loaded, strict=False):
            assert load.m == orig.m
            assert load.group == orig.group
            # Pos arrays may be zero-padded to max degree, check values match
            np.testing.assert_array_almost_equal(load.pos[: len(orig.pos)], orig.pos)
            assert load.column_range == orig.column_range

    def test_roundtrip_integer_fiber(self, tmp_path):
        """Integer fiber identifier survives roundtrip."""
        trace = Trace(m=1, group=42, pos=np.array([1.0, 100.0]), column_range=(0, 1000))

        path = tmp_path / "traces.fits"
        save_traces(path, [trace])
        loaded, _ = load_traces(path)

        assert loaded[0].group == 42
        assert isinstance(loaded[0].group, int)

    def test_roundtrip_none_m(self, tmp_path):
        """None spectral order number survives roundtrip."""
        trace = Trace(
            m=None, group="A", pos=np.array([1.0, 100.0]), column_range=(0, 1000)
        )

        path = tmp_path / "traces.fits"
        save_traces(path, [trace])
        loaded, _ = load_traces(path)

        assert loaded[0].m is None

    def test_header_preservation(self, tmp_path, minimal_trace):
        """FITS header is preserved through roundtrip."""
        header = fits.Header()
        header["INSTRUME"] = "TEST"
        header["OBJECT"] = "HD123456"

        path = tmp_path / "traces.fits"
        save_traces(path, [minimal_trace], header=header)
        _, loaded_header = load_traces(path)

        assert loaded_header["INSTRUME"] == "TEST"
        assert loaded_header["OBJECT"] == "HD123456"

    def test_steps_preservation(self, tmp_path, minimal_trace):
        """E_STEPS header keyword is preserved."""
        path = tmp_path / "traces.fits"
        save_traces(path, [minimal_trace], steps=["trace", "curvature", "wavecal"])
        _, header = load_traces(path)

        assert header["E_STEPS"] == "trace,curvature,wavecal"

    def test_format_version_written(self, tmp_path, minimal_trace):
        """E_FMTVER header keyword is written."""
        path = tmp_path / "traces.fits"
        save_traces(path, [minimal_trace])
        _, header = load_traces(path)

        assert header["E_FMTVER"] == 3


class TestLegacyNpzLoading:
    """Tests for loading legacy NPZ format."""

    def test_load_npz_with_traces_key(self, tmp_path):
        """Load NPZ file with 'traces' key."""
        path = tmp_path / "traces.npz"
        traces = np.array([[1.0, 100.0], [2.0, 200.0]])
        column_range = np.array([[0, 1000], [50, 950]])
        np.savez(path, traces=traces, column_range=column_range)

        loaded, header = load_traces(path)

        assert len(loaded) == 2
        np.testing.assert_array_almost_equal(loaded[0].pos, [1.0, 100.0])
        np.testing.assert_array_almost_equal(loaded[1].pos, [2.0, 200.0])
        assert loaded[0].column_range == (0, 1000)
        assert loaded[1].column_range == (50, 950)

    def test_load_npz_with_orders_key(self, tmp_path):
        """Load NPZ file with legacy 'orders' key."""
        path = tmp_path / "traces.npz"
        orders = np.array([[1.0, 100.0], [2.0, 200.0]])
        column_range = np.array([[0, 1000], [50, 950]])
        np.savez(path, orders=orders, column_range=column_range)

        loaded, header = load_traces(path)

        assert len(loaded) == 2
        np.testing.assert_array_almost_equal(loaded[0].pos, [1.0, 100.0])

    def test_load_npz_with_heights(self, tmp_path):
        """Load NPZ file with heights array."""
        path = tmp_path / "traces.npz"
        traces = np.array([[1.0, 100.0]])
        column_range = np.array([[0, 1000]])
        heights = np.array([25.5])
        np.savez(path, traces=traces, column_range=column_range, heights=heights)

        loaded, _ = load_traces(path)

        assert loaded[0].height == pytest.approx(25.5)

    def test_load_npz_without_heights(self, tmp_path):
        """Load NPZ file without heights array."""
        path = tmp_path / "traces.npz"
        traces = np.array([[1.0, 100.0]])
        column_range = np.array([[0, 1000]])
        np.savez(path, traces=traces, column_range=column_range)

        loaded, _ = load_traces(path)

        assert loaded[0].height is None

    def test_load_npz_assigns_sequential_identity(self, tmp_path):
        """NPZ loading assigns sequential m and default fiber."""
        path = tmp_path / "traces.npz"
        traces = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        column_range = np.array([[0, 1000], [0, 1000], [0, 1000]])
        np.savez(path, traces=traces, column_range=column_range)

        loaded, _ = load_traces(path)

        assert [t.m for t in loaded] == [0, 1, 2]
        assert all(t.group == 0 for t in loaded)

    def test_load_npz_returns_empty_header(self, tmp_path):
        """NPZ loading returns empty FITS header."""
        path = tmp_path / "traces.npz"
        traces = np.array([[1.0, 100.0]])
        column_range = np.array([[0, 1000]])
        np.savez(path, traces=traces, column_range=column_range)

        _, header = load_traces(path)

        assert len(header) == 0


class TestEdgeCases:
    """Edge case tests."""

    def test_save_empty_list_raises(self, tmp_path):
        """Saving empty trace list raises ValueError."""
        path = tmp_path / "traces.fits"
        with pytest.raises(ValueError, match="empty"):
            save_traces(path, [])

    def test_traces_with_varying_slit_shapes(self, tmp_path):
        """Traces with different slit polynomial shapes roundtrip correctly."""
        traces = [
            Trace(
                m=1,
                group="A",
                pos=np.array([1.0, 100.0]),
                column_range=(0, 1000),
                slit=np.array([[0.1, 0.0], [0.01, 0.0]]),  # 2x2
            ),
            Trace(
                m=2,
                group="B",
                pos=np.array([2.0, 200.0]),
                column_range=(0, 1000),
                slit=np.array(
                    [[0.1, 0.0, 0.0], [0.01, 0.0, 0.0], [0.001, 0.0, 0.0]]
                ),  # 3x3
            ),
        ]

        path = tmp_path / "traces.fits"
        save_traces(path, traces)
        loaded, _ = load_traces(path)

        assert loaded[0].slit.shape == (2, 2)
        assert loaded[1].slit.shape == (3, 3)
        np.testing.assert_array_almost_equal(loaded[0].slit, traces[0].slit)
        np.testing.assert_array_almost_equal(loaded[1].slit, traces[1].slit)

    def test_mixed_optional_fields(self, tmp_path):
        """Traces where some have optional fields and others don't."""
        traces = [
            Trace(
                m=1,
                group="A",
                pos=np.array([1.0, 100.0]),
                column_range=(0, 1000),
                wave=np.array([0.1, 5000.0]),
            ),
            Trace(
                m=2,
                group="B",
                pos=np.array([2.0, 200.0]),
                column_range=(0, 1000),
                # No wave
            ),
        ]

        path = tmp_path / "traces.fits"
        save_traces(path, traces)
        loaded, _ = load_traces(path)

        np.testing.assert_array_almost_equal(loaded[0].wave, traces[0].wave)
        assert loaded[1].wave is None

    def test_roundtrip_2d_wave_polynomial(self, tmp_path):
        """2D wavelength polynomial survives save/load roundtrip."""
        wave_2d = np.array(
            [
                [5000.0, 10.0, 0.01],  # c00, c01, c02
                [0.1, 0.001, 0.0],  # c10, c11, c12
                [1e-6, 0.0, 0.0],  # c20, c21, c22
            ]
        )
        traces = [
            Trace(
                m=85,
                group="A",
                pos=np.array([1.0, 100.0]),
                column_range=(0, 2000),
                wave=wave_2d,
            ),
            Trace(
                m=86,
                group="A",
                pos=np.array([1.0, 150.0]),
                column_range=(0, 2000),
                wave=wave_2d,
            ),
        ]

        path = tmp_path / "traces.fits"
        save_traces(path, traces)
        loaded, header = load_traces(path)

        assert len(loaded) == 2
        for orig, load in zip(traces, loaded, strict=False):
            assert load.m == orig.m
            assert load.wave.ndim == 2
            np.testing.assert_array_almost_equal(load.wave, wave_2d)

        # Verify WAVE_X and WAVE_M headers are set
        assert header.get("WAVE_X") == 3
        assert header.get("WAVE_M") == 3

    def test_wlen_consistent_after_roundtrip(self, tmp_path):
        """wlen gives same results before and after save/load."""
        wave_2d = np.array(
            [
                [5000.0, 10.0],
                [0.1, 0.001],
            ]
        )
        trace_orig = Trace(
            m=85,
            group="A",
            pos=np.array([1.0, 100.0]),
            column_range=(0, 2000),
            wave=wave_2d,
        )

        x = np.array([0, 500, 1000, 1500, 2000])
        wlen_before = trace_orig.wlen(x)

        path = tmp_path / "traces.fits"
        save_traces(path, [trace_orig])
        loaded, _ = load_traces(path)
        trace_loaded = loaded[0]

        wlen_after = trace_loaded.wlen(x)
        np.testing.assert_array_almost_equal(wlen_before, wlen_after)
