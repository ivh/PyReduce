import numpy as np
import pytest

from pyreduce.combine_frames import combine_frames
from pyreduce.slit_curve import Curvature as CurvatureModule
from pyreduce.slit_curve import gaussian, lorentzian


class TestGaussianLorentzian:
    """Unit tests for peak functions."""

    @pytest.mark.unit
    def test_gaussian_at_peak(self):
        """Gaussian should return A at x=mu."""
        A, mu, sig = 10.0, 5.0, 2.0
        assert gaussian(mu, A, mu, sig) == pytest.approx(A)

    @pytest.mark.unit
    def test_gaussian_symmetric(self):
        """Gaussian should be symmetric around mu."""
        A, mu, sig = 10.0, 5.0, 2.0
        x_left = mu - 1.5
        x_right = mu + 1.5
        assert gaussian(x_left, A, mu, sig) == pytest.approx(
            gaussian(x_right, A, mu, sig)
        )

    @pytest.mark.unit
    def test_gaussian_decays(self):
        """Gaussian should decay away from center."""
        A, mu, sig = 10.0, 5.0, 2.0
        assert gaussian(mu + sig, A, mu, sig) < A
        assert gaussian(mu + 2 * sig, A, mu, sig) < gaussian(mu + sig, A, mu, sig)

    @pytest.mark.unit
    def test_gaussian_array(self):
        """Gaussian should work with arrays."""
        x = np.array([0, 1, 2, 3, 4])
        result = gaussian(x, 10.0, 2.0, 1.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        assert result[2] == pytest.approx(10.0)  # peak at x=2

    @pytest.mark.unit
    def test_lorentzian_at_peak(self):
        """Lorentzian peak value at x0."""
        A, x0, mu = 10.0, 5.0, 2.0
        peak_val = lorentzian(x0, A, x0, mu)
        # At x=x0: A * mu / (0 + 0.25*mu^2) = A * mu / (0.25*mu^2) = 4*A/mu
        expected = 4 * A / mu
        assert peak_val == pytest.approx(expected)

    @pytest.mark.unit
    def test_lorentzian_symmetric(self):
        """Lorentzian should be symmetric around x0."""
        A, x0, mu = 10.0, 5.0, 2.0
        x_left = x0 - 1.5
        x_right = x0 + 1.5
        assert lorentzian(x_left, A, x0, mu) == pytest.approx(
            lorentzian(x_right, A, x0, mu)
        )


class TestCurvatureInit:
    """Unit tests for Curvature initialization."""

    @pytest.fixture
    def simple_orders(self):
        """Create simple polynomial orders for testing."""
        # 3 orders, each a polynomial of degree 2
        orders = np.array(
            [
                [100.0, 0.0, 0.0],  # y = 100
                [200.0, 0.0, 0.0],  # y = 200
                [300.0, 0.0, 0.0],  # y = 300
            ]
        )
        return orders

    @pytest.mark.unit
    def test_invalid_curve_degree(self, simple_orders):
        """curve_degree must be 1-5."""
        with pytest.raises(ValueError, match="Curvature degree must be 1-5"):
            CurvatureModule(simple_orders, curve_degree=6)
        with pytest.raises(ValueError, match="Curvature degree must be 1-5"):
            CurvatureModule(simple_orders, curve_degree=0)

    @pytest.mark.unit
    def test_invalid_mode(self, simple_orders):
        """mode must be '1D' or '2D'."""
        with pytest.raises(ValueError, match="mode"):
            CurvatureModule(simple_orders, mode="3D")

    @pytest.mark.unit
    def test_valid_modes(self, simple_orders):
        """Valid modes should not raise."""
        for mode in ["1D", "2D"]:
            module = CurvatureModule(simple_orders, mode=mode)
            assert module.mode == mode

    @pytest.mark.unit
    def test_ntrace_property(self, simple_orders):
        """ntrace should return number of traces."""
        module = CurvatureModule(simple_orders)
        assert module.ntrace == 3

    @pytest.mark.unit
    def test_n_property_with_trace_range(self, simple_orders):
        """n should return number of orders in range."""
        module = CurvatureModule(simple_orders, trace_range=(1, 3))
        assert module.n == 2

    @pytest.mark.unit
    def test_fit_degree_1d_scalar(self, simple_orders):
        """In 1D mode, fit_degree should be scalar."""
        module = CurvatureModule(simple_orders, mode="1D", fit_degree=[3, 4])
        assert module.fit_degree == 3

    @pytest.mark.unit
    def test_fit_degree_2d_tuple(self, simple_orders):
        """In 2D mode, fit_degree should be tuple."""
        module = CurvatureModule(simple_orders, mode="2D", fit_degree=3)
        assert module.fit_degree == (3, 3)


class TestCurvatureFitting:
    """Unit tests for curvature fitting functions."""

    @pytest.fixture
    def simple_orders(self):
        """Create simple polynomial orders for testing."""
        orders = np.array(
            [
                [50.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ]
        )
        return orders

    @pytest.mark.unit
    def test_fit_1d_mode(self, simple_orders):
        """Test fitting in 1D mode with synthetic data."""
        module = CurvatureModule(simple_orders, mode="1D", fit_degree=1, curve_degree=2)

        # Synthetic peaks and curvature values (now as coeffs arrays)
        peaks = [np.array([100, 200, 300]), np.array([150, 250])]
        # coeffs[i] has shape (n_peaks, curve_degree) = (n_peaks, 2)
        all_coeffs = [
            np.array([[0.01, 0.001], [0.01, 0.001], [0.01, 0.001]]),  # order 0
            np.array([[0.02, 0.002], [0.02, 0.002]]),  # order 1
        ]

        fitted_coeffs = module.fit(peaks, all_coeffs)

        # Shape: (n_orders, curve_degree, fit_degree + 1)
        assert fitted_coeffs.shape == (2, 2, 2)

    @pytest.mark.unit
    def test_eval_1d_mode(self, simple_orders):
        """Test evaluating curvature in 1D mode."""
        module = CurvatureModule(simple_orders, mode="1D", fit_degree=1, curve_degree=2)

        # fitted_coeffs has shape (n_orders, curve_degree, fit_degree + 1)
        # Each [i, j, :] is polyval coefficients for order i, curvature term j
        # [0.0, 0.01] means value = 0.0*x + 0.01 = 0.01 (constant)
        fitted_coeffs = np.array(
            [
                [[0.0, 0.01], [0.0, 0.001]],  # order 0: c1=0.01, c2=0.001
                [[0.0, 0.02], [0.0, 0.002]],  # order 1: c1=0.02, c2=0.002
            ]
        )

        peaks = np.array([100, 200])
        order = np.array([0, 1])

        coeffs = module.eval(peaks, order, fitted_coeffs)

        # coeffs has shape (n_points, curve_degree)
        assert coeffs[0, 0] == pytest.approx(0.01, rel=0.1)  # c1 for order 0
        assert coeffs[1, 0] == pytest.approx(0.02, rel=0.1)  # c1 for order 1


class TestCurvatureFitFromPositions:
    """Unit tests for _fit_curvature_from_positions method."""

    @pytest.fixture
    def simple_orders(self):
        orders = np.array([[100.0, 0.0, 0.0]])
        return orders

    @pytest.mark.unit
    def test_fit_curvature_linear(self, simple_orders):
        """Test fitting linear curvature from peak positions."""
        module = CurvatureModule(simple_orders, mode="1D", curve_degree=1)

        # Simulate peak positions that shift linearly with y-offset
        # If c1 = 0.05, then at y-offsets [-2, -1, 0, 1, 2],
        # positions shift by [-0.1, -0.05, 0, 0.05, 0.1]
        peaks = np.array([100, 200, 300])
        offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # For each peak, create positions that follow x(y) = x0 + c1*y
        c1_true = 0.05
        positions = np.zeros((3, 5))
        for i, peak in enumerate(peaks):
            positions[i, :] = peak + c1_true * offsets

        coeffs, residuals = module._fit_curvature_from_positions(
            peaks, positions, offsets
        )

        # coeffs has shape (n_peaks, curve_degree) = (3, 1)
        assert coeffs.shape == (3, 1)
        # c1 should be close to 0.05 for all peaks
        assert np.allclose(coeffs[:, 0], c1_true, atol=0.01)
        # residuals should be near zero for perfect polynomial data
        assert residuals.shape == (3, 5)
        assert np.allclose(residuals, 0, atol=0.01)

    @pytest.mark.unit
    def test_fit_curvature_quadratic(self, simple_orders):
        """Test fitting quadratic curvature from peak positions."""
        module = CurvatureModule(simple_orders, mode="1D", curve_degree=2)

        peaks = np.array([100, 200, 300])
        offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Positions follow x(y) = x0 + c1*y + c2*y^2
        c1_true = 0.05
        c2_true = 0.01
        positions = np.zeros((3, 5))
        for i, peak in enumerate(peaks):
            positions[i, :] = peak + c1_true * offsets + c2_true * offsets**2

        coeffs, residuals = module._fit_curvature_from_positions(
            peaks, positions, offsets
        )

        # coeffs has shape (n_peaks, curve_degree) = (3, 2)
        assert coeffs.shape == (3, 2)
        assert np.allclose(coeffs[:, 0], c1_true, atol=0.01)
        assert np.allclose(coeffs[:, 1], c2_true, atol=0.005)
        # residuals should be near zero for perfect polynomial data
        assert residuals.shape == (3, 5)
        assert np.allclose(residuals, 0, atol=0.01)

    @pytest.mark.unit
    def test_fit_curvature_insufficient_data(self, simple_orders):
        """Test that insufficient data returns zeros."""
        module = CurvatureModule(simple_orders, mode="1D", curve_degree=2)

        peaks = np.array([100])
        offsets = np.array([0.0, 1.0])  # Only 2 points, need 3 for quadratic
        positions = np.array([[np.nan, 100.0]])  # Only 1 valid point

        coeffs, residuals = module._fit_curvature_from_positions(
            peaks, positions, offsets
        )

        # Should return zeros when insufficient data
        assert coeffs.shape == (1, 2)
        assert coeffs[0, 0] == 0.0
        assert coeffs[0, 1] == 0.0
        # Residuals should be NaN when fit fails
        assert residuals.shape == (1, 2)
        assert np.all(np.isnan(residuals))


class TestSlitdeltasComputation:
    """Tests for slitdeltas computation from fit residuals."""

    @pytest.fixture
    def simple_orders(self):
        """Simple set of 2 polynomial trace coefficients."""
        return np.array([[50.0, 0.0], [70.0, 0.0]])

    @pytest.fixture
    def configured_module(self, simple_orders):
        """Module with extraction_height properly configured."""
        module = CurvatureModule(
            simple_orders,
            mode="1D",
            curve_degree=1,
            curve_height=10,
            extraction_height=5,
        )
        # Manually set up extraction_height (normally done in _fix_inputs)
        module.extraction_height = np.array([[5, 5], [5, 5]])
        module.trace_range = (0, 2)  # Sets n = 2 via property
        return module

    @pytest.mark.unit
    def test_compute_slitdeltas_basic(self, configured_module):
        """Test basic slitdeltas computation from residuals."""
        module = configured_module

        # Create mock offsets and residuals for 2 traces
        all_offsets = [
            np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
            np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        ]
        all_residuals = [
            np.array(
                [
                    [0.1, 0.05, 0.0, -0.05, -0.1],
                    [0.12, 0.06, 0.0, -0.06, -0.12],
                ]
            ),
            np.array(
                [
                    [0.2, 0.1, 0.0, -0.1, -0.2],
                ]
            ),
        ]

        nrows = 5
        slitdeltas = module._compute_slitdeltas(all_offsets, all_residuals, nrows)

        assert slitdeltas.shape == (2, nrows)
        assert not np.any(np.isnan(slitdeltas))

    @pytest.mark.unit
    def test_compute_slitdeltas_interpolation(self, configured_module):
        """Test that slitdeltas are correctly interpolated to nrows."""
        module = configured_module
        module.trace_range = (0, 1)
        module.extraction_height = np.array([[5, 5]])

        # Offsets must span the extraction range [-5, 5] for interpolation
        all_offsets = [np.array([-5.0, -2.5, 0.0, 2.5, 5.0])]
        all_residuals = [np.array([[0.5, 0.25, 0.0, -0.25, -0.5]])]

        nrows = 5
        slitdeltas = module._compute_slitdeltas(all_offsets, all_residuals, nrows)

        assert slitdeltas.shape == (1, nrows)
        # Should be monotonically decreasing (positive to negative)
        assert slitdeltas[0, 0] > slitdeltas[0, -1]
        # Check approximate values
        assert np.isclose(slitdeltas[0, 2], 0.0, atol=0.01)

    @pytest.mark.unit
    def test_compute_slitdeltas_empty_residuals(self, configured_module):
        """Test handling of empty residuals."""
        module = configured_module
        module.trace_range = (0, 1)
        module.extraction_height = np.array([[5, 5]])

        all_offsets = [np.array([-1.0, 0.0, 1.0])]
        all_residuals = [np.zeros((0, 3))]

        nrows = 5
        slitdeltas = module._compute_slitdeltas(all_offsets, all_residuals, nrows)

        assert slitdeltas.shape == (1, nrows)
        assert np.allclose(slitdeltas, 0)

    @pytest.mark.unit
    def test_compute_slitdeltas_nan_handling(self, configured_module):
        """Test that NaN residuals are properly ignored."""
        module = configured_module
        module.trace_range = (0, 1)
        module.extraction_height = np.array([[5, 5]])

        all_offsets = [np.array([-1.0, 0.0, 1.0])]
        all_residuals = [
            np.array(
                [
                    [0.2, 0.0, -0.2],
                    [np.nan, 0.0, np.nan],
                ]
            )
        ]

        nrows = 5
        slitdeltas = module._compute_slitdeltas(all_offsets, all_residuals, nrows)

        assert slitdeltas.shape == (1, nrows)
        assert not np.any(np.isnan(slitdeltas))

    @pytest.mark.unit
    def test_execute_returns_slitdeltas(self, simple_orders):
        """Test that execute() returns SlitCurvature with slitdeltas."""
        from pyreduce.curvature_model import SlitCurvature

        module = CurvatureModule(
            simple_orders,
            mode="1D",
            curve_degree=1,
            curve_height=10,
            extraction_height=5,
        )

        nrow, ncol = 100, 500
        img = np.random.normal(100, 10, (nrow, ncol))
        for x in [100, 200, 300, 400]:
            img[:, x] += 500

        result = module.execute(img, compute_slitdeltas=True)

        assert isinstance(result, SlitCurvature)
        assert result.slitdeltas is not None
        assert result.slitdeltas.shape[0] == 2  # 2 traces

    @pytest.mark.unit
    def test_execute_without_slitdeltas(self, simple_orders):
        """Test that execute() can skip slitdeltas computation."""
        module = CurvatureModule(
            simple_orders,
            mode="1D",
            curve_degree=1,
            curve_height=10,
            extraction_height=5,
        )

        nrow, ncol = 100, 500
        img = np.random.normal(100, 10, (nrow, ncol))

        result = module.execute(img, compute_slitdeltas=False)

        assert result.slitdeltas is None


# Tests that require instrument data follow below


@pytest.fixture
def original(files, instrument, channel, mask):
    if len(files["curvature"]) == 0:
        return None, None

    files = files["curvature"]
    original, chead = combine_frames(files, instrument, channel, mask=mask)

    return original, chead


@pytest.mark.slow
def test_curvature(original, orders, trace_range, settings):
    from pyreduce.curvature_model import SlitCurvature

    original, chead = original
    orders, column_range = orders
    settings = settings["curvature"]

    if original is None:
        pytest.skip("No curvature files")

    module = CurvatureModule(
        orders,
        column_range=column_range,
        trace_range=trace_range,
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
    curvature = module.execute(original)

    assert isinstance(curvature, SlitCurvature)
    assert curvature.degree == 2
    assert curvature.coeffs.ndim == 3
    assert curvature.coeffs.shape[0] == trace_range[1] - trace_range[0]
    assert curvature.coeffs.shape[1] == original.shape[1]
    assert curvature.coeffs.shape[2] == 3  # degree + 1

    # Test backward compatibility
    p1, p2 = curvature.to_p1_p2()
    assert isinstance(p1, np.ndarray)
    assert p2.ndim == 2
    assert p2.shape[0] == trace_range[1] - trace_range[0]
    assert p2.shape[1] == original.shape[1]

    # Reduce the number of orders this way
    orders = orders[trace_range[0] : trace_range[1]]
    column_range = column_range[trace_range[0] : trace_range[1]]

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
    curvature = module.execute(original)

    assert curvature is not None
    assert curvature.coeffs.ndim == 3
    assert curvature.coeffs.shape[0] == trace_range[1] - trace_range[0]
    assert curvature.coeffs.shape[1] == original.shape[1]
    assert curvature.degree == 1


@pytest.mark.slow
def test_curvature_exception(original, orders, trace_range):
    original, chead = original
    orders, column_range = orders

    if original is None:
        pytest.skip("No curvature files")

    orders = orders[trace_range[0] : trace_range[1]]
    column_range = column_range[trace_range[0] : trace_range[1]]

    original = np.copy(original)

    # Wrong curve_degree input (must be 1-5)
    with pytest.raises(ValueError):
        module = CurvatureModule(
            orders, column_range=column_range, plot=False, curve_degree=6
        )
        module.execute(original)

    # Wrong mode
    with pytest.raises(ValueError):
        module = CurvatureModule(
            orders, column_range=column_range, plot=False, mode="3D"
        )
        module.execute(original)


@pytest.mark.slow
def test_curvature_zero(original, orders, trace_range):
    original, chead = original
    orders, column_range = orders

    if original is None:
        pytest.skip("No curvature files")
    orders = orders[trace_range[0] : trace_range[1]]
    column_range = column_range[trace_range[0] : trace_range[1]]

    original = np.zeros_like(original)

    # With zero image, should produce zero curvature
    module = CurvatureModule(
        orders, column_range=column_range, plot=False, sigma_cutoff=0
    )
    _ = module.execute(original)
