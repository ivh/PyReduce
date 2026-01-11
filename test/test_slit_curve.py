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
        """curve_degree must be 1 or 2."""
        with pytest.raises(ValueError, match="curvature degrees"):
            CurvatureModule(simple_orders, curve_degree=3)

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
    def test_n_property_with_order_range(self, simple_orders):
        """n should return number of orders in range."""
        module = CurvatureModule(simple_orders, order_range=(1, 3))
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
        module = CurvatureModule(simple_orders, mode="1D", fit_degree=1)

        # Synthetic peaks and curvature values
        peaks = [np.array([100, 200, 300]), np.array([150, 250])]
        p1 = [np.array([0.01, 0.01, 0.01]), np.array([0.02, 0.02])]
        p2 = [np.array([0.001, 0.001, 0.001]), np.array([0.002, 0.002])]

        coef_p1, coef_p2 = module.fit(peaks, p1, p2)

        assert coef_p1.shape == (2, 2)  # 2 orders, degree 1 + 1
        assert coef_p2.shape == (2, 2)

    @pytest.mark.unit
    def test_eval_1d_mode(self, simple_orders):
        """Test evaluating curvature in 1D mode."""
        module = CurvatureModule(simple_orders, mode="1D", fit_degree=1)

        # Coefficients in polyval format: [high_degree, ..., constant]
        # [0.0, 0.01] means p1 = 0.0*x + 0.01 = 0.01 (constant)
        coef_p1 = np.array([[0.0, 0.01], [0.0, 0.02]])
        coef_p2 = np.array([[0.0, 0.001], [0.0, 0.002]])

        peaks = np.array([100, 200])
        order = np.array([0, 1])

        p1, p2 = module.eval(peaks, order, coef_p1, coef_p2)

        assert p1[0] == pytest.approx(0.01, rel=0.1)
        assert p1[1] == pytest.approx(0.02, rel=0.1)


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
        # If p1 = 0.05, then at y-offsets [-2, -1, 0, 1, 2],
        # positions shift by [-0.1, -0.05, 0, 0.05, 0.1]
        peaks = np.array([100, 200, 300])
        offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # For each peak, create positions that follow x(y) = x0 + p1*y
        p1_true = 0.05
        positions = np.zeros((3, 5))
        for i, peak in enumerate(peaks):
            positions[i, :] = peak + p1_true * offsets

        p1, p2 = module._fit_curvature_from_positions(peaks, positions, offsets)

        # p1 should be close to 0.05 for all peaks
        assert np.allclose(p1, p1_true, atol=0.01)
        # p2 should be close to 0 for linear case
        assert np.allclose(p2, 0.0, atol=0.001)

    @pytest.mark.unit
    def test_fit_curvature_quadratic(self, simple_orders):
        """Test fitting quadratic curvature from peak positions."""
        module = CurvatureModule(simple_orders, mode="1D", curve_degree=2)

        peaks = np.array([100, 200, 300])
        offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Positions follow x(y) = x0 + p1*y + p2*y^2
        p1_true = 0.05
        p2_true = 0.01
        positions = np.zeros((3, 5))
        for i, peak in enumerate(peaks):
            positions[i, :] = peak + p1_true * offsets + p2_true * offsets**2

        p1, p2 = module._fit_curvature_from_positions(peaks, positions, offsets)

        assert np.allclose(p1, p1_true, atol=0.01)
        assert np.allclose(p2, p2_true, atol=0.005)

    @pytest.mark.unit
    def test_fit_curvature_insufficient_data(self, simple_orders):
        """Test that insufficient data returns zeros."""
        module = CurvatureModule(simple_orders, mode="1D", curve_degree=2)

        peaks = np.array([100])
        offsets = np.array([0.0, 1.0])  # Only 2 points, need 3 for quadratic
        positions = np.array([[np.nan, 100.0]])  # Only 1 valid point

        p1, p2 = module._fit_curvature_from_positions(peaks, positions, offsets)

        # Should return zeros when insufficient data
        assert p1[0] == 0.0
        assert p2[0] == 0.0


# Tests that require instrument data follow below


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
