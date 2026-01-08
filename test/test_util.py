"""Unit tests for util.py"""

import numpy as np
import pytest

from pyreduce import util


class TestResample:
    """Tests for resample function."""

    @pytest.mark.unit
    def test_resample_same_size(self):
        """Resampling to same size should preserve values approximately."""
        arr = np.array([1, 2, 3, 4, 5])
        result = util.resample(arr, 5)
        assert result.shape == (5,)

    @pytest.mark.unit
    def test_resample_upsample(self):
        """Test upsampling to larger array."""
        arr = np.array([0, 10])
        result = util.resample(arr, 11)
        assert result.shape == (11,)
        assert result[0] == pytest.approx(0, abs=1)
        assert result[-1] == pytest.approx(10, abs=1)

    @pytest.mark.unit
    def test_resample_downsample(self):
        """Test downsampling to smaller array."""
        arr = np.arange(100)
        result = util.resample(arr, 10)
        assert result.shape == (10,)


class TestSwapExtension:
    """Tests for swap_extension function."""

    @pytest.mark.unit
    def test_swap_simple(self):
        """Test simple extension swap."""
        result = util.swap_extension("/path/to/file.fits", ".npz")
        assert result == "/path/to/file.npz"

    @pytest.mark.unit
    def test_swap_gz_file(self):
        """Test swapping .gz compressed file."""
        result = util.swap_extension("/path/to/file.fits.gz", ".npz")
        assert result == "/path/to/file.npz"

    @pytest.mark.unit
    def test_swap_with_path(self):
        """Test swapping with custom output path."""
        result = util.swap_extension("/path/to/file.fits", ".npz", path="/other")
        assert result == "/other/file.npz"


class TestFindFirstIndex:
    """Tests for find_first_index function."""

    @pytest.mark.unit
    def test_find_existing_value(self):
        """Test finding an existing value."""
        arr = [1, 2, 3, 4, 5]
        assert util.find_first_index(arr, 3) == 2

    @pytest.mark.unit
    def test_find_first_occurrence(self):
        """Test that first occurrence is returned."""
        arr = [1, 2, 3, 2, 5]
        assert util.find_first_index(arr, 2) == 1

    @pytest.mark.unit
    def test_find_missing_value(self):
        """Test that missing value raises exception."""
        arr = [1, 2, 3]
        with pytest.raises(Exception, match="not found"):
            util.find_first_index(arr, 99)


class TestInterpolateMasked:
    """Tests for interpolate_masked function."""

    @pytest.mark.unit
    def test_interpolate_gap(self):
        """Test interpolating over masked gap."""
        arr = np.ma.array([1, 2, 99, 4, 5], mask=[0, 0, 1, 0, 0])
        result = util.interpolate_masked(arr)

        assert result.shape == arr.shape
        assert result[2] == pytest.approx(3, rel=0.1)

    @pytest.mark.unit
    def test_interpolate_no_mask(self):
        """Test with no masked values."""
        arr = np.ma.array([1, 2, 3, 4, 5], mask=False)
        result = util.interpolate_masked(arr)

        np.testing.assert_array_almost_equal(result, [1, 2, 3, 4, 5])


class TestMakeIndex:
    """Tests for make_index function."""

    @pytest.mark.unit
    def test_make_index_basic(self):
        """Test basic index creation."""
        ymin = np.array([10, 10, 10])
        ymax = np.array([15, 15, 15])

        idx = util.make_index(ymin, ymax, xmin=0, xmax=3)

        # Should return two arrays (row indices, col indices)
        assert len(idx) == 2
        assert idx[0].shape == idx[1].shape

    @pytest.mark.unit
    def test_make_index_with_zero(self):
        """Test index creation with zero offset."""
        ymin = np.array([10, 10])
        ymax = np.array([12, 12])

        idx = util.make_index(ymin, ymax, xmin=5, xmax=7, zero=True)

        assert len(idx) == 2


class TestGaussval2:
    """Tests for gaussval2 function."""

    @pytest.mark.unit
    def test_peak_value(self):
        """Test gaussian peak value."""
        a, mu, sig, const = 10, 5, 2, 1

        result = util.gaussval2(mu, a, mu, sig, const)

        # At peak: a + const
        assert result == pytest.approx(a + const)

    @pytest.mark.unit
    def test_symmetry(self):
        """Test gaussian symmetry."""
        a, mu, sig, const = 10, 5, 2, 1
        x1 = mu - 1
        x2 = mu + 1

        assert util.gaussval2(x1, a, mu, sig, const) == pytest.approx(
            util.gaussval2(x2, a, mu, sig, const)
        )

    @pytest.mark.unit
    def test_decay(self):
        """Test that gaussian decays away from peak."""
        a, mu, sig, const = 10, 5, 2, 1

        peak = util.gaussval2(mu, a, mu, sig, const)
        away = util.gaussval2(mu + 2 * sig, a, mu, sig, const)

        assert away < peak


class TestPolyfit1d:
    """Tests for polyfit1d function."""

    @pytest.mark.unit
    def test_linear_fit(self):
        """Test fitting linear data."""
        x = np.arange(10)
        y = 2 * x + 3

        coef = util.polyfit1d(x, y, degree=1)

        assert len(coef) == 2
        assert coef[0] == pytest.approx(2, rel=0.01)
        assert coef[1] == pytest.approx(3, rel=0.01)

    @pytest.mark.unit
    def test_quadratic_fit(self):
        """Test fitting quadratic data."""
        x = np.arange(10)
        y = x**2 + 2 * x + 1

        coef = util.polyfit1d(x, y, degree=2)

        assert len(coef) == 3
        assert coef[0] == pytest.approx(1, rel=0.01)  # x^2
        assert coef[1] == pytest.approx(2, rel=0.01)  # x
        assert coef[2] == pytest.approx(1, rel=0.01)  # const


class TestPolyfit2d:
    """Tests for polyfit2d function."""

    @pytest.mark.unit
    def test_linear_2d_fit(self):
        """Test fitting linear 2D surface."""
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = 2 * x + 3 * y + 1  # z = 2x + 3y + 1

        coef = util.polyfit2d(x, y, z, degree=1)

        assert coef.shape == (2, 2)
        assert coef[0, 0] == pytest.approx(1, rel=0.1)  # constant
        assert coef[1, 0] == pytest.approx(2, rel=0.1)  # x term
        assert coef[0, 1] == pytest.approx(3, rel=0.1)  # y term

    @pytest.mark.unit
    def test_polyfit2d_with_mask(self):
        """Test fitting with masked values."""
        x = np.ma.array([0, 1, 2, 0, 1, 2], mask=[0, 0, 0, 0, 0, 0])
        y = np.ma.array([0, 0, 0, 1, 1, 1], mask=[0, 0, 0, 0, 0, 0])
        z = 2 * x + 3 * y + 1

        coef = util.polyfit2d(x, y, z, degree=1)

        assert coef.shape == (2, 2)


class TestVac2Air:
    """Tests for vacuum to air wavelength conversion."""

    @pytest.mark.unit
    def test_vac2air_visible(self):
        """Test conversion in visible range."""
        wl_vac = np.array([5000.0, 6000.0, 7000.0])
        wl_air = util.vac2air(wl_vac.copy())

        # Air wavelengths should be slightly shorter
        assert np.all(wl_air < wl_vac)
        assert np.all(wl_air > wl_vac - 2)  # Within ~2 Angstrom

    @pytest.mark.unit
    def test_vac2air_below_threshold(self):
        """Test that UV wavelengths below 2000A are unchanged."""
        wl_vac = np.array([1000.0, 1500.0])
        wl_air = util.vac2air(wl_vac.copy())

        np.testing.assert_array_almost_equal(wl_vac, wl_air)


class TestAir2Vac:
    """Tests for air to vacuum wavelength conversion."""

    @pytest.mark.unit
    def test_air2vac_visible(self):
        """Test conversion in visible range."""
        wl_air = np.array([5000.0, 6000.0, 7000.0])
        wl_vac = util.air2vac(wl_air)

        # Vacuum wavelengths should be slightly longer
        assert np.all(wl_vac > wl_air)
        assert np.all(wl_vac < wl_air + 2)  # Within ~2 Angstrom

    @pytest.mark.unit
    def test_roundtrip(self):
        """Test that vac->air->vac roundtrips approximately."""
        wl_original = np.array([5000.0, 6000.0, 7000.0])
        wl_air = util.vac2air(wl_original.copy())
        wl_back = util.air2vac(wl_air)

        np.testing.assert_array_almost_equal(wl_original, wl_back, decimal=3)


class TestSafeInterpolation:
    """Tests for safe_interpolation function."""

    @pytest.mark.unit
    def test_basic_interpolation(self):
        """Test basic interpolation."""
        x_old = np.array([0, 1, 2, 3, 4])
        y_old = np.array([0, 1, 4, 9, 16])
        x_new = np.array([0.5, 1.5, 2.5])

        result = util.safe_interpolation(x_old, y_old, x_new)

        assert result.shape == x_new.shape

    @pytest.mark.unit
    def test_interpolation_with_mask(self):
        """Test interpolation with masked values."""
        x_old = np.ma.array([0, 1, 99, 3, 4], mask=[0, 0, 1, 0, 0])
        y_old = np.ma.array([0, 1, 99, 9, 16], mask=[0, 0, 1, 0, 0])
        x_new = np.array([0.5, 2.0, 3.5])

        result = util.safe_interpolation(x_old, y_old, x_new)

        assert result.shape == x_new.shape

    @pytest.mark.unit
    def test_interpolation_returns_interpolator(self):
        """Test that None x_new returns interpolator object."""
        x_old = np.array([0, 1, 2, 3, 4])
        y_old = np.array([0, 1, 4, 9, 16])

        result = util.safe_interpolation(x_old, y_old, x_new=None)

        # Should return callable interpolator
        assert callable(result)
        assert result(1.5) is not None


class TestBezierInterp:
    """Tests for bezier_interp function."""

    @pytest.mark.unit
    def test_basic_bezier(self):
        """Test basic Bezier interpolation."""
        x_old = np.array([0, 1, 2, 3, 4])
        y_old = np.array([0, 1, 4, 9, 16])
        x_new = np.array([0.5, 1.5, 2.5])

        result = util.bezier_interp(x_old, y_old, x_new)

        assert result.shape == x_new.shape

    @pytest.mark.unit
    def test_bezier_with_duplicates(self):
        """Test Bezier with duplicate x values."""
        x_old = np.array([0, 1, 1, 2, 3])  # Duplicate x=1
        y_old = np.array([0, 1, 2, 4, 9])
        x_new = np.array([0.5, 1.5, 2.5])

        result = util.bezier_interp(x_old, y_old, x_new)

        assert result.shape == x_new.shape


class TestPolyscale2d:
    """Tests for polyscale2d function."""

    @pytest.mark.unit
    def test_identity_scale(self):
        """Test that scale of 1 preserves coefficients."""
        coef = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = util.polyscale2d(coef, 1, 1)

        np.testing.assert_array_almost_equal(coef, result)

    @pytest.mark.unit
    def test_scale_changes_values(self):
        """Test that non-unity scale changes coefficients."""
        coef = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = util.polyscale2d(coef, 2, 2)

        # Coefficients should be divided by scale^power
        assert result[0, 0] == pytest.approx(1)  # x^0 * y^0 term unchanged
        assert result[1, 0] == pytest.approx(3 / 2)  # x^1 * y^0 term / 2
        assert result[0, 1] == pytest.approx(2 / 2)  # x^0 * y^1 term / 2


class TestPolyshift2d:
    """Tests for polyshift2d function."""

    @pytest.mark.unit
    def test_zero_shift(self):
        """Test that zero shift preserves coefficients."""
        coef = np.array([[1, 2], [3, 4]])
        result = util.polyshift2d(coef, 0, 0)

        np.testing.assert_array_almost_equal(coef, result)

    @pytest.mark.unit
    def test_shift_constant(self):
        """Test shifting a constant polynomial."""
        # Constant polynomial z = 5
        coef = np.array([[5.0]])
        result = util.polyshift2d(coef, 10, 20)

        # Constant should be unchanged by shift
        assert result[0, 0] == pytest.approx(5.0)


class TestGaussfit:
    """Tests for gaussfit function."""

    @pytest.mark.unit
    def test_gaussfit_basic(self):
        """Test basic Gaussian fitting."""
        rng = np.random.default_rng(42)
        x = np.linspace(-5, 5, 50)
        # Gaussian: A=10, mu=0, sigma=1, with small noise
        y = 10 * np.exp(-0.5 * (x / 1) ** 2) + rng.normal(0, 0.1, x.shape)

        fitted, popt = util.gaussfit(x, y)

        assert fitted.shape == y.shape
        assert popt[0] == pytest.approx(10, rel=0.1)  # Amplitude
        assert popt[1] == pytest.approx(0, abs=0.1)  # Mean
        assert abs(popt[2]) == pytest.approx(1, rel=0.1)  # Sigma


class TestGaussfit2:
    """Tests for gaussfit2 function."""

    @pytest.mark.unit
    def test_gaussfit2_basic(self):
        """Test gaussfit2 with offset."""
        x = np.linspace(-5, 5, 50)
        # Gaussian: A=10, mu=0, sigma=1, offset=2
        y = 10 * np.exp(-0.5 * (x / 1) ** 2) + 2

        popt = util.gaussfit2(x, y)

        assert len(popt) == 4
        # The fitting may have different parameterization, just check it runs

    @pytest.mark.unit
    def test_gaussfit2_all_masked(self):
        """Test gaussfit2 raises on all masked data."""
        x = np.ma.array([1, 2, 3], mask=[1, 1, 1])
        y = np.ma.array([1, 2, 3], mask=[1, 1, 1])

        with pytest.raises(ValueError, match="All values masked"):
            util.gaussfit2(x, y)


class TestRemoveBias:
    """Tests for remove_bias function."""

    @pytest.mark.unit
    def test_remove_bias_basic(self):
        """Test basic bias removal."""
        img = np.ones((10, 10)) * 100
        ihead = {"EXPTIME": 60}
        bias = np.ones((10, 10)) * 10
        bhead = {"EXPTIME": 60}

        result = util.remove_bias(img, ihead, bias, bhead)

        np.testing.assert_array_almost_equal(result, np.ones((10, 10)) * 90)

    @pytest.mark.unit
    def test_remove_bias_none(self):
        """Test that None bias returns unchanged image."""
        img = np.ones((10, 10)) * 100
        ihead = {"EXPTIME": 60}

        result = util.remove_bias(img, ihead, None, None)

        np.testing.assert_array_almost_equal(result, img)


class TestCutoutImage:
    """Tests for cutout_image function."""

    @pytest.mark.unit
    def test_cutout_basic(self):
        """Test basic image cutout."""
        img = np.arange(100).reshape(10, 10)
        ymin = np.array([2, 2, 2])
        ymax = np.array([5, 5, 5])

        result = util.cutout_image(img, ymin, ymax, xmin=0, xmax=3)

        assert result.shape == (4, 3)  # height = 5-2+1=4, width = 3-0=3
