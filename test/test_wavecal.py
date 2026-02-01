import numpy as np
import pytest

from pyreduce.extract import extract
from pyreduce.wavelength_calibration import LineList, WavelengthCalibration


class TestLineList:
    """Unit tests for LineList class."""

    @pytest.mark.unit
    def test_empty_linelist(self):
        """Test creating empty LineList."""
        ll = LineList()
        assert len(ll) == 0

    @pytest.mark.unit
    def test_from_list(self):
        """Test creating LineList from lists."""
        wave = [5000.0, 5100.0, 5200.0]
        order = [0, 0, 0]
        pos = [100, 200, 300]
        width = [3, 3, 3]
        height = [1.0, 0.8, 0.6]
        flag = [True, True, True]

        ll = LineList.from_list(wave, order, pos, width, height, flag)

        assert len(ll) == 3
        assert ll["wlc"][0] == pytest.approx(5000.0)
        assert ll["order"][0] == 0
        assert ll["posm"][0] == pytest.approx(100)

    @pytest.mark.unit
    def test_add_line(self):
        """Test adding a single line."""
        ll = LineList()
        ll.add_line(wave=5000.0, order=0, pos=100, width=3, height=1.0, flag=True)

        assert len(ll) == 1
        assert ll["wlc"][0] == pytest.approx(5000.0)

    @pytest.mark.unit
    def test_append(self):
        """Test appending two linelists."""
        ll1 = LineList.from_list([5000], [0], [100], [3], [1.0], [True])
        ll2 = LineList.from_list([6000], [1], [200], [3], [0.8], [True])

        ll1.append(ll2)

        assert len(ll1) == 2

    @pytest.mark.unit
    def test_getitem(self):
        """Test indexing LineList."""
        ll = LineList.from_list(
            [5000, 6000], [0, 1], [100, 200], [3, 3], [1, 0.8], [True, True]
        )

        assert ll["wlc"][0] == pytest.approx(5000)
        assert ll["order"][1] == 1

    @pytest.mark.unit
    def test_setitem(self):
        """Test setting values in LineList."""
        ll = LineList.from_list([5000], [0], [100], [3], [1.0], [True])
        ll["flag"][0] = False

        assert not ll["flag"][0]


class TestWavelengthCalibrationInit:
    """Unit tests for WavelengthCalibration initialization."""

    @pytest.mark.unit
    def test_default_init(self):
        """Test default initialization."""
        wc = WavelengthCalibration(plot=False)

        assert wc.threshold == 100
        assert wc.iterations == 3
        assert wc.dimensionality == "2D"

    @pytest.mark.unit
    def test_1d_mode(self):
        """Test 1D mode initialization."""
        wc = WavelengthCalibration(dimensionality="1D", degree=4, plot=False)

        assert wc.dimensionality == "1D"
        assert wc.degree == 4

    @pytest.mark.unit
    def test_2d_mode(self):
        """Test 2D mode initialization."""
        wc = WavelengthCalibration(dimensionality="2D", degree=(4, 5), plot=False)

        assert wc.dimensionality == "2D"
        assert wc.degree == (4, 5)

    @pytest.mark.unit
    def test_invalid_dimensionality(self):
        """Test that invalid dimensionality raises error."""
        with pytest.raises(ValueError, match="dimensionality"):
            WavelengthCalibration(dimensionality="3D", plot=False)

    @pytest.mark.unit
    def test_step_mode(self):
        """Test step mode property."""
        wc = WavelengthCalibration(nstep=0, plot=False)
        assert not wc.step_mode

        wc = WavelengthCalibration(nstep=5, plot=False)
        assert wc.step_mode


class TestWavelengthCalibrationEvaluate:
    """Unit tests for WavelengthCalibration.evaluate_solution."""

    @pytest.mark.unit
    def test_evaluate_1d_solution(self):
        """Test evaluating 1D solution."""
        wc = WavelengthCalibration(dimensionality="1D", degree=2, plot=False)

        # Solution: order 0 has wavelength = 5000 + 0.1*x
        # order 1 has wavelength = 6000 + 0.1*x
        solution = np.array(
            [
                [0.0, 0.1, 5000.0],  # Order 0
                [0.0, 0.1, 6000.0],  # Order 1
            ]
        )

        pos = np.array([0, 100, 0, 100])
        order = np.array([0, 0, 1, 1])

        result = wc.evaluate_solution(pos, order, solution)

        assert result[0] == pytest.approx(5000.0)
        assert result[1] == pytest.approx(5010.0)
        assert result[2] == pytest.approx(6000.0)
        assert result[3] == pytest.approx(6010.0)

    @pytest.mark.unit
    def test_evaluate_2d_solution(self):
        """Test evaluating 2D solution."""
        wc = WavelengthCalibration(dimensionality="2D", degree=(1, 1), plot=False)

        # Solution: wavelength = 5000 + 0.1*x + 1000*order
        # polyval2d format: coef[i,j] for x^i * y^j
        solution = np.array(
            [
                [5000.0, 1000.0],  # constant + order term
                [0.1, 0.0],  # x term
            ]
        )

        pos = np.array([0, 100, 0, 100])
        order = np.array([0, 0, 1, 1])

        result = wc.evaluate_solution(pos, order, solution)

        assert result[0] == pytest.approx(5000.0)
        assert result[1] == pytest.approx(5010.0)
        assert result[2] == pytest.approx(6000.0)
        assert result[3] == pytest.approx(6010.0)

    @pytest.mark.unit
    def test_evaluate_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        wc = WavelengthCalibration(dimensionality="1D", degree=2, plot=False)
        solution = np.array([[0.0, 0.1, 5000.0]])

        pos = np.array([0, 1, 2])
        order = np.array([0, 0])  # Different shape

        with pytest.raises(ValueError, match="same shape"):
            wc.evaluate_solution(pos, order, solution)


class TestWavelengthCalibrationResidual:
    """Unit tests for WavelengthCalibration.calculate_residual."""

    @pytest.mark.unit
    def test_calculate_residual(self):
        """Test residual calculation."""
        wc = WavelengthCalibration(dimensionality="1D", degree=2, plot=False)

        # Solution: wavelength = 5000 + 0.1*x
        solution = np.array([[0.0, 0.1, 5000.0]])

        # Lines with known positions
        lines = np.zeros(3, dtype=LineList.dtype)
        lines["posm"] = [0, 100, 200]
        lines["order"] = [0, 0, 0]
        lines["wll"] = [5000.0, 5010.0, 5020.0]  # Expected wavelengths
        lines["flag"] = [True, True, True]

        residual = wc.calculate_residual(solution, lines)

        # Residuals should be zero (solution matches expected)
        assert np.all(np.abs(residual) < 1)  # Less than 1 m/s

    @pytest.mark.unit
    def test_residual_with_offset(self):
        """Test residual when solution doesn't match."""
        wc = WavelengthCalibration(dimensionality="1D", degree=2, plot=False)

        # Solution: wavelength = 5000 + 0.1*x
        solution = np.array([[0.0, 0.1, 5000.0]])

        lines = np.zeros(1, dtype=LineList.dtype)
        lines["posm"] = [100]
        lines["order"] = [0]
        lines["wll"] = [5015.0]  # Expected 5010 from solution, so 5 A offset
        lines["flag"] = [True]

        residual = wc.calculate_residual(solution, lines)

        # Residual = (5010 - 5015) / 5015 * c ~ -300 km/s
        assert residual[0] < 0
        assert np.abs(residual[0]) > 100000  # Large residual


class TestWavelengthCalibrationNormalize:
    """Unit tests for WavelengthCalibration.normalize."""

    @pytest.mark.unit
    def test_normalize_line_heights(self):
        """Test that line heights are normalized per order."""
        wc = WavelengthCalibration(plot=False, closing=0)

        # Larger spectrum with positive values
        obs = np.array([[10.0, 50.0, 100.0, 80.0, 30.0, 20.0, 60.0, 90.0]])
        lines = np.zeros(2, dtype=LineList.dtype)
        lines["order"] = [0, 0]
        lines["height"] = [10.0, 5.0]  # Heights to be normalized

        obs_norm, lines_norm = wc.normalize(obs, lines)

        # Line heights should be normalized per order (max=1)
        assert lines_norm["height"][0] == pytest.approx(1.0)
        assert lines_norm["height"][1] == pytest.approx(0.5)


class TestWavelengthCalibrationCreateImage:
    """Unit tests for WavelengthCalibration.create_image_from_lines."""

    @pytest.mark.unit
    def test_create_image_basic(self):
        """Test creating reference image from lines."""
        wc = WavelengthCalibration(plot=False)
        wc.nord = 2
        wc.ncol = 100

        lines = np.zeros(2, dtype=LineList.dtype)
        lines["order"] = [0, 1]
        lines["xfirst"] = [40, 60]
        lines["xlast"] = [50, 70]
        lines["width"] = [3, 3]
        lines["height"] = [1.0, 0.8]

        img = wc.create_image_from_lines(lines)

        assert img.shape == (2, 100)
        # Lines should create non-zero regions
        assert np.any(img[0, 40:50] > 0)
        assert np.any(img[1, 60:70] > 0)


class TestWavelengthCalibrationApplyOffset:
    """Unit tests for WavelengthCalibration.apply_alignment_offset."""

    @pytest.mark.unit
    def test_apply_offset(self):
        """Test applying alignment offset to lines."""
        wc = WavelengthCalibration(plot=False)

        lines = np.zeros(2, dtype=LineList.dtype)
        lines["order"] = [0, 1]
        lines["xfirst"] = [10, 20]
        lines["xlast"] = [20, 30]
        lines["posm"] = [15, 25]

        offset = (1, 5)  # Shift order by 1, x by 5
        lines = wc.apply_alignment_offset(lines, offset)

        assert lines["order"][0] == 1
        assert lines["order"][1] == 2
        assert lines["xfirst"][0] == 15
        assert lines["xlast"][0] == 25
        assert lines["posm"][0] == 20


class TestWavelengthCalibrationMakeWave:
    """Unit tests for WavelengthCalibration.make_wave."""

    @pytest.mark.unit
    def test_make_wave_1d(self):
        """Test creating wavelength image in 1D mode."""
        wc = WavelengthCalibration(dimensionality="1D", degree=1, plot=False)
        wc.nord = 2
        wc.ncol = 100

        solution = np.array(
            [
                [0.1, 5000.0],  # Order 0: wave = 5000 + 0.1*x
                [0.1, 6000.0],  # Order 1: wave = 6000 + 0.1*x
            ]
        )

        wave_img = wc.make_wave(solution)

        assert wave_img.shape == (2, 100)
        assert wave_img[0, 0] == pytest.approx(5000.0)
        assert wave_img[0, 50] == pytest.approx(5005.0)
        assert wave_img[1, 0] == pytest.approx(6000.0)

    @pytest.mark.unit
    def test_make_wave_2d(self):
        """Test creating wavelength image in 2D mode."""
        wc = WavelengthCalibration(dimensionality="2D", degree=(1, 1), plot=False)
        wc.nord = 2
        wc.ncol = 100

        # Solution: wavelength = 5000 + 0.1*x + 1000*order
        solution = np.array(
            [
                [5000.0, 1000.0],
                [0.1, 0.0],
            ]
        )

        wave_img = wc.make_wave(solution)

        assert wave_img.shape == (2, 100)
        assert wave_img[0, 0] == pytest.approx(5000.0)
        assert wave_img[1, 0] == pytest.approx(6000.0)


# Tests that require instrument data follow below


@pytest.mark.instrument
@pytest.mark.downloads
@pytest.mark.slow
def test_wavecal(
    files, instr, instrument, channel, mask, orders, settings, trace_range
):
    name = "wavecal_master"
    if len(files[name]) == 0:
        pytest.skip(f"No wavecal files found for instrument {instrument}")

    orders, column_range = orders
    files = files[name][0]
    orig, thead = instr.load_fits(files, channel, mask=mask)
    thead["obase"] = (0, "base order number")

    # Extract wavecal spectrum
    wavecal_spec, _, _, _ = extract(
        orig,
        orders,
        gain=thead["e_gain"],
        readnoise=thead["e_readn"],
        dark=thead["e_drk"],
        extraction_type="simple",
        column_range=column_range,
        trace_range=trace_range,
        extraction_height=settings[name]["extraction_height"],
        plot=False,
    )

    assert isinstance(wavecal_spec, np.ndarray)
    assert wavecal_spec.ndim == 2
    assert wavecal_spec.shape[0] == trace_range[1] - trace_range[0]
    assert wavecal_spec.shape[1] == orig.shape[1]
    assert np.issubdtype(wavecal_spec.dtype, np.floating)

    # assert np.min(wavecal_spec) == 0
    # assert np.max(wavecal_spec) == 1

    reference = instr.get_wavecal_filename(thead, channel, **settings["instrument"])
    reference = np.load(reference, allow_pickle=True)
    linelist = reference["cs_lines"]

    name = "wavecal"
    module = WavelengthCalibration(
        plot=False,
        manual=False,
        threshold=settings[name]["threshold"],
        degree=settings[name]["degree"],
    )
    wave, solution, lines = module.execute(wavecal_spec, linelist)

    assert isinstance(wave, np.ndarray)
    assert wave.ndim == 2
    assert wave.shape[0] == trace_range[1] - trace_range[0]
    assert wave.shape[1] == orig.shape[1]
    assert np.issubdtype(wave.dtype, np.floating)
