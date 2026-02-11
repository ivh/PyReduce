import numpy as np
import pytest
from scipy.signal.windows import gaussian

from pyreduce import extract
from pyreduce.trace_model import Trace

# All tests in this file are unit tests using synthetic data
pytestmark = pytest.mark.unit


@pytest.fixture
def width():
    return 100


@pytest.fixture
def height():
    return 50


@pytest.fixture(params=["linear", "sine"])
def spec(request, width):
    name = request.param

    if name == "linear":
        return 5 + np.linspace(0, 5, width)
    if name == "sine":
        return 5 + np.sin(np.linspace(0, 20 * np.pi, width))


@pytest.fixture(params=["gaussian"])
def slitf(request, height, oversample):
    name = request.param

    if name == "gaussian":
        y = gaussian(height * oversample, height / 8 * oversample)
        return y / np.sum(y)
    if name == "rectangular":
        x = np.arange(height * oversample)
        y = np.where(
            (x > height * oversample / 4) & (x < height * oversample * 3 / 4), 1, 0
        )
        return y / np.sum(y)


@pytest.fixture(params=[1, 2, 10], ids=["os=1", "os=2", "os=10"])
def oversample(request):
    return request.param


@pytest.fixture(params=["straight"], ids=["ycen=straight"])
def ycen(request, width, height):
    name = request.param
    if name == "straight":
        return np.full(width, (height - 1) / 2)

    if name == "linear":
        delta = 2
        return np.linspace(height / 2 - delta / 2, height / 2 + delta / 2, num=width)


@pytest.fixture
def orders(width, ycen):
    fit = np.polyfit(np.arange(width), ycen, deg=5)
    return np.atleast_2d(fit)


@pytest.fixture
def trace_objects(orders, width):
    """Create Trace objects from polynomial orders."""
    return [
        Trace(m=i, group=0, pos=orders[i], column_range=(0, width))
        for i in range(len(orders))
    ]


@pytest.fixture
def sample_data(width, height, spec, slitf, oversample, ycen, p1=0):
    img = spec[None, :] * slitf[:, None]
    # TODO more sophisticated sample data creation
    out = np.zeros((height, width))
    for i in range(width):
        out[:, i] = np.interp(
            np.arange(height), np.linspace(0, height, height * oversample), img[:, i]
        )

    return out, spec, slitf


def test_class_swath():
    swath = extract.Swath(5)

    assert len(swath) == 5
    assert len(swath.spec) == 5
    assert len(swath.slitf) == 5
    assert len(swath.model) == 5
    assert len(swath.unc) == 5
    assert len(swath.mask) == 5
    assert len(swath.info) == 5

    for i in range(5):
        assert swath.spec[i] is None
        assert swath.slitf[i] is None
        assert swath.model[i] is None
        assert swath.unc[i] is None
        assert swath.mask[i] is None
        assert swath.info[i] is None

        tmp = swath[i]
        for j in range(5):
            assert tmp[j] is None


def test_extend_traces():
    # Test normal case
    traces = np.array([[0.1, 5], [0.1, 7]])
    extended = extract.extend_traces(traces, 10)

    assert np.array_equal(traces, extended[1:-1])
    assert np.array_equal(extended[0], [0.1, 3])
    assert np.array_equal(extended[-1], [0.1, 9])

    # Test just one trace
    traces = np.array([0.1, 5], ndmin=2)
    extended = extract.extend_traces(traces, 10)

    assert np.array_equal(traces, extended[1:-1])
    assert np.array_equal(extended[0], [0, 0])
    assert np.array_equal(extended[-1], [0, 10])


def test_fix_column_range():
    # Some orders will be shortened
    nrow, ncol = 50, 1000
    orders = np.array([[0.2, 3], [0.2, 5], [0.2, 7], [0.2, 9]])
    ew = np.array([20, 20, 20, 20])  # full height = 20 (10 each side)
    cr = np.array([[0, 1000], [0, 1000], [0, 1000], [0, 1000]])

    fixed_cr, fixed_orders = extract.fix_column_range(cr, orders, ew, nrow, ncol)

    assert np.array_equal(fixed_cr[1], [25, 175])
    assert np.array_equal(fixed_cr[2], [15, 165])
    assert np.array_equal(fixed_cr[0], fixed_cr[1])
    assert np.array_equal(fixed_cr[-1], fixed_cr[-1])
    assert fixed_orders.shape == orders.shape  # Orders unchanged in this case

    # Nothing should change here
    orders = np.array([[20], [20], [20]])
    ew = np.array([20, 20, 20])  # full height = 20
    cr = np.array([[0, 1000], [0, 1000], [0, 1000]])

    fixed_cr, fixed_orders = extract.fix_column_range(
        np.copy(cr), orders, ew, nrow, ncol
    )
    assert np.array_equal(fixed_cr, cr)
    assert np.array_equal(fixed_orders, orders)


def test_make_bins(width):
    # swath_width, xlow, xhigh, ycen, ncol
    xlow, xhigh = 0, width
    ycen = np.linspace(0, 10, width)
    swath_width = width // 10
    nbin, bins_start, bins_end = extract.make_bins(swath_width, xlow, xhigh, ycen)

    assert isinstance(nbin, (int, np.integer))
    assert nbin == 10
    assert isinstance(bins_start, np.ndarray)
    assert isinstance(bins_end, np.ndarray)
    assert len(bins_start) == 2 * nbin - 1
    assert len(bins_end) == 2 * nbin - 1

    nbin, bins_start, bins_end = extract.make_bins(None, xlow, xhigh, ycen)

    assert isinstance(nbin, (int, np.integer))
    assert isinstance(bins_start, np.ndarray)
    assert isinstance(bins_end, np.ndarray)
    assert len(bins_start) == 2 * nbin - 1
    assert len(bins_end) == 2 * nbin - 1

    nbin, bins_start, bins_end = extract.make_bins(width * 2, xlow, xhigh, ycen)

    assert nbin == 1
    assert len(bins_start) == 1
    assert len(bins_end) == 1
    assert bins_start[0] == 0
    assert bins_end[0] == width


def test_fix_parameters():
    orders = [[0, 0, 50]]
    ncol, nrow, nord = 100, 100, 1

    # Everything None, i.e. most default settings
    # extraction_height is now full height, stored internally as half-height
    for xwd in [None, 0.4, 8, 20]:
        for cr in [None, (1, 90), [[4, 100]]]:
            xwd, cr, orders = extract.fix_parameters(xwd, cr, orders, ncol, nrow, nord)
            assert isinstance(xwd, np.ndarray)
            assert isinstance(cr, np.ndarray)
            assert isinstance(orders, np.ndarray)

            assert xwd.ndim == 1
            assert xwd.shape[0] == nord
            assert cr.ndim == 2
            assert cr.shape[0] == nord
            assert cr.shape[1] == 2
            assert orders.ndim == 2
            assert orders.shape[0] == nord
            assert orders.shape[1] == 3

    # Test that extraction_height=200 (full height, 100 per side) results in no valid pixels,
    # which now logs a warning and removes the order instead of raising ValueError
    xwd, cr, orders_out = extract.fix_parameters(200, None, orders, ncol, nrow, nord)
    # The order should be removed, resulting in an empty array
    assert len(orders_out) == 0


def test_simple_extraction(sample_data, orders, width, oversample):
    img, spec, slitf = sample_data

    extraction_height = np.array([20])  # full height = 20 pixels (10 each side)
    column_range = np.array([[0, width]])

    nord = len(orders)
    curvature = np.zeros((nord, width, 3))

    spec_out, unc_out = extract.simple_extraction(
        img, orders, extraction_height, column_range, curvature=curvature
    )

    assert isinstance(spec_out, np.ndarray)
    assert spec_out.ndim == 2
    assert spec_out.shape[0] == 1
    assert spec_out.shape[1] == width

    assert isinstance(unc_out, np.ndarray)
    assert unc_out.ndim == 2
    assert unc_out.shape[0] == 1
    assert unc_out.shape[1] == width

    assert np.abs(np.diff(spec_out / spec)).max() < 1e-8
    assert np.abs(np.diff(unc_out / spec_out)).max() < oversample / 5 + 1e-1


def test_vertical_extraction(sample_data, trace_objects, width, height, oversample):
    img, spec, slitf = sample_data

    spectra = extract.extract(img, trace_objects)

    assert isinstance(spectra, list)
    assert len(spectra) == len(trace_objects)

    for s in spectra:
        assert hasattr(s, "spec")
        assert hasattr(s, "sig")
        assert s.spec.shape == (width,)
        assert s.sig.shape == (width,)

    spec_vert = spectra[0].spec
    assert not np.all(np.isnan(spec_vert))
    valid = ~np.isnan(spec_vert)
    assert np.abs(np.diff(spec[valid] / spec_vert[valid])).max() <= 1e-1


def test_curved_equal_vertical_extraction(sample_data, orders, width):
    # Curved extraction with zero curvature should match vertical extraction
    img, spec, slitf = sample_data

    # Create traces with zero curvature
    traces_curved = [
        Trace(
            m=i,
            group=0,
            pos=orders[i],
            column_range=(0, width),
            slit=np.zeros((3, 6)),  # degree 2, 6 x-coeffs
        )
        for i in range(len(orders))
    ]
    traces_vert = [
        Trace(m=i, group=0, pos=orders[i], column_range=(0, width))
        for i in range(len(orders))
    ]

    spectra_curved = extract.extract(img, traces_curved)
    spectra_vert = extract.extract(img, traces_vert)

    spec_curved = spectra_curved[0].spec
    spec_vert = spectra_vert[0].spec
    slitf_curved = spectra_curved[0].slitfu
    slitf_vert = spectra_vert[0].slitfu

    valid = ~np.isnan(spec_curved) & ~np.isnan(spec_vert)
    assert np.allclose(spec_curved[valid], spec_vert[valid], rtol=1e-2)
    assert np.allclose(slitf_curved, slitf_vert, rtol=1e-1)


def test_optimal_extraction(sample_data, orders, height, width):
    img, spec, slitf = sample_data
    xwd = np.array([height])  # full height
    cr = np.array([[0, width]])
    # curvature shape: (ntrace, ncol, n_coeffs), all zeros for vertical extraction
    curvature = np.zeros((1, width, 3))

    res_spec, res_slitf, res_unc = extract.optimal_extraction(
        img, orders, xwd, cr, curvature=curvature
    )

    assert isinstance(res_spec, np.ndarray)
    assert isinstance(res_slitf, list)
    assert isinstance(res_unc, np.ndarray)

    assert res_spec.ndim == 2
    assert res_spec.shape[0] == 1
    assert res_spec.shape[1] == width
    assert not np.any(np.isnan(res_spec))

    assert res_unc.ndim == 2
    assert res_unc.shape[0] == 1
    assert res_unc.shape[1] == width

    assert len(res_slitf) == 1
    assert len(res_slitf[0]) != 0


def test_extract_spectrum(sample_data, orders, ycen, width, height):
    img, spec, slitf = sample_data

    column_range = np.array([[20, width]])
    extraction_height = np.array([20])  # full height

    yrange = extract.get_y_scale(ycen, column_range[0], extraction_height[0], height)
    xrange = column_range[0]

    out_spec = np.zeros(width)
    out_sunc = np.zeros(width)
    nslitf = sum(yrange) + 2 + 1  # osample=1
    out_slitf = np.zeros(nslitf)
    out_mask = np.zeros(width)

    extract.extract_spectrum(
        np.copy(img),
        np.copy(ycen),
        np.copy(yrange),
        np.copy(xrange),
        out_spec=out_spec,
        out_sunc=out_sunc,
        out_slitf=out_slitf,
        out_mask=out_mask,
    )

    assert np.any(out_spec != 0)
    assert np.any(out_sunc != 0)
    assert np.any(out_slitf != 0)
    assert np.any(out_mask)

    spec, slitf, mask, sunc = extract.extract_spectrum(
        np.copy(img), np.copy(ycen), np.copy(yrange), np.copy(xrange)
    )

    assert np.array_equal(out_spec, spec)
    assert np.array_equal(out_sunc, sunc)
    assert np.array_equal(out_slitf, slitf)
    assert np.array_equal(out_mask, mask)


def test_get_y_scale(ycen, height, width):
    xrange = (0, width)
    xwd = 20  # full height
    y_lower_lim, y_upper_lim = extract.get_y_scale(ycen, xrange, xwd, height)

    assert isinstance(y_lower_lim, int)
    assert isinstance(y_upper_lim, int)
    assert y_lower_lim >= 0
    assert y_upper_lim < height

    xwd = 4 * height  # full height = 4 * height
    y_lower_lim, y_upper_lim = extract.get_y_scale(ycen, xrange, xwd, height)
    assert isinstance(y_lower_lim, int)
    assert isinstance(y_upper_lim, int)
    assert y_lower_lim >= 0
    assert y_upper_lim < height

    ycen_tmp = ycen + height
    xwd = 20  # full height
    y_lower_lim, y_upper_lim = extract.get_y_scale(ycen_tmp, xrange, xwd, height)
    assert isinstance(y_lower_lim, int)
    assert isinstance(y_upper_lim, int)
    assert y_lower_lim >= 0
    assert y_upper_lim < height


def test_extract(sample_data, trace_objects):
    img, spec, slitf = sample_data

    with pytest.raises(ValueError):
        extract.extract(img, trace_objects, extraction_type="foobar")


class TestPresetSlitfunc:
    """Tests for preset_slitfunc functionality."""

    @pytest.fixture
    def simple_img(self):
        """Create a simple test image with known spectrum and slitfunction."""
        height, width = 21, 100
        # Gaussian slit function
        y = np.arange(height) - height // 2
        slitf = np.exp(-0.5 * (y / 3) ** 2)
        slitf /= slitf.sum()
        # Linear spectrum
        spec = 100 + np.linspace(0, 50, width)
        # Create image
        img = slitf[:, None] * spec[None, :]
        return img, spec, slitf

    @pytest.fixture
    def simple_ycen(self, simple_img):
        """Order center at middle of image."""
        img, _, _ = simple_img
        height, width = img.shape
        return np.full(width, height // 2, dtype=float)

    @pytest.fixture
    def simple_orders(self, simple_ycen):
        """Polynomial coefficients for straight trace."""
        width = len(simple_ycen)
        fit = np.polyfit(np.arange(width), simple_ycen, deg=2)
        return np.atleast_2d(fit)

    @pytest.fixture
    def simple_traces(self, simple_orders, simple_img):
        """Trace objects for the simple test case."""
        _, _, _ = simple_img
        width = 100
        return [
            Trace(m=i, group=0, pos=simple_orders[i], column_range=(0, width))
            for i in range(len(simple_orders))
        ]

    def test_preset_slitfunc_correct_size(self, simple_img, simple_ycen):
        """Test extraction with correctly sized preset slitfunc."""
        img, _, _ = simple_img
        height, width = img.shape
        osample = 5
        yrange = (5, 5)  # 5 pixels above and below
        nslitf = osample * (yrange[0] + yrange[1] + 2) + 1

        # Create a preset slitfunc of correct size
        preset = np.ones(nslitf)
        preset /= preset.sum() / osample

        xrange = [0, width]
        spec, slitf, mask, unc = extract.extract_spectrum(
            img.copy(),
            simple_ycen.copy(),
            yrange,
            xrange,
            osample=osample,
            preset_slitfunc=preset,
        )

        assert spec is not None
        assert len(spec) == width
        assert np.any(spec > 0)

    def test_preset_slitfunc_size_mismatch_error(self, simple_img, simple_ycen):
        """Test that size mismatch raises helpful error."""
        img, _, _ = simple_img
        height, width = img.shape
        osample = 5
        yrange = (5, 5)
        expected_nslitf = osample * (yrange[0] + yrange[1] + 2) + 1

        # Create preset with wrong size
        wrong_size = expected_nslitf + 10
        preset = np.ones(wrong_size)

        xrange = [0, width]
        with pytest.raises(ValueError, match="preset_slitfunc size mismatch"):
            extract.extract_spectrum(
                img.copy(),
                simple_ycen.copy(),
                yrange,
                xrange,
                osample=osample,
                preset_slitfunc=preset,
            )

    def test_preset_slitfunc_through_extract(self, simple_img, simple_traces):
        """Test preset_slitfunc passed through extract() function."""
        img, _, _ = simple_img
        height, width = img.shape
        osample = 5
        extraction_height = 5  # fixed integer

        # First extract normally to get a slitfunc
        spectra1 = extract.extract(
            img.copy(),
            simple_traces,
            extraction_type="optimal",
            extraction_height=extraction_height,
            osample=osample,
        )

        # Get slitfunc list from Spectrum objects
        slitfunc_list = [s.slitfu for s in spectra1]

        # Now extract with preset slitfunc
        spectra2 = extract.extract(
            img.copy(),
            simple_traces,
            extraction_type="optimal",
            extraction_height=extraction_height,
            osample=osample,
            preset_slitfunc=slitfunc_list,
            maxiter=1,
        )

        # Results should be similar
        assert spectra2 is not None
        assert len(spectra2) == len(spectra1)
        assert spectra2[0].spec.shape == spectra1[0].spec.shape


class TestAdaptSlitfunc:
    """Tests for _adapt_slitfunc helper function."""

    def test_same_parameters_returns_copy(self):
        """When parameters match, should return a copy."""
        from pyreduce.cwrappers import _adapt_slitfunc

        osample = 10
        yrange = (5, 5)
        nslitf = osample * (yrange[0] + yrange[1] + 2) + 1
        slitfunc = np.random.rand(nslitf)

        result = _adapt_slitfunc(slitfunc, osample, yrange, osample, yrange)

        assert result is not slitfunc  # should be a copy
        np.testing.assert_array_equal(result, slitfunc)

    def test_truncate_extraction_height(self):
        """Test truncating to smaller extraction height."""
        from pyreduce.cwrappers import _adapt_slitfunc

        osample = 10
        src_yrange = (10, 10)
        tgt_yrange = (5, 5)

        src_nslitf = osample * (src_yrange[0] + src_yrange[1] + 2) + 1
        tgt_nslitf = osample * (tgt_yrange[0] + tgt_yrange[1] + 2) + 1

        # Gaussian-like slitfunc
        src_y = np.linspace(-11, 11, src_nslitf)
        slitfunc = np.exp(-0.5 * (src_y / 3) ** 2)
        slitfunc /= slitfunc.sum() / osample

        result = _adapt_slitfunc(slitfunc, osample, src_yrange, osample, tgt_yrange)

        assert len(result) == tgt_nslitf
        # Should still be normalized to osample
        assert abs(result.sum() - osample) < 0.1

    def test_resample_osample(self):
        """Test resampling to different osample."""
        from pyreduce.cwrappers import _adapt_slitfunc

        src_osample = 10
        tgt_osample = 5
        yrange = (5, 5)

        src_nslitf = src_osample * (yrange[0] + yrange[1] + 2) + 1
        tgt_nslitf = tgt_osample * (yrange[0] + yrange[1] + 2) + 1

        # Gaussian-like slitfunc
        src_y = np.linspace(-6, 6, src_nslitf)
        slitfunc = np.exp(-0.5 * (src_y / 2) ** 2)
        slitfunc /= slitfunc.sum() / src_osample

        result = _adapt_slitfunc(slitfunc, src_osample, yrange, tgt_osample, yrange)

        assert len(result) == tgt_nslitf
        # Should be normalized to target osample
        assert abs(result.sum() - tgt_osample) < 0.1


class TestSlitdeltasExtraction:
    """Tests for slitdeltas handling in extraction."""

    def test_extract_spectrum_with_slitdeltas(self):
        """Test that slitdeltas parameter is accepted and used."""
        nrow, ncol = 50, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[20:30, :] += 100

        ycen = np.full(ncol, 25.0)
        yrange = (5, 5)
        xrange = np.array([10, 90])

        slitdeltas = np.linspace(-0.1, 0.1, 11)

        spec, slitf, mask, unc = extract.extract_spectrum(
            img,
            ycen,
            yrange,
            xrange,
            slitdeltas=slitdeltas,
            osample=1,
        )

        assert spec.shape == (ncol,)
        assert not np.all(np.isnan(spec[xrange[0] : xrange[1]]))

    def test_extract_spectrum_slitdeltas_interpolation(self):
        """Test that slitdeltas are interpolated when length differs."""
        nrow, ncol = 50, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[20:30, :] += 100

        ycen = np.full(ncol, 25.0)
        yrange = (5, 5)
        xrange = np.array([10, 90])

        # Provide different length slitdeltas
        slitdeltas = np.linspace(-0.1, 0.1, 21)

        spec, slitf, mask, unc = extract.extract_spectrum(
            img,
            ycen,
            yrange,
            xrange,
            slitdeltas=slitdeltas,
            osample=1,
        )

        assert spec.shape == (ncol,)

    def test_extract_spectrum_no_slitdeltas(self):
        """Test extraction works without slitdeltas (None)."""
        nrow, ncol = 50, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[20:30, :] += 100

        ycen = np.full(ncol, 25.0)
        yrange = (5, 5)
        xrange = np.array([10, 90])

        spec, slitf, mask, unc = extract.extract_spectrum(
            img,
            ycen,
            yrange,
            xrange,
            slitdeltas=None,
            osample=1,
        )

        assert spec.shape == (ncol,)

    def test_optimal_extraction_with_slitdeltas(self):
        """Test optimal_extraction passes slitdeltas to extract_spectrum."""
        nrow, ncol = 50, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[20:30, :] += 100  # Add signal at trace location

        # Trace polynomial: y = 0*x + 25 (constant at row 25)
        traces = np.array([[0.0, 25.0]])
        extraction_height = np.array([10])
        column_range = np.array([[0, ncol]])

        # slitdeltas has shape (ntrace, nrows)
        slitdeltas = np.zeros((1, 10))
        slitdeltas[0, :] = np.linspace(-0.05, 0.05, 10)

        spec, slitf, unc = extract.optimal_extraction(
            img,
            traces,
            extraction_height,
            column_range,
            slitdeltas=slitdeltas,
            osample=1,
        )

        assert spec.shape == (1, ncol)

    def test_extract_with_slitdeltas(self):
        """Test main extract() function passes slitdeltas through."""
        nrow, ncol = 50, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[20:30, :] += 100

        # Create Trace with slitdelta
        slitdelta = np.linspace(-0.02, 0.02, 10)
        traces = [
            Trace(
                m=0,
                group=0,
                pos=np.array([0.0, 25.0]),
                column_range=(0, ncol),
                slitdelta=slitdelta,
            )
        ]

        spectra = extract.extract(
            img,
            traces,
            extraction_height=10,
            osample=1,
        )

        assert len(spectra) == 1
        assert spectra[0].spec.shape == (ncol,)

    def test_extract_multiple_traces(self):
        """Test extraction of multiple traces."""
        nrow, ncol = 80, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[20:30, :] += 100  # First trace
        img[50:60, :] += 100  # Second trace

        # Two traces: y = 25 and y = 55
        traces = [
            Trace(m=0, group=0, pos=np.array([0.0, 25.0]), column_range=(0, ncol)),
            Trace(m=1, group=0, pos=np.array([0.0, 55.0]), column_range=(0, ncol)),
        ]

        spectra = extract.extract(
            img,
            traces,
            extraction_height=10,
            osample=1,
        )

        assert len(spectra) == 2
        assert spectra[0].spec.shape == (ncol,)
        assert spectra[1].spec.shape == (ncol,)


class TestTraceCurvatureExtraction:
    """Tests for Trace.slit curvature being used in extraction."""

    def test_extract_with_trace_slit(self):
        """Test extraction accepts and uses Trace.slit curvature."""
        nrow, ncol = 50, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[20:30, :] += 100

        # Create trace with slit curvature polynomial
        # slit[i, :] = coefficients for y^i term as function of x
        # Linear tilt: offset = 0.01 * y (slight tilt)
        slit = np.array([[0.0, 0.0], [0.0, 0.01]])  # (deg_y+1, deg_x+1)

        traces = [
            Trace(
                m=5,
                group="A",
                pos=np.array([0.0, 25.0]),
                column_range=(0, ncol),
                slit=slit,
            )
        ]

        spectra = extract.extract(
            img,
            traces,
            extraction_height=10,
            osample=1,
        )

        assert len(spectra) == 1
        assert spectra[0].spec.shape == (ncol,)
        # Verify identity preserved
        assert spectra[0].m == 5
        assert spectra[0].group == "A"

    def test_extract_with_both_slit_and_slitdelta(self):
        """Test extraction with both Trace.slit and Trace.slitdelta."""
        nrow, ncol = 50, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[20:30, :] += 100

        # Curvature polynomial
        slit = np.array([[0.0, 0.0], [0.0, 0.005]])
        # Per-row corrections
        slitdelta = np.linspace(-0.02, 0.02, 10)

        traces = [
            Trace(
                m=10,
                group="cal",
                pos=np.array([0.0, 25.0]),
                column_range=(0, ncol),
                slit=slit,
                slitdelta=slitdelta,
            )
        ]

        spectra = extract.extract(
            img,
            traces,
            extraction_height=10,
            osample=1,
        )

        assert len(spectra) == 1
        assert spectra[0].m == 10
        assert spectra[0].group == "cal"

    def test_extract_mixed_traces_some_with_curvature(self):
        """Test extraction when some traces have curvature and others don't."""
        nrow, ncol = 80, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[20:30, :] += 100
        img[50:60, :] += 100

        slit = np.array([[0.0, 0.0], [0.0, 0.01]])

        traces = [
            Trace(
                m=1,
                group="A",
                pos=np.array([0.0, 25.0]),
                column_range=(0, ncol),
                slit=slit,  # Has curvature
            ),
            Trace(
                m=2,
                group="B",
                pos=np.array([0.0, 55.0]),
                column_range=(0, ncol),
                # No curvature
            ),
        ]

        spectra = extract.extract(
            img,
            traces,
            extraction_height=10,
            osample=1,
        )

        assert len(spectra) == 2
        assert spectra[0].m == 1
        assert spectra[1].m == 2

    def test_extract_higher_degree_curvature(self):
        """Test extraction with higher-degree curvature polynomial."""
        nrow, ncol = 50, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[20:30, :] += 100

        # Quadratic curvature: offset = c0 + c1*y + c2*y^2
        # Each row is coefficients for y^i term as polynomial in x
        slit = np.array(
            [
                [0.0, 0.0, 0.0],  # y^0 term
                [0.0, 0.0, 0.01],  # y^1 term (linear tilt)
                [0.0, 0.0, 0.001],  # y^2 term (curvature)
            ]
        )

        traces = [
            Trace(
                m=1,
                group=0,
                pos=np.array([0.0, 25.0]),
                column_range=(0, ncol),
                slit=slit,
            )
        ]

        spectra = extract.extract(
            img,
            traces,
            extraction_height=10,
            osample=1,
        )

        assert len(spectra) == 1

    def test_trace_slit_at_x_used_correctly(self):
        """Verify slit_at_x evaluates curvature polynomial correctly."""
        # This tests the Trace method that extract() uses internally
        # slit[i, :] = polyval coefficients for y^i term as function of x
        # np.polyval([a, b], x) = a*x + b (highest power first)
        # So [0.0, 0.1] means 0*x + 0.1 = 0.1 (constant)
        slit = np.array(
            [
                [0.0, 0.1],  # y^0 term: constant 0.1
                [0.0, 0.02],  # y^1 term: constant 0.02
            ]
        )

        trace = Trace(
            m=1, group=0, pos=np.array([0.0, 100.0]), column_range=(0, 1000), slit=slit
        )

        # At x=0: coeffs should be [0.1, 0.02]
        coeffs = trace.slit_at_x(0)
        np.testing.assert_array_almost_equal(coeffs, [0.1, 0.02])

        # At x=500: same since polynomials are constant
        coeffs = trace.slit_at_x(500)
        np.testing.assert_array_almost_equal(coeffs, [0.1, 0.02])

        # Test with x-varying polynomial: [0.001, 0.1] means 0.001*x + 0.1
        slit_varying = np.array(
            [
                [0.001, 0.0],  # y^0 term: 0.001*x
                [0.0, 0.02],  # y^1 term: constant 0.02
            ]
        )
        trace2 = Trace(
            m=1,
            group=0,
            pos=np.array([0.0, 100.0]),
            column_range=(0, 1000),
            slit=slit_varying,
        )
        coeffs = trace2.slit_at_x(1000)
        np.testing.assert_array_almost_equal(coeffs, [1.0, 0.02])

    def test_trace_identity_preserved_through_extraction(self):
        """Verify m, fiber are copied from Trace to Spectrum."""
        nrow, ncol = 50, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[20:30, :] += 100

        traces = [
            Trace(
                m=42,
                group="science_A",
                pos=np.array([0.0, 25.0]),
                column_range=(0, ncol),
                height=12.0,
            )
        ]

        spectra = extract.extract(
            img,
            traces,
            extraction_height=10,  # Should be overridden by trace.height
            osample=1,
        )

        assert spectra[0].m == 42
        assert spectra[0].group == "science_A"

    def test_trace_height_overrides_default(self):
        """Verify Trace.height overrides default extraction_height."""
        nrow, ncol = 50, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[15:35, :] += 100  # Wider signal for larger extraction

        traces = [
            Trace(
                m=1,
                group=0,
                pos=np.array([0.0, 25.0]),
                column_range=(0, ncol),
                height=20.0,  # Override to 20 pixels
            )
        ]

        # Default is 10, but trace says 20
        spectra = extract.extract(
            img,
            traces,
            extraction_height=10,
            osample=1,
        )

        # The extraction_height used should be from trace (20)
        assert spectra[0].extraction_height == pytest.approx(20.0)

    def test_trace_wave_available_for_evaluation(self):
        """Verify Trace.wave can be evaluated after extraction.

        Note: extract() does not copy wave to Spectrum - that's done by
        pipeline steps. But the trace.wave should remain accessible for
        later evaluation.
        """
        nrow, ncol = 50, 100
        img = np.random.normal(100, 10, (nrow, ncol)).astype(np.float64)
        img[20:30, :] += 100

        # Wavelength polynomial: wave = 0.5*x + 5000
        wave_coef = np.array([0.5, 5000.0])

        traces = [
            Trace(
                m=1,
                group=0,
                pos=np.array([0.0, 25.0]),
                column_range=(0, ncol),
                wave=wave_coef,
            )
        ]

        spectra = extract.extract(
            img,
            traces,
            extraction_height=10,
            osample=1,
        )

        # Extraction doesn't copy wave, but trace.wave should still work
        x = np.arange(ncol)
        wave = traces[0].wlen(x)
        assert wave is not None
        assert wave[0] == pytest.approx(5000.0)
        assert wave[99] == pytest.approx(5000.0 + 0.5 * 99)

        # extract() now evaluates trace wavelength into the Spectrum
        assert spectra[0].wave is not None
        assert spectra[0].wave[0] == pytest.approx(5000.0)
        assert spectra[0].wave[99] == pytest.approx(5000.0 + 0.5 * 99)
