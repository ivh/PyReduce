"""Tests for pyreduce.spectra module."""

import numpy as np
import pytest
from astropy.io import fits

from pyreduce.spectra import ExtractionParams, Spectra, Spectrum

pytestmark = pytest.mark.unit


class TestSpectrumDataclass:
    """Tests for Spectrum dataclass."""

    def test_mask_property_identifies_nan(self):
        """mask property correctly identifies NaN pixels."""
        spec = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
        sig = np.array([0.1, 0.2, np.nan, 0.4, np.nan])
        spectrum = Spectrum(m=1, group="A", spec=spec, sig=sig)

        expected_mask = np.array([False, False, True, False, True])
        np.testing.assert_array_equal(spectrum.mask, expected_mask)

    def test_mask_property_no_nans(self):
        """mask property returns all False when no NaN."""
        spec = np.array([1.0, 2.0, 3.0, 4.0])
        sig = np.array([0.1, 0.2, 0.3, 0.4])
        spectrum = Spectrum(m=1, group="A", spec=spec, sig=sig)

        assert not spectrum.mask.any()

    def test_normalized_with_continuum(self):
        """normalized() returns correct values when continuum is present."""
        spec = np.array([100.0, 200.0, 150.0])
        sig = np.array([10.0, 20.0, 15.0])
        cont = np.array([100.0, 100.0, 100.0])
        spectrum = Spectrum(m=1, group="A", spec=spec, sig=sig, cont=cont)

        spec_norm, sig_norm = spectrum.normalized()

        np.testing.assert_array_almost_equal(spec_norm, [1.0, 2.0, 1.5])
        np.testing.assert_array_almost_equal(sig_norm, [0.1, 0.2, 0.15])

    def test_normalized_without_continuum_raises(self):
        """normalized() raises ValueError when no continuum."""
        spectrum = Spectrum(
            m=1, group="A", spec=np.array([1.0, 2.0]), sig=np.array([0.1, 0.2])
        )

        with pytest.raises(ValueError, match="No continuum"):
            spectrum.normalized()

    def test_from_trace_factory(self):
        """from_trace factory copies identity from Trace."""
        from pyreduce.trace_model import Trace

        trace = Trace(
            m=42, group="cal", pos=np.array([1.0, 100.0]), column_range=(0, 1000)
        )
        spec = np.array([1.0, 2.0, 3.0])
        sig = np.array([0.1, 0.2, 0.3])

        spectrum = Spectrum.from_trace(trace, spec, sig, extraction_height=25.0)

        assert spectrum.m == 42
        assert spectrum.group == "cal"
        np.testing.assert_array_equal(spectrum.spec, spec)
        np.testing.assert_array_equal(spectrum.sig, sig)
        assert spectrum.extraction_height == 25.0


class TestExtractionParams:
    """Tests for ExtractionParams dataclass."""

    def test_to_header_writes_keywords(self):
        """to_header writes all keywords to header."""
        params = ExtractionParams(
            osample=10, lambda_sf=0.1, lambda_sp=0.0, swath_width=300
        )
        header = fits.Header()

        params.to_header(header)

        assert header["E_OSAMPLE"] == 10
        assert header["E_LAMBDASF"] == 0.1
        assert header["E_LAMBDASP"] == 0.0
        assert header["E_SWATHW"] == 300

    def test_to_header_skips_none_swath_width(self):
        """to_header skips E_SWATHW when swath_width is None."""
        params = ExtractionParams(osample=10, lambda_sf=0.1, lambda_sp=0.0)
        header = fits.Header()

        params.to_header(header)

        assert "E_SWATHW" not in header

    def test_from_header_reads_keywords(self):
        """from_header reads all keywords from header."""
        header = fits.Header()
        header["E_OSAMPLE"] = 10
        header["E_LAMBDASF"] = 0.1
        header["E_LAMBDASP"] = 0.0
        header["E_SWATHW"] = 300

        params = ExtractionParams.from_header(header)

        assert params.osample == 10
        assert params.lambda_sf == 0.1
        assert params.lambda_sp == 0.0
        assert params.swath_width == 300

    def test_from_header_returns_none_if_missing(self):
        """from_header returns None if E_OSAMPLE is missing."""
        header = fits.Header()

        params = ExtractionParams.from_header(header)

        assert params is None

    def test_roundtrip(self):
        """ExtractionParams survives header roundtrip."""
        original = ExtractionParams(
            osample=8, lambda_sf=0.5, lambda_sp=0.1, swath_width=200
        )
        header = fits.Header()
        original.to_header(header)

        loaded = ExtractionParams.from_header(header)

        assert loaded.osample == original.osample
        assert loaded.lambda_sf == original.lambda_sf
        assert loaded.lambda_sp == original.lambda_sp
        assert loaded.swath_width == original.swath_width


class TestSpectraSaveLoadRoundtrip:
    """Tests for Spectra.save/read roundtrip."""

    @pytest.fixture
    def basic_spectra(self):
        """Create basic Spectra with spec and sig only."""
        header = fits.Header()
        header["OBJECT"] = "HD123456"
        data = [
            Spectrum(m=1, group="A", spec=np.arange(100.0), sig=np.ones(100) * 0.1),
            Spectrum(m=2, group="A", spec=np.arange(100.0) * 2, sig=np.ones(100) * 0.2),
        ]
        return Spectra(header=header, data=data)

    @pytest.fixture
    def full_spectra(self):
        """Create Spectra with all optional fields."""
        header = fits.Header()
        header["OBJECT"] = "HD654321"
        params = ExtractionParams(osample=10, lambda_sf=0.1, lambda_sp=0.0)

        ncol = 50
        data = [
            Spectrum(
                m=10,
                group="A",
                spec=np.arange(ncol, dtype=float),
                sig=np.ones(ncol) * 0.1,
                wave=np.linspace(5000, 5100, ncol),
                cont=np.ones(ncol) * 50,
                slitfu=np.array([0.1, 0.5, 1.0, 0.5, 0.1]),
                extraction_height=20.0,
            ),
            Spectrum(
                m=11,
                group="B",
                spec=np.arange(ncol, dtype=float) * 2,
                sig=np.ones(ncol) * 0.2,
                wave=np.linspace(5100, 5200, ncol),
                cont=np.ones(ncol) * 100,
                slitfu=np.array([0.2, 0.6, 1.0, 0.6, 0.2]),
                extraction_height=22.0,
            ),
        ]
        return Spectra(header=header, data=data, params=params)

    def test_roundtrip_basic_spectra(self, tmp_path, basic_spectra):
        """Basic Spectra survives save/load roundtrip."""
        path = tmp_path / "spectra.fits"
        basic_spectra.save(path)
        loaded = Spectra.read(path)

        assert loaded.ntrace == 2
        assert loaded.ncol == 100
        assert loaded.header["OBJECT"] == "HD123456"

        for orig, load in zip(basic_spectra.data, loaded.data, strict=False):
            assert load.m == orig.m
            assert load.group == orig.group
            np.testing.assert_array_almost_equal(load.spec, orig.spec)
            np.testing.assert_array_almost_equal(load.sig, orig.sig)

    def test_roundtrip_full_spectra(self, tmp_path, full_spectra):
        """Full Spectra with all fields survives roundtrip."""
        path = tmp_path / "spectra.fits"
        full_spectra.save(path)
        loaded = Spectra.read(path)

        assert loaded.ntrace == 2
        assert loaded.params is not None
        assert loaded.params.osample == 10

        for orig, load in zip(full_spectra.data, loaded.data, strict=False):
            assert load.m == orig.m
            assert load.group == orig.group
            np.testing.assert_array_almost_equal(load.spec, orig.spec)
            np.testing.assert_array_almost_equal(load.sig, orig.sig)
            np.testing.assert_array_almost_equal(load.wave, orig.wave)
            np.testing.assert_array_almost_equal(load.cont, orig.cont)
            np.testing.assert_array_almost_equal(load.slitfu, orig.slitfu)
            assert load.extraction_height == pytest.approx(orig.extraction_height)

    def test_roundtrip_nan_masking(self, tmp_path):
        """NaN masking is preserved through roundtrip."""
        spec = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
        sig = np.array([0.1, 0.2, np.nan, 0.4, np.nan, 0.6])
        data = [Spectrum(m=1, group="A", spec=spec, sig=sig)]
        spectra = Spectra(header=fits.Header(), data=data)

        path = tmp_path / "spectra.fits"
        spectra.save(path)
        loaded = Spectra.read(path)

        np.testing.assert_array_equal(np.isnan(loaded.data[0].spec), np.isnan(spec))
        np.testing.assert_array_equal(np.isnan(loaded.data[0].sig), np.isnan(sig))

    def test_roundtrip_integer_fiber(self, tmp_path):
        """Integer fiber identifier survives roundtrip."""
        data = [
            Spectrum(m=1, group=42, spec=np.ones(10), sig=np.ones(10) * 0.1),
        ]
        spectra = Spectra(header=fits.Header(), data=data)

        path = tmp_path / "spectra.fits"
        spectra.save(path)
        loaded = Spectra.read(path)

        assert loaded.data[0].group == 42
        assert isinstance(loaded.data[0].group, int)

    def test_roundtrip_none_m(self, tmp_path):
        """None spectral order survives roundtrip."""
        data = [
            Spectrum(m=None, group="A", spec=np.ones(10), sig=np.ones(10) * 0.1),
        ]
        spectra = Spectra(header=fits.Header(), data=data)

        path = tmp_path / "spectra.fits"
        spectra.save(path)
        loaded = Spectra.read(path)

        assert loaded.data[0].m is None

    def test_format_version_written(self, tmp_path, basic_spectra):
        """E_FMTVER header keyword is written."""
        path = tmp_path / "spectra.fits"
        basic_spectra.save(path)
        loaded = Spectra.read(path)

        assert loaded.header["E_FMTVER"] == 2

    def test_steps_written(self, tmp_path, basic_spectra):
        """E_STEPS header keyword is written."""
        path = tmp_path / "spectra.fits"
        basic_spectra.save(path, steps=["science", "continuum"])
        loaded = Spectra.read(path)

        assert loaded.header["E_STEPS"] == "science,continuum"


class TestSpectraSelect:
    """Tests for Spectra.select method."""

    @pytest.fixture
    def multi_order_spectra(self):
        """Create Spectra with multiple orders and fibers."""
        data = [
            Spectrum(m=1, group="A", spec=np.ones(10), sig=np.ones(10) * 0.1),
            Spectrum(m=1, group="B", spec=np.ones(10) * 2, sig=np.ones(10) * 0.2),
            Spectrum(m=2, group="A", spec=np.ones(10) * 3, sig=np.ones(10) * 0.3),
            Spectrum(m=2, group="B", spec=np.ones(10) * 4, sig=np.ones(10) * 0.4),
        ]
        return Spectra(header=fits.Header(), data=data)

    def test_select_by_order(self, multi_order_spectra):
        """select filters by spectral order."""
        selected = multi_order_spectra.select(m=1)

        assert len(selected) == 2
        assert all(s.m == 1 for s in selected)

    def test_select_by_fiber(self, multi_order_spectra):
        """select filters by fiber."""
        selected = multi_order_spectra.select(group="A")

        assert len(selected) == 2
        assert all(s.group == "A" for s in selected)

    def test_select_by_both(self, multi_order_spectra):
        """select filters by both order and fiber."""
        selected = multi_order_spectra.select(m=2, group="B")

        assert len(selected) == 1
        assert selected[0].m == 2
        assert selected[0].group == "B"

    def test_select_no_match(self, multi_order_spectra):
        """select returns empty list when no match."""
        selected = multi_order_spectra.select(m=999)

        assert len(selected) == 0


class TestSpectraGetArrays:
    """Tests for Spectra.get_arrays method."""

    def test_get_arrays_shapes(self):
        """get_arrays returns correct shapes."""
        ncol = 50
        data = [
            Spectrum(
                m=1,
                group="A",
                spec=np.ones(ncol),
                sig=np.ones(ncol) * 0.1,
                wave=np.linspace(5000, 5100, ncol),
            ),
            Spectrum(
                m=2,
                group="B",
                spec=np.ones(ncol) * 2,
                sig=np.ones(ncol) * 0.2,
                wave=np.linspace(5100, 5200, ncol),
            ),
        ]
        spectra = Spectra(header=fits.Header(), data=data)

        arrays = spectra.get_arrays()

        assert arrays["spec"].shape == (2, ncol)
        assert arrays["sig"].shape == (2, ncol)
        assert arrays["wave"].shape == (2, ncol)
        assert arrays["m"].shape == (2,)
        assert arrays["group"].shape == (2,)

    def test_get_arrays_values(self):
        """get_arrays returns correct values."""
        data = [
            Spectrum(
                m=1, group="A", spec=np.array([1.0, 2.0]), sig=np.array([0.1, 0.2])
            ),
            Spectrum(
                m=2, group="B", spec=np.array([3.0, 4.0]), sig=np.array([0.3, 0.4])
            ),
        ]
        spectra = Spectra(header=fits.Header(), data=data)

        arrays = spectra.get_arrays()

        np.testing.assert_array_equal(arrays["spec"], [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(arrays["m"], [1, 2])
        np.testing.assert_array_equal(arrays["group"], ["A", "B"])

    def test_get_arrays_none_for_missing_optional(self):
        """get_arrays returns None for missing optional fields."""
        data = [
            Spectrum(m=1, group="A", spec=np.ones(10), sig=np.ones(10) * 0.1),
        ]
        spectra = Spectra(header=fits.Header(), data=data)

        arrays = spectra.get_arrays()

        assert arrays["wave"] is None
        assert arrays["cont"] is None


class TestLegacyFormatReading:
    """Tests for reading legacy Echelle format."""

    def test_read_legacy_format(self, tmp_path):
        """Can read legacy format with COLUMNS masking."""
        # Create a legacy format file matching actual Echelle output structure
        ncol, ntrace = 100, 3
        spec = np.ones((ntrace, ncol), dtype=np.float32) * 100
        sig = np.ones((ntrace, ncol), dtype=np.float32) * 10
        columns = np.array([[10, 90], [15, 85], [20, 80]], dtype=np.int16)

        header = fits.Header()
        header["E_FMTVER"] = 1  # Legacy version

        # Legacy format stores data in a single row with flattened arrays
        col_spec = fits.Column(
            name="SPEC",
            format=f"{ntrace * ncol}E",
            dim=f"({ncol},{ntrace})",
            array=[spec],
        )
        col_sig = fits.Column(
            name="SIG",
            format=f"{ntrace * ncol}E",
            dim=f"({ncol},{ntrace})",
            array=[sig],
        )
        col_columns = fits.Column(
            name="COLUMNS",
            format=f"{ntrace * 2}I",
            dim=f"(2,{ntrace})",
            array=[columns],
        )

        table = fits.BinTableHDU.from_columns([col_spec, col_sig, col_columns])

        primary = fits.PrimaryHDU(header=header)
        hdulist = fits.HDUList([primary, table])

        path = tmp_path / "legacy.fits"
        hdulist.writeto(path, overwrite=True)

        # Read and verify
        loaded = Spectra.read(path)

        assert loaded.ntrace == 3

        # Check masking was applied (outside columns should be NaN)
        assert np.isnan(loaded.data[0].spec[0])  # Before column_range start
        assert not np.isnan(loaded.data[0].spec[50])  # Inside column_range
        assert np.isnan(loaded.data[0].spec[95])  # After column_range end


class TestSpectraMixedOptionalFields:
    """Tests for spectra where some have optional fields and others don't."""

    def test_mixed_slitfu_lengths(self, tmp_path):
        """Spectra with different slit function lengths roundtrip correctly."""
        data = [
            Spectrum(
                m=1,
                group="A",
                spec=np.ones(10),
                sig=np.ones(10) * 0.1,
                slitfu=np.array([0.1, 0.5, 1.0, 0.5, 0.1]),
            ),
            Spectrum(
                m=2,
                group="B",
                spec=np.ones(10) * 2,
                sig=np.ones(10) * 0.2,
                slitfu=np.array([0.1, 0.3, 0.5, 0.7, 1.0, 0.7, 0.5, 0.3, 0.1]),
            ),
        ]
        spectra = Spectra(header=fits.Header(), data=data)

        path = tmp_path / "spectra.fits"
        spectra.save(path)
        loaded = Spectra.read(path)

        assert len(loaded.data[0].slitfu) == 5
        assert len(loaded.data[1].slitfu) == 9
        np.testing.assert_array_almost_equal(loaded.data[0].slitfu, data[0].slitfu)
        np.testing.assert_array_almost_equal(loaded.data[1].slitfu, data[1].slitfu)

    def test_some_with_wave_some_without(self, tmp_path):
        """Spectra where some have wavelength and others don't."""
        data = [
            Spectrum(
                m=1,
                group="A",
                spec=np.ones(10),
                sig=np.ones(10) * 0.1,
                wave=np.linspace(5000, 5100, 10),
            ),
            Spectrum(
                m=2,
                group="B",
                spec=np.ones(10) * 2,
                sig=np.ones(10) * 0.2,
                # No wave
            ),
        ]
        spectra = Spectra(header=fits.Header(), data=data)

        path = tmp_path / "spectra.fits"
        spectra.save(path)
        loaded = Spectra.read(path)

        np.testing.assert_array_almost_equal(loaded.data[0].wave, data[0].wave)
        # Second spectrum should have NaN wave (from padding)
        assert np.all(np.isnan(loaded.data[1].wave))
