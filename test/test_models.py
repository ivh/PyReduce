"""Tests for Pydantic instrument configuration models."""

from glob import glob
from os.path import basename, dirname, join

import pytest
import yaml
from pydantic import ValidationError

from pyreduce.instruments.models import (
    AmplifierConfig,
    BeamArmConfig,
    DetectorConfig,
    DimensionConfig,
    FiberBundleConfig,
    FiberGroupConfig,
    FibersConfig,
    InstrumentConfig,
    OpticalPathConfig,
    validate_instrument_config,
)

# Get all YAML instrument files
yaml_instruments = glob(
    join(dirname(__file__), "../pyreduce/instruments/*/config.yaml")
)
yaml_instruments = [basename(dirname(f)) for f in yaml_instruments]
yaml_instruments = [f for f in yaml_instruments if f not in ["defaults"]]


@pytest.fixture(params=yaml_instruments)
def yaml_instrument(request):
    return request.param


@pytest.fixture
def yaml_instrument_path(yaml_instrument):
    return join(
        dirname(__file__), f"../pyreduce/instruments/{yaml_instrument}/config.yaml"
    )


@pytest.fixture
def yaml_instrument_data(yaml_instrument_path):
    with open(yaml_instrument_path) as f:
        return yaml.safe_load(f)


class TestInstrumentConfigValidation:
    """Test InstrumentConfig model validation."""

    @pytest.mark.unit
    def test_yaml_instrument_valid(self, yaml_instrument, yaml_instrument_data):
        """Validate all YAML instrument files with Pydantic."""
        config = validate_instrument_config(yaml_instrument_data)
        assert config.instrument is not None

    @pytest.mark.unit
    def test_minimal_config(self):
        """Test minimal valid configuration."""
        data = {
            "instrument": "INSTRUME",
            "id_instrument": "TEST",
        }
        config = InstrumentConfig(**data)
        assert config.instrument == "INSTRUME"
        assert config.id_instrument == "TEST"

    @pytest.mark.unit
    def test_missing_required_field(self):
        """Test that missing required fields raise ValidationError."""
        data = {
            "instrument": "INSTRUME",
            # Missing id_instrument
        }
        with pytest.raises(ValidationError):
            InstrumentConfig(**data)

    @pytest.mark.unit
    def test_arms_as_list(self):
        """Test channels field accepts list."""
        data = {
            "instrument": "INSTRUME",
            "id_instrument": "TEST",
            "channels": ["BLUE", "RED"],
        }
        config = InstrumentConfig(**data)
        assert config.channels == ["BLUE", "RED"]

    @pytest.mark.unit
    def test_arms_as_single_string(self):
        """Test channels field converts single string to list."""
        data = {
            "instrument": "INSTRUME",
            "id_instrument": "TEST",
            "channels": "SINGLE",
        }
        config = InstrumentConfig(**data)
        assert config.channels == ["SINGLE"]

    @pytest.mark.unit
    def test_extension_as_int(self):
        """Test extension can be int."""
        data = {
            "instrument": "INSTRUME",
            "id_instrument": "TEST",
            "extension": 0,
        }
        config = InstrumentConfig(**data)
        assert config.extension == 0

    @pytest.mark.unit
    def test_extension_as_list(self):
        """Test extension can be list."""
        data = {
            "instrument": "INSTRUME",
            "id_instrument": "TEST",
            "extension": [1, 2],
        }
        config = InstrumentConfig(**data)
        assert config.extension == [1, 2]

    @pytest.mark.unit
    def test_extension_as_string(self):
        """Test extension can be string (e.g., for FITS extension names)."""
        data = {
            "instrument": "INSTRUME",
            "id_instrument": "TEST",
            "extension": "SCI",
        }
        config = InstrumentConfig(**data)
        assert config.extension == "SCI"

    @pytest.mark.unit
    def test_gain_as_number(self):
        """Test gain can be numeric."""
        data = {
            "instrument": "INSTRUME",
            "id_instrument": "TEST",
            "gain": 1.5,
        }
        config = InstrumentConfig(**data)
        assert config.gain == 1.5

    @pytest.mark.unit
    def test_gain_as_header_keyword(self):
        """Test gain can be header keyword string."""
        data = {
            "instrument": "INSTRUME",
            "id_instrument": "TEST",
            "gain": "HIERARCH ESO DET OUT1 CONAD",
        }
        config = InstrumentConfig(**data)
        assert config.gain == "HIERARCH ESO DET OUT1 CONAD"

    @pytest.mark.unit
    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed (instrument-specific params)."""
        data = {
            "instrument": "INSTRUME",
            "id_instrument": "TEST",
            "custom_field": "custom_value",
            "another_field": [1, 2, 3],
        }
        config = InstrumentConfig(**data)
        # Extra fields accessible via model_extra
        assert "custom_field" in config.model_extra
        assert config.model_extra["custom_field"] == "custom_value"

    @pytest.mark.unit
    def test_wavelength_range(self):
        """Test wavelength_range field."""
        data = {
            "instrument": "INSTRUME",
            "id_instrument": "TEST",
            "wavelength_range": [[[380, 400], [400, 420]]],
        }
        config = InstrumentConfig(**data)
        assert config.wavelength_range is not None


class TestFutureModels:
    """Test future nested configuration models."""

    @pytest.mark.unit
    def test_amplifier_config(self):
        """Test AmplifierConfig model."""
        data = {
            "id": "amp1",
            "gain": 1.2,
            "readnoise": 3.5,
        }
        config = AmplifierConfig(**data)
        assert config.id == "amp1"
        assert config.gain == 1.2
        assert config.readnoise == 3.5

    @pytest.mark.unit
    def test_amplifier_with_header_ref(self):
        """Test AmplifierConfig with header references."""
        data = {
            "id": "amp1",
            "gain": {"key": "ESO DET OUT1 CONAD"},
            "readnoise": {"key": "ESO DET OUT1 RON"},
            "region": {"x": [0, 2048], "y": [0, 4096]},
        }
        config = AmplifierConfig(**data)
        assert config.gain == {"key": "ESO DET OUT1 CONAD"}

    @pytest.mark.unit
    def test_detector_config(self):
        """Test DetectorConfig model."""
        data = {
            "name": "blue_ccd",
            "naxis": [4096, 4096],
            "orientation": 1,
            "amplifiers": [
                {"id": "amp1", "gain": 1.2, "readnoise": 3.5},
                {"id": "amp2", "gain": 1.18, "readnoise": 3.6},
            ],
        }
        config = DetectorConfig(**data)
        assert config.name == "blue_ccd"
        assert len(config.amplifiers) == 2

    @pytest.mark.unit
    def test_beam_arm_config(self):
        """Test BeamArmConfig model."""
        data = {
            "name": "ordinary",
            "polarization": "O",
            "trace_offset": 48.5,
        }
        config = BeamArmConfig(**data)
        assert config.name == "ordinary"
        assert config.polarization == "O"

    @pytest.mark.unit
    def test_optical_path_config(self):
        """Test OpticalPathConfig model."""
        data = {
            "name": "fiber_a",
            "beam_arms": [
                {"name": "ordinary", "polarization": "O"},
                {"name": "extraordinary", "polarization": "E", "trace_offset": 48.5},
            ],
        }
        config = OpticalPathConfig(**data)
        assert config.name == "fiber_a"
        assert len(config.beam_arms) == 2

    @pytest.mark.unit
    def test_dimension_config(self):
        """Test DimensionConfig model."""
        data = {
            "values": ["BLUE", "RED"],
            "header_key": "ESO INS MODE",
            "optional": False,
        }
        config = DimensionConfig(**data)
        assert config.values == ["BLUE", "RED"]


class TestFiberConfig:
    """Tests for fiber grouping configuration models."""

    @pytest.mark.unit
    def test_fiber_group_config_minimal(self):
        """Test FiberGroupConfig with minimal fields."""
        data = {"range": [1, 36]}
        config = FiberGroupConfig(**data)
        assert config.range == (1, 36)
        assert config.merge == "center"  # default

    @pytest.mark.unit
    def test_fiber_group_config_with_merge(self):
        """Test FiberGroupConfig with merge method."""
        data = {"range": [1, 36], "merge": "average"}
        config = FiberGroupConfig(**data)
        assert config.merge == "average"

    @pytest.mark.unit
    def test_fiber_group_config_merge_indices(self):
        """Test FiberGroupConfig with index-based merge."""
        data = {"range": [1, 36], "merge": [1, 18, 35]}
        config = FiberGroupConfig(**data)
        assert config.merge == [1, 18, 35]

    @pytest.mark.unit
    def test_fiber_group_config_rejects_extra_fields(self):
        """Test FiberGroupConfig rejects unknown fields."""
        data = {"range": [1, 36], "unknown_field": "value"}
        with pytest.raises(ValidationError):
            FiberGroupConfig(**data)

    @pytest.mark.unit
    def test_fiber_bundle_config_minimal(self):
        """Test FiberBundleConfig with minimal fields."""
        data = {"size": 7}
        config = FiberBundleConfig(**data)
        assert config.size == 7
        assert config.count is None
        assert config.merge == "center"  # default

    @pytest.mark.unit
    def test_fiber_bundle_config_with_count(self):
        """Test FiberBundleConfig with count validation."""
        data = {"size": 7, "count": 90, "merge": "average"}
        config = FiberBundleConfig(**data)
        assert config.size == 7
        assert config.count == 90
        assert config.merge == "average"

    @pytest.mark.unit
    def test_fiber_bundle_config_rejects_extra_fields(self):
        """Test FiberBundleConfig rejects unknown fields."""
        data = {"size": 7, "bad_field": True}
        with pytest.raises(ValidationError):
            FiberBundleConfig(**data)

    @pytest.mark.unit
    def test_fibers_config_with_groups(self):
        """Test FibersConfig with named groups."""
        data = {
            "groups": {
                "A": {"range": [1, 36], "merge": "average"},
                "cal": {"range": [37, 40], "merge": "average"},
                "B": {"range": [40, 76], "merge": "average"},
            },
            "use": {"science": ["A", "B"], "wavecal": ["cal"]},
        }
        config = FibersConfig(**data)
        assert "A" in config.groups
        assert config.groups["A"].range == (1, 36)
        assert config.use["science"] == ["A", "B"]

    @pytest.mark.unit
    def test_fibers_config_with_bundles(self):
        """Test FibersConfig with bundle pattern."""
        data = {
            "bundles": {"size": 7, "merge": "center"},
            "use": {"science": "groups", "curvature": "groups"},
        }
        config = FibersConfig(**data)
        assert config.bundles.size == 7
        assert config.use["science"] == "groups"

    @pytest.mark.unit
    def test_fibers_config_use_all(self):
        """Test FibersConfig with 'all' trace selection."""
        data = {
            "groups": {"A": {"range": [1, 36]}},
            "use": {"norm_flat": "all"},
        }
        config = FibersConfig(**data)
        assert config.use["norm_flat"] == "all"

    @pytest.mark.unit
    def test_fibers_config_rejects_extra_fields(self):
        """Test FibersConfig rejects unknown fields."""
        data = {
            "groups": {"A": {"range": [1, 36]}},
            "unknown": "bad",
        }
        with pytest.raises(ValidationError):
            FibersConfig(**data)

    @pytest.mark.unit
    def test_instrument_config_with_fibers(self):
        """Test InstrumentConfig includes fibers field."""
        data = {
            "instrument": "TEST",
            "id_instrument": "TEST",
            "fibers": {
                "groups": {"A": {"range": [1, 36], "merge": "average"}},
                "use": {"science": ["A"]},
            },
        }
        config = InstrumentConfig(**data)
        assert config.fibers is not None
        assert "A" in config.fibers.groups

    @pytest.mark.unit
    def test_instrument_config_fibers_none_by_default(self):
        """Test InstrumentConfig has fibers=None by default."""
        data = {"instrument": "TEST", "id_instrument": "TEST"}
        config = InstrumentConfig(**data)
        assert config.fibers is None
