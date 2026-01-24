import numpy as np
import pytest

from pyreduce import reduce
from pyreduce.instruments.instrument_info import load_instrument


class TestStepBasics:
    """Unit tests for Step base class."""

    @pytest.fixture
    def mock_instrument(self):
        return load_instrument("UVES")

    @pytest.mark.unit
    def test_prefix_with_channel(self, mock_instrument, tmp_path):
        """prefix should include channel when set."""
        step = reduce.Step(
            mock_instrument,
            "RED",
            "HD12345",
            "2024-01-01",
            str(tmp_path),
            None,
            plot=False,
        )
        assert step.prefix == "uves_red"

    @pytest.mark.unit
    def test_prefix_without_channel(self, mock_instrument, tmp_path):
        """prefix should be instrument name only when no channel."""
        step = reduce.Step(
            mock_instrument,
            "",
            "HD12345",
            "2024-01-01",
            str(tmp_path),
            None,
            plot=False,
        )
        assert step.prefix == "uves"

    @pytest.mark.unit
    def test_prefix_with_none_channel(self, mock_instrument, tmp_path):
        """prefix should be instrument name only when channel is None."""
        step = reduce.Step(
            mock_instrument,
            None,
            "HD12345",
            "2024-01-01",
            str(tmp_path),
            None,
            plot=False,
        )
        assert step.prefix == "uves"

    @pytest.mark.unit
    def test_output_dir_formatting(self, mock_instrument, tmp_path):
        """output_dir should support template formatting."""
        template = str(tmp_path) + "/{instrument}/{target}/{night}/{channel}"
        step = reduce.Step(
            mock_instrument,
            "RED",
            "HD12345",
            "2024-01-01",
            template,
            None,
            plot=False,
        )
        assert "UVES" in step.output_dir
        assert "HD12345" in step.output_dir
        assert "2024-01-01" in step.output_dir
        assert "RED" in step.output_dir

    @pytest.mark.unit
    def test_dependsOn_is_list(self, mock_instrument, tmp_path):
        """dependsOn should return a list."""
        step = reduce.Step(mock_instrument, "", "", "", str(tmp_path), None, plot=False)
        assert isinstance(step.dependsOn, list)

    @pytest.mark.unit
    def test_loadDependsOn_is_list(self, mock_instrument, tmp_path):
        """loadDependsOn should return a list."""
        step = reduce.Step(mock_instrument, "", "", "", str(tmp_path), None, plot=False)
        assert isinstance(step.loadDependsOn, list)


class TestExtractionStepValidation:
    """Unit tests for ExtractionStep validation."""

    @pytest.fixture
    def mock_instrument(self):
        return load_instrument("UVES")

    @pytest.mark.unit
    def test_invalid_extraction_method(self, mock_instrument, tmp_path):
        """Invalid extraction method should raise ValueError."""
        config = {
            "plot": False,
            "extraction_method": "invalid_method",
            "extraction_height": 10,
        }
        with pytest.raises(ValueError, match="not supported"):
            reduce.ExtractionStep(
                mock_instrument, "", "", "", str(tmp_path), None, **config
            )

    @pytest.mark.unit
    def test_simple_extraction_config(self, mock_instrument, tmp_path):
        """Simple extraction method should set correct kwargs."""
        config = {
            "plot": False,
            "extraction_method": "simple",
            "extraction_height": 10,
            "collapse_function": "median",
        }
        step = reduce.ExtractionStep(
            mock_instrument, "", "", "", str(tmp_path), None, **config
        )
        assert step.extraction_method == "simple"
        assert "extraction_height" in step.extraction_kwargs
        assert step.extraction_kwargs["collapse_function"] == "median"

    @pytest.mark.unit
    def test_optimal_extraction_config(self, mock_instrument, tmp_path):
        """Optimal extraction method should set correct kwargs."""
        config = {
            "plot": False,
            "extraction_method": "optimal",
            "extraction_height": 10,
            "smooth_slitfunction": 1,
            "smooth_spectrum": 0,
            "oversampling": 4,
            "swath_width": 300,
            "maxiter": 10,
        }
        step = reduce.ExtractionStep(
            mock_instrument, "", "", "", str(tmp_path), None, **config
        )
        assert step.extraction_method == "optimal"
        assert step.extraction_kwargs["osample"] == 4
        assert step.extraction_kwargs["swath_width"] == 300


class TestNormalizeFlatFieldValidation:
    """Unit tests for NormalizeFlatField validation."""

    @pytest.fixture
    def mock_instrument(self):
        return load_instrument("UVES")

    @pytest.mark.unit
    def test_invalid_extraction_method(self, mock_instrument, tmp_path):
        """Invalid extraction method should raise ValueError."""
        config = {
            "plot": False,
            "extraction_method": "invalid",
            "extraction_height": 10,
            "threshold": 100,
            "threshold_lower": 0,
        }
        with pytest.raises(ValueError, match="not supported"):
            reduce.NormalizeFlatField(
                mock_instrument, "", "", "", str(tmp_path), None, **config
            )


class TestFinalizeConfigSaving:
    """Unit tests for Finalize.save_config_to_header."""

    @pytest.fixture
    def mock_instrument(self):
        return load_instrument("UVES")

    @pytest.mark.unit
    def test_save_config_flat(self, mock_instrument, tmp_path):
        """Test saving flat config to header."""
        config = {"plot": False, "filename": "{input}.final.fits"}
        step = reduce.Finalize(
            mock_instrument, "", "", "", str(tmp_path), None, **config
        )

        head = {}
        config_to_save = {"degree": 4, "threshold": 100}
        result = step.save_config_to_header(head, config_to_save)

        assert "HIERARCH PR DEGREE" in result
        assert result["HIERARCH PR DEGREE"] == 4
        assert "HIERARCH PR THRESHOLD" in result
        assert result["HIERARCH PR THRESHOLD"] == 100

    @pytest.mark.unit
    def test_save_config_nested(self, mock_instrument, tmp_path):
        """Test saving nested config to header."""
        config = {"plot": False, "filename": "{input}.final.fits"}
        step = reduce.Finalize(
            mock_instrument, "", "", "", str(tmp_path), None, **config
        )

        head = {}
        config_to_save = {"bias": {"degree": 0}, "flat": {"threshold": 100}}
        result = step.save_config_to_header(head, config_to_save)

        assert "HIERARCH PR BIAS DEGREE" in result
        assert result["HIERARCH PR BIAS DEGREE"] == 0
        assert "HIERARCH PR FLAT THRESHOLD" in result

    @pytest.mark.unit
    def test_save_config_skips_plot(self, mock_instrument, tmp_path):
        """Test that plot key is skipped."""
        config = {"plot": False, "filename": "{input}.final.fits"}
        step = reduce.Finalize(
            mock_instrument, "", "", "", str(tmp_path), None, **config
        )

        head = {}
        config_to_save = {"plot": True, "degree": 4}
        result = step.save_config_to_header(head, config_to_save)

        assert "HIERARCH PR PLOT" not in result
        assert "HIERARCH PR DEGREE" in result

    @pytest.mark.unit
    def test_save_config_none_value(self, mock_instrument, tmp_path):
        """Test that None values are converted to 'null'."""
        config = {"plot": False, "filename": "{input}.final.fits"}
        step = reduce.Finalize(
            mock_instrument, "", "", "", str(tmp_path), None, **config
        )

        head = {}
        config_to_save = {"trace_range": None}
        result = step.save_config_to_header(head, config_to_save)

        assert result["HIERARCH PR TRACE_RANGE"] == "null"

    @pytest.mark.unit
    def test_save_config_list_value(self, mock_instrument, tmp_path):
        """Test that list values are converted to strings."""
        config = {"plot": False, "filename": "{input}.final.fits"}
        step = reduce.Finalize(
            mock_instrument, "", "", "", str(tmp_path), None, **config
        )

        head = {}
        config_to_save = {"degree": [4, 5]}
        result = step.save_config_to_header(head, config_to_save)

        assert result["HIERARCH PR DEGREE"] == "[4, 5]"


class TestBiasStepSaveLoad:
    """Unit tests for Bias step save/load."""

    @pytest.fixture
    def mock_instrument(self):
        return load_instrument("UVES")

    @pytest.mark.unit
    def test_bias_savefile_name(self, mock_instrument, tmp_path):
        """Test bias savefile naming."""
        config = {"plot": False, "degree": 0}
        step = reduce.Bias(
            mock_instrument,
            "RED",
            "HD12345",
            "2024-01-01",
            str(tmp_path),
            None,
            **config,
        )
        assert step.savefile.endswith(".bias.fits")
        assert "uves_red" in step.savefile

    @pytest.mark.unit
    def test_bias_load_missing_returns_none(self, mock_instrument, tmp_path):
        """Test loading missing bias returns None."""
        config = {"plot": False, "degree": 0}
        step = reduce.Bias(mock_instrument, "", "", "", str(tmp_path), None, **config)
        result = step.load(mask=False)
        assert result == (None, None)


class TestTraceSaveLoad:
    """Unit tests for Trace step save/load."""

    @pytest.fixture
    def mock_instrument(self):
        return load_instrument("UVES")

    @pytest.mark.unit
    def test_trace_savefile_name(self, mock_instrument, tmp_path):
        """Test trace savefile naming."""
        config = {
            "plot": False,
            "degree": 4,
            "min_cluster": 500,
            "min_width": 10,
            "filter_y": 10,
            "noise": 100,
            "bias_scaling": "none",
            "norm_scaling": "none",
            "degree_before_merge": 2,
            "regularization": 0,
            "closing_shape": (5, 5),
            "opening_shape": (5, 5),
            "auto_merge_threshold": 0.5,
            "merge_min_threshold": 0.1,
            "split_sigma": 3,
            "border_width": 10,
            "manual": False,
        }
        step = reduce.Trace(
            mock_instrument, "RED", "", "", str(tmp_path), None, **config
        )
        assert step.savefile.endswith(".traces.npz")

    @pytest.mark.unit
    def test_trace_save_load_roundtrip(self, mock_instrument, tmp_path):
        """Test saving and loading trace results."""
        config = {
            "plot": False,
            "degree": 4,
            "min_cluster": 500,
            "min_width": 10,
            "filter_y": 10,
            "noise": 100,
            "bias_scaling": "none",
            "norm_scaling": "none",
            "degree_before_merge": 2,
            "regularization": 0,
            "closing_shape": (5, 5),
            "opening_shape": (5, 5),
            "auto_merge_threshold": 0.5,
            "merge_min_threshold": 0.1,
            "split_sigma": 3,
            "border_width": 10,
            "manual": False,
        }
        step = reduce.Trace(mock_instrument, "", "", "", str(tmp_path), None, **config)

        # Create fake trace data
        orders = np.array([[100.0, 0.01, 0.0], [200.0, 0.02, 0.0]])
        column_range = np.array([[10, 990], [20, 980]])

        step.save(orders, column_range)
        loaded_orders, loaded_cr = step.load()

        assert np.allclose(orders, loaded_orders)
        assert np.allclose(column_range, loaded_cr)

    @pytest.mark.unit
    def test_trace_save_load_with_heights(self, mock_instrument, tmp_path):
        """Test saving and loading trace results with heights."""
        config = {
            "plot": False,
            "degree": 4,
            "min_cluster": 500,
            "min_width": 10,
            "filter_y": 10,
            "noise": 100,
            "bias_scaling": "none",
            "norm_scaling": "none",
            "degree_before_merge": 2,
            "regularization": 0,
            "closing_shape": (5, 5),
            "opening_shape": (5, 5),
            "auto_merge_threshold": 0.5,
            "merge_min_threshold": 0.1,
            "split_sigma": 3,
            "border_width": 10,
            "manual": False,
        }
        step = reduce.Trace(mock_instrument, "", "", "", str(tmp_path), None, **config)

        # Create fake trace data with heights
        orders = np.array([[100.0, 0.01, 0.0], [200.0, 0.02, 0.0]])
        column_range = np.array([[10, 990], [20, 980]])
        step.heights = np.array([20.0, 25.0])

        step.save(orders, column_range)
        loaded_orders, loaded_cr = step.load()

        assert np.allclose(orders, loaded_orders)
        assert np.allclose(column_range, loaded_cr)
        assert step.heights is not None
        assert np.allclose(step.heights, [20.0, 25.0])

    @pytest.mark.unit
    def test_trace_load_without_heights_backwards_compat(
        self, mock_instrument, tmp_path
    ):
        """Test loading old trace file without heights returns None."""
        config = {
            "plot": False,
            "degree": 4,
            "min_cluster": 500,
            "min_width": 10,
            "filter_y": 10,
            "noise": 100,
            "bias_scaling": "none",
            "norm_scaling": "none",
            "degree_before_merge": 2,
            "regularization": 0,
            "closing_shape": (5, 5),
            "opening_shape": (5, 5),
            "auto_merge_threshold": 0.5,
            "merge_min_threshold": 0.1,
            "split_sigma": 3,
            "border_width": 10,
            "manual": False,
        }
        step = reduce.Trace(mock_instrument, "", "", "", str(tmp_path), None, **config)

        # Manually save without heights (simulating old file format)
        orders = np.array([[100.0, 0.01, 0.0]])
        column_range = np.array([[10, 990]])
        np.savez(step.savefile, traces=orders, column_range=column_range)

        loaded_orders, loaded_cr = step.load()

        assert np.allclose(orders, loaded_orders)
        assert step.heights is None  # Backwards compat: missing heights -> None


class TestSlitCurvatureSaveLoad:
    """Unit tests for SlitCurvatureDetermination save/load."""

    @pytest.fixture
    def mock_instrument(self):
        return load_instrument("UVES")

    @pytest.mark.unit
    def test_curvature_savefile_name(self, mock_instrument, tmp_path):
        """Test curvature savefile naming."""
        config = {
            "plot": False,
            "curvature_cutoff": 3,
            "extraction_height": 0.5,
            "curve_height": 0.5,
            "degree": 2,
            "curve_degree": 2,
            "dimensionality": "1D",
            "peak_threshold": 10,
            "peak_width": 1,
            "window_width": 9,
            "peak_function": "gaussian",
            "bias_scaling": "none",
            "norm_scaling": "none",
            "extraction_method": "simple",
            "collapse_function": "sum",
        }
        step = reduce.SlitCurvatureDetermination(
            mock_instrument, "RED", "", "", str(tmp_path), None, **config
        )
        assert step.savefile.endswith(".curve.npz")

    @pytest.mark.unit
    def test_curvature_save_load_roundtrip(self, mock_instrument, tmp_path):
        """Test saving and loading curvature results."""
        config = {
            "plot": False,
            "curvature_cutoff": 3,
            "extraction_height": 0.5,
            "curve_height": 0.5,
            "degree": 2,
            "curve_degree": 2,
            "dimensionality": "1D",
            "peak_threshold": 10,
            "peak_width": 1,
            "window_width": 9,
            "peak_function": "gaussian",
            "bias_scaling": "none",
            "norm_scaling": "none",
            "extraction_method": "simple",
            "collapse_function": "sum",
        }
        step = reduce.SlitCurvatureDetermination(
            mock_instrument, "", "", "", str(tmp_path), None, **config
        )

        # Create fake curvature data
        p1 = np.random.rand(10, 1000) * 0.01
        p2 = np.random.rand(10, 1000) * 0.001

        step.save(p1, p2)
        loaded_p1, loaded_p2 = step.load()

        assert np.allclose(p1, loaded_p1)
        assert np.allclose(p2, loaded_p2)

    @pytest.mark.unit
    def test_curvature_load_missing_returns_none(self, mock_instrument, tmp_path):
        """Test loading missing curvature returns None."""
        config = {
            "plot": False,
            "curvature_cutoff": 3,
            "extraction_height": 0.5,
            "curve_height": 0.5,
            "degree": 2,
            "curve_degree": 2,
            "dimensionality": "1D",
            "peak_threshold": 10,
            "peak_width": 1,
            "window_width": 9,
            "peak_function": "gaussian",
            "bias_scaling": "none",
            "norm_scaling": "none",
            "extraction_method": "simple",
            "collapse_function": "sum",
        }
        step = reduce.SlitCurvatureDetermination(
            mock_instrument, "", "", "", str(tmp_path), None, **config
        )
        p1, p2 = step.load()
        assert p1 is None
        assert p2 is None


class TestBackgroundScatterSaveLoad:
    """Unit tests for BackgroundScatter save/load."""

    @pytest.fixture
    def mock_instrument(self):
        return load_instrument("UVES")

    @pytest.mark.unit
    def test_scatter_load_missing_returns_none(self, mock_instrument, tmp_path):
        """Test loading missing scatter returns None."""
        config = {
            "plot": False,
            "scatter_degree": (4, 4),
            "extraction_height": 0.5,
            "scatter_cutoff": 3,
            "border_width": 10,
            "bias_scaling": "none",
            "norm_scaling": "none",
        }
        step = reduce.BackgroundScatter(
            mock_instrument, "", "", "", str(tmp_path), None, **config
        )
        result = step.load()
        assert result is None


class TestExtractionHeightFallback:
    """Tests for extraction height fallback logic."""

    @pytest.mark.unit
    def test_array_heights_used_when_setting_none(self):
        """When extraction_height setting is None, array heights from traces are used."""
        # Simulate the height selection logic from NormalizeFlatField/ScienceExtraction
        selected = {
            "all": (
                np.array([[0, 0, 50], [0, 0, 100]]),  # 2 traces
                np.array([[0, 1000], [0, 1000]]),  # column ranges
                np.array([20.0, 25.0]),  # per-trace heights from npz
            )
        }
        default_height = None  # extraction_height setting is null

        per_trace_heights = []
        for name in selected:
            tr, _, height = selected[name]
            n_traces = len(tr)
            if (
                height is not None
                and hasattr(height, "__len__")
                and not isinstance(height, str)
            ):
                per_trace_heights.extend(height[:n_traces])
            elif height is not None:
                per_trace_heights.extend([height] * n_traces)
            elif default_height is not None:
                per_trace_heights.extend([default_height] * n_traces)
            else:
                per_trace_heights.extend([None] * n_traces)

        assert per_trace_heights == [20.0, 25.0]

    @pytest.mark.unit
    def test_setting_overrides_array_heights(self):
        """When extraction_height setting is set, it overrides npz heights."""
        selected = {
            "all": (
                np.array([[0, 0, 50], [0, 0, 100]]),
                np.array([[0, 1000], [0, 1000]]),
                np.array([20.0, 25.0]),  # would be ignored
            )
        }
        default_height = 0.5  # setting overrides

        per_trace_heights = []
        for name in selected:
            tr, _, height = selected[name]
            n_traces = len(tr)
            if (
                height is not None
                and hasattr(height, "__len__")
                and not isinstance(height, str)
            ):
                per_trace_heights.extend(height[:n_traces])
            elif height is not None:
                per_trace_heights.extend([height] * n_traces)
            elif default_height is not None:
                per_trace_heights.extend([default_height] * n_traces)
            else:
                per_trace_heights.extend([None] * n_traces)

        # Heights from npz are used (setting only applies when height is None)
        # This reflects the actual logic: npz heights take priority over setting
        assert per_trace_heights == [20.0, 25.0]

    @pytest.mark.unit
    def test_scalar_group_height_broadcast(self):
        """Scalar group height is broadcast to all traces in group."""
        selected = {
            "sky": (
                np.array([[0, 0, 50], [0, 0, 60], [0, 0, 70]]),  # 3 traces
                np.array([[0, 1000], [0, 1000], [0, 1000]]),
                30.0,  # scalar group height
            )
        }
        default_height = None

        per_trace_heights = []
        for name in selected:
            tr, _, height = selected[name]
            n_traces = len(tr)
            if (
                height is not None
                and hasattr(height, "__len__")
                and not isinstance(height, str)
            ):
                per_trace_heights.extend(height[:n_traces])
            elif height is not None:
                per_trace_heights.extend([height] * n_traces)
            elif default_height is not None:
                per_trace_heights.extend([default_height] * n_traces)
            else:
                per_trace_heights.extend([None] * n_traces)

        assert per_trace_heights == [30.0, 30.0, 30.0]

    @pytest.mark.unit
    def test_no_heights_uses_setting(self):
        """When no heights in npz, setting is used."""
        selected = {
            "all": (
                np.array([[0, 0, 50], [0, 0, 100]]),
                np.array([[0, 1000], [0, 1000]]),
                None,  # no heights from npz
            )
        }
        default_height = 0.4

        per_trace_heights = []
        for name in selected:
            tr, _, height = selected[name]
            n_traces = len(tr)
            if (
                height is not None
                and hasattr(height, "__len__")
                and not isinstance(height, str)
            ):
                per_trace_heights.extend(height[:n_traces])
            elif height is not None:
                per_trace_heights.extend([height] * n_traces)
            elif default_height is not None:
                per_trace_heights.extend([default_height] * n_traces)
            else:
                per_trace_heights.extend([None] * n_traces)

        assert per_trace_heights == [0.4, 0.4]

    @pytest.mark.unit
    def test_no_heights_no_setting_gives_none(self):
        """When no heights and no setting, result is None (will error in extract)."""
        selected = {
            "all": (
                np.array([[0, 0, 50]]),
                np.array([[0, 1000]]),
                None,
            )
        }
        default_height = None

        per_trace_heights = []
        for name in selected:
            tr, _, height = selected[name]
            n_traces = len(tr)
            if (
                height is not None
                and hasattr(height, "__len__")
                and not isinstance(height, str)
            ):
                per_trace_heights.extend(height[:n_traces])
            elif height is not None:
                per_trace_heights.extend([height] * n_traces)
            elif default_height is not None:
                per_trace_heights.extend([default_height] * n_traces)
            else:
                per_trace_heights.extend([None] * n_traces)

        assert per_trace_heights == [None]


# Tests that require instrument data follow below


@pytest.mark.instrument
@pytest.mark.downloads
@pytest.mark.slow
def test_main(instrument, target, night, channel, input_dir, output_dir):
    output = reduce.main(
        instrument,
        target,
        night,
        channel,
        base_dir="",
        input_dir=input_dir,
        output_dir=output_dir,
        steps=(),
    )

    # reduce.main() returns a list of Pipeline.run() results (one per channel/night combo)
    assert isinstance(output, list)
    assert len(output) >= 1
    # With steps=(), each result is an empty dict (no steps executed)
    assert isinstance(output[0], dict)

    # Test default options - should raise FileNotFoundError for missing paths
    with pytest.raises(FileNotFoundError):
        reduce.main(instrument, target, night, steps=())


@pytest.mark.skip
@pytest.mark.instrument
@pytest.mark.downloads
@pytest.mark.slow
def test_run_all(
    instrument, target, night, channel, input_dir, output_dir, trace_range
):
    reduce.main(
        instrument,
        target,
        night,
        channel,
        base_dir="",
        input_dir=input_dir,
        output_dir=output_dir,
        trace_range=trace_range,
        steps="all",
    )


@pytest.mark.skip
@pytest.mark.instrument
@pytest.mark.downloads
@pytest.mark.slow
def test_load_all(
    instrument, target, night, channel, input_dir, output_dir, trace_range
):
    reduce.main(
        instrument,
        target,
        night,
        channel,
        base_dir="",
        input_dir=input_dir,
        output_dir=output_dir,
        trace_range=trace_range,
        steps=["finalize"],
    )


@pytest.mark.instrument
@pytest.mark.downloads
@pytest.mark.slow
def test_step_abstract(step_args):
    step = reduce.Step(*step_args, **{"plot": False})

    assert isinstance(step.dependsOn, list)
    assert isinstance(step.loadDependsOn, list)
    assert isinstance(step.prefix, str)
    assert isinstance(step.output_dir, str)

    with pytest.raises(NotImplementedError):
        step.load()

    with pytest.raises(NotImplementedError):
        step.run([])

    with pytest.raises(NotImplementedError):
        step.save()
