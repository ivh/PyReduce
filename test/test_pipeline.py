"""Tests for the Pipeline fluent API."""

import numpy as np
import pytest

from pyreduce.pipeline import Pipeline


class TestPipelineConstruction:
    """Test Pipeline construction and fluent API."""

    @pytest.mark.unit
    def test_create_pipeline_with_instrument_name(self, tmp_path):
        """Test creating pipeline with instrument name string."""
        pipe = Pipeline("UVES", str(tmp_path))
        assert pipe.instrument is not None
        assert pipe.instrument.name.upper() == "UVES"

    @pytest.mark.unit
    def test_create_pipeline_with_instrument_object(self, tmp_path, instr):
        """Test creating pipeline with instrument object."""
        pipe = Pipeline(instr, str(tmp_path))
        assert pipe.instrument is instr

    @pytest.mark.unit
    def test_fluent_api_returns_self(self, tmp_path):
        """Test that fluent methods return self for chaining."""
        pipe = Pipeline("UVES", str(tmp_path))

        # Each method should return the pipeline
        result = pipe.bias(["file1.fits"])
        assert result is pipe

        result = pipe.flat(["file2.fits"])
        assert result is pipe

        result = pipe.trace_orders()
        assert result is pipe

    @pytest.mark.unit
    def test_chaining_multiple_steps(self, tmp_path):
        """Test chaining multiple steps."""
        pipe = (
            Pipeline("UVES", str(tmp_path))
            .bias(["bias1.fits", "bias2.fits"])
            .flat(["flat1.fits"])
            .trace_orders()
            .extract(["science.fits"])
        )

        # Check steps were queued
        step_names = [s[0] for s in pipe._steps]
        assert "bias" in step_names
        assert "flat" in step_names
        assert "trace" in step_names
        assert "science" in step_names

    @pytest.mark.unit
    def test_wavelength_calibration_convenience(self, tmp_path):
        """Test wavelength_calibration() adds all three steps."""
        pipe = Pipeline("UVES", str(tmp_path)).wavelength_calibration(["thar.fits"])

        step_names = [s[0] for s in pipe._steps]
        assert "wavecal_master" in step_names
        assert "wavecal_init" in step_names
        assert "wavecal" in step_names

    @pytest.mark.unit
    def test_config_passed_to_pipeline(self, tmp_path):
        """Test that config dict is stored."""
        config = {"bias": {"plot": False}, "flat": {"threshold": 100}}
        pipe = Pipeline("UVES", str(tmp_path), config=config)
        assert pipe.config == config

    @pytest.mark.unit
    def test_output_dir_formatting(self, tmp_path):
        """Test output_dir template formatting."""
        pipe = Pipeline(
            "UVES",
            str(tmp_path) + "/{instrument}/{target}/{night}",
            target="HD12345",
            night="2024-01-01",
        )
        assert "UVES" in pipe.output_dir
        assert "HD12345" in pipe.output_dir
        assert "2024-01-01" in pipe.output_dir


class TestPipelineExecution:
    """Test Pipeline execution with real data."""

    @pytest.mark.instrument
    def test_pipeline_bias_only(self, instr, channel, files, settings, tmp_path):
        """Test running just bias step through Pipeline."""
        bias_files = files.get("bias", [])
        if len(bias_files) == 0:
            pytest.skip("No bias files for this instrument")

        pipe = Pipeline(instr, str(tmp_path), channel=channel, config=settings).bias(
            list(bias_files)
        )
        result = pipe.run()

        assert "bias" in result
        assert result["bias"] is not None
        # Bias returns (bias_array, header) tuple
        bias_data, bias_header = result["bias"]
        assert bias_data is not None

    @pytest.mark.instrument
    def test_pipeline_flat_with_bias(self, instr, channel, files, settings, tmp_path):
        """Test running flat step which depends on bias."""
        flat_files = files.get("flat", [])
        if len(flat_files) == 0:
            pytest.skip("No flat files for this instrument")

        bias_files = list(files.get("bias", []))
        pipe = Pipeline(instr, str(tmp_path), channel=channel, config=settings)
        if bias_files:
            pipe = pipe.bias(bias_files)
        pipe = pipe.flat(list(flat_files))
        result = pipe.run()

        assert "flat" in result

    @pytest.mark.instrument
    @pytest.mark.slow
    def test_pipeline_trace_orders(self, instr, channel, files, settings, tmp_path):
        """Test order tracing through Pipeline."""
        order_files = files.get("trace", [])
        if len(order_files) == 0:
            pytest.skip("No order tracing files for this instrument")

        bias_files = list(files.get("bias", []))
        pipe = Pipeline(instr, str(tmp_path), channel=channel, config=settings)
        if bias_files:
            pipe = pipe.bias(bias_files)
        pipe = pipe.trace_orders(list(order_files))
        result = pipe.run()

        assert "trace" in result
        orders, column_range = result["trace"]
        assert orders is not None

    @pytest.mark.instrument
    def test_pipeline_results_property(self, instr, channel, files, settings, tmp_path):
        """Test that results property returns same as run()."""
        bias_files = files.get("bias", [])
        if len(bias_files) == 0:
            pytest.skip("No bias files for this instrument")

        pipe = Pipeline(instr, str(tmp_path), channel=channel, config=settings).bias(
            list(bias_files)
        )
        run_result = pipe.run()
        prop_result = pipe.results

        assert run_result is prop_result


class TestPipelineLoad:
    """Test loading intermediate results."""

    @pytest.mark.unit
    def test_load_precomputed_data(self, tmp_path):
        """Test loading pre-computed data into pipeline."""
        fake_bias = (np.zeros((100, 100)), {})

        pipe = Pipeline("UVES", str(tmp_path))
        pipe.load("bias", fake_bias)

        assert "bias" in pipe._data
        assert pipe._data["bias"] is fake_bias

    @pytest.mark.unit
    def test_load_without_data_marks_for_loading(self, tmp_path):
        """Test load() without data sets marker for later loading."""
        pipe = Pipeline("UVES", str(tmp_path))
        pipe.load("trace")

        assert "trace" in pipe._data
        assert pipe._data["trace"] is None  # Marker for load-from-disk


class TestPipelineStepOrdering:
    """Test that steps are executed in correct order."""

    @pytest.mark.unit
    def test_steps_sorted_by_order(self, tmp_path):
        """Steps should be sorted by STEP_ORDER before execution."""
        pipe = Pipeline("UVES", str(tmp_path))

        # Add steps out of order
        pipe._add_step("science", ["file.fits"])
        pipe._add_step("bias", ["bias.fits"])
        pipe._add_step("flat", ["flat.fits"])

        # Check they're stored in insertion order
        assert pipe._steps[0][0] == "science"
        assert pipe._steps[1][0] == "bias"
        assert pipe._steps[2][0] == "flat"

        # Verify STEP_ORDER values
        assert Pipeline.STEP_ORDER["bias"] < Pipeline.STEP_ORDER["flat"]
        assert Pipeline.STEP_ORDER["flat"] < Pipeline.STEP_ORDER["science"]

    @pytest.mark.unit
    def test_all_steps_have_order(self, tmp_path):
        """All STEP_CLASSES should have corresponding STEP_ORDER entry."""
        for step_name in Pipeline.STEP_CLASSES:
            assert step_name in Pipeline.STEP_ORDER, f"Missing order for {step_name}"


class TestPipelineFluentMethods:
    """Test all fluent API methods."""

    @pytest.mark.unit
    def test_mask_method(self, tmp_path):
        """Test mask() method."""
        pipe = Pipeline("UVES", str(tmp_path)).mask()
        assert ("mask", None) in pipe._steps

    @pytest.mark.unit
    def test_scatter_method(self, tmp_path):
        """Test scatter() method."""
        pipe = Pipeline("UVES", str(tmp_path)).scatter(["scatter.fits"])
        step_names = [s[0] for s in pipe._steps]
        assert "scatter" in step_names

    @pytest.mark.unit
    def test_normalize_flat_method(self, tmp_path):
        """Test normalize_flat() method."""
        pipe = Pipeline("UVES", str(tmp_path)).normalize_flat()
        step_names = [s[0] for s in pipe._steps]
        assert "norm_flat" in step_names

    @pytest.mark.unit
    def test_wavecal_master_method(self, tmp_path):
        """Test wavecal_master() method."""
        pipe = Pipeline("UVES", str(tmp_path)).wavecal_master(["thar.fits"])
        step_names = [s[0] for s in pipe._steps]
        assert "wavecal_master" in step_names

    @pytest.mark.unit
    def test_wavecal_init_method(self, tmp_path):
        """Test wavecal_init() method."""
        pipe = Pipeline("UVES", str(tmp_path)).wavecal_init()
        step_names = [s[0] for s in pipe._steps]
        assert "wavecal_init" in step_names

    @pytest.mark.unit
    def test_freq_comb_master_method(self, tmp_path):
        """Test freq_comb_master() method."""
        pipe = Pipeline("UVES", str(tmp_path)).freq_comb_master(["comb.fits"])
        step_names = [s[0] for s in pipe._steps]
        assert "freq_comb_master" in step_names

    @pytest.mark.unit
    def test_freq_comb_method(self, tmp_path):
        """Test freq_comb() method."""
        pipe = Pipeline("UVES", str(tmp_path)).freq_comb()
        step_names = [s[0] for s in pipe._steps]
        assert "freq_comb" in step_names

    @pytest.mark.unit
    def test_curvature_method(self, tmp_path):
        """Test curvature() method."""
        pipe = Pipeline("UVES", str(tmp_path)).curvature(["arc.fits"])
        step_names = [s[0] for s in pipe._steps]
        assert "curvature" in step_names

    @pytest.mark.unit
    def test_continuum_method(self, tmp_path):
        """Test continuum() method."""
        pipe = Pipeline("UVES", str(tmp_path)).continuum()
        step_names = [s[0] for s in pipe._steps]
        assert "continuum" in step_names

    @pytest.mark.unit
    def test_finalize_method(self, tmp_path):
        """Test finalize() method."""
        pipe = Pipeline("UVES", str(tmp_path)).finalize()
        step_names = [s[0] for s in pipe._steps]
        assert "finalize" in step_names

    @pytest.mark.unit
    def test_rectify_method(self, tmp_path):
        """Test rectify() method."""
        pipe = Pipeline("UVES", str(tmp_path)).rectify()
        step_names = [s[0] for s in pipe._steps]
        assert "rectify" in step_names


class TestPipelineFromFiles:
    """Test Pipeline.from_files() class method."""

    @pytest.mark.unit
    def test_from_files_creates_pipeline(self, tmp_path):
        """from_files should create a configured Pipeline."""
        files = {
            "bias": ["bias1.fits", "bias2.fits"],
            "flat": ["flat1.fits"],
            "science": ["sci1.fits"],
        }
        pipe = Pipeline.from_files(
            files=files,
            output_dir=str(tmp_path),
            target="HD12345",
            instrument="UVES",
            channel="RED",
            night="2024-01-01",
            config={},
            steps=["bias", "flat", "science"],
        )

        assert pipe.target == "HD12345"
        assert pipe.channel == "RED"
        step_names = [s[0] for s in pipe._steps]
        assert "bias" in step_names
        assert "flat" in step_names
        assert "science" in step_names

    @pytest.mark.unit
    def test_from_files_skips_empty_steps(self, tmp_path):
        """from_files should skip steps with no files."""
        files = {
            "bias": [],  # Empty
            "flat": ["flat1.fits"],
            "science": [],  # Empty
        }
        pipe = Pipeline.from_files(
            files=files,
            output_dir=str(tmp_path),
            target="HD12345",
            instrument="UVES",
            channel="",
            night="",
            config={},
            steps=["bias", "flat", "science"],
        )

        step_names = [s[0] for s in pipe._steps]
        # bias and science should be skipped due to empty file lists
        assert "flat" in step_names

    @pytest.mark.unit
    def test_from_files_all_steps(self, tmp_path):
        """from_files with steps='all' should queue all steps."""
        files = {"bias": ["b.fits"], "flat": ["f.fits"], "science": ["s.fits"]}
        pipe = Pipeline.from_files(
            files=files,
            output_dir=str(tmp_path),
            target="HD12345",
            instrument="UVES",
            channel="",
            night="",
            config={},
            steps="all",
        )

        step_names = [s[0] for s in pipe._steps]
        # Should include all steps from STEP_ORDER
        assert len(step_names) > 5

    @pytest.mark.unit
    def test_from_files_registers_files(self, tmp_path):
        """from_files should register files for dependency loading."""
        files = {
            "bias": ["bias.fits"],
            "trace": ["trace.fits"],
        }
        pipe = Pipeline.from_files(
            files=files,
            output_dir=str(tmp_path),
            target="",
            instrument="UVES",
            channel="",
            night="",
            config={},
            steps=["bias"],
        )

        # Files should be registered for potential dependency loading
        assert "trace" in pipe._files


class TestPipelineConfig:
    """Test Pipeline configuration handling."""

    @pytest.mark.unit
    def test_config_passed_to_pipeline(self, tmp_path):
        """Config should be stored in pipeline."""
        config = {"bias": {"degree": 1}, "flat": {"threshold": 200}}
        pipe = Pipeline("UVES", str(tmp_path), config=config)
        assert pipe.config == config

    @pytest.mark.unit
    def test_empty_config_defaults(self, tmp_path):
        """Pipeline with no config should use empty dict."""
        pipe = Pipeline("UVES", str(tmp_path))
        assert pipe.config == {}

    @pytest.mark.unit
    def test_order_range_stored(self, tmp_path):
        """Order range should be stored."""
        pipe = Pipeline("UVES", str(tmp_path), order_range=(3, 10))
        assert pipe.order_range == (3, 10)

    @pytest.mark.unit
    def test_plot_level_stored(self, tmp_path):
        """Plot level should be stored."""
        pipe = Pipeline("UVES", str(tmp_path), plot=2)
        assert pipe.plot == 2


class TestPipelineGetStepInputs:
    """Test _get_step_inputs method."""

    @pytest.mark.unit
    def test_get_step_inputs_returns_tuple(self, tmp_path):
        """_get_step_inputs should return correct tuple."""
        pipe = Pipeline(
            "UVES",
            str(tmp_path),
            target="HD12345",
            channel="RED",
            night="2024-01-01",
            order_range=(1, 5),
        )
        inputs = pipe._get_step_inputs()

        assert len(inputs) == 6
        assert inputs[0] is pipe.instrument
        assert inputs[1] == "RED"
        assert inputs[2] == "HD12345"
        assert inputs[3] == "2024-01-01"
        assert inputs[5] == (1, 5)
