"""Tests for the Pipeline fluent API."""

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
        assert "orders" in step_names
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
    def test_pipeline_bias_only(self, instr, arm, files, settings, tmp_path):
        """Test running just bias step through Pipeline."""
        bias_files = files.get("bias", [])
        if len(bias_files) == 0:
            pytest.skip("No bias files for this instrument")

        pipe = Pipeline(instr, str(tmp_path), arm=arm, config=settings).bias(
            list(bias_files)
        )
        result = pipe.run()

        assert "bias" in result
        assert result["bias"] is not None
        # Bias returns (bias_array, header) tuple
        bias_data, bias_header = result["bias"]
        assert bias_data is not None

    @pytest.mark.instrument
    def test_pipeline_flat_with_bias(self, instr, arm, files, settings, tmp_path):
        """Test running flat step which depends on bias."""
        flat_files = files.get("flat", [])
        if len(flat_files) == 0:
            pytest.skip("No flat files for this instrument")

        bias_files = list(files.get("bias", []))
        pipe = Pipeline(instr, str(tmp_path), arm=arm, config=settings)
        if bias_files:
            pipe = pipe.bias(bias_files)
        pipe = pipe.flat(list(flat_files))
        result = pipe.run()

        assert "flat" in result

    @pytest.mark.instrument
    def test_pipeline_trace_orders(self, instr, arm, files, settings, tmp_path):
        """Test order tracing through Pipeline."""
        order_files = files.get("orders", [])
        if len(order_files) == 0:
            pytest.skip("No order tracing files for this instrument")

        bias_files = list(files.get("bias", []))
        pipe = Pipeline(instr, str(tmp_path), arm=arm, config=settings)
        if bias_files:
            pipe = pipe.bias(bias_files)
        pipe = pipe.trace_orders(list(order_files))
        result = pipe.run()

        assert "orders" in result
        orders, column_range = result["orders"]
        assert orders is not None

    @pytest.mark.instrument
    def test_pipeline_results_property(self, instr, arm, files, settings, tmp_path):
        """Test that results property returns same as run()."""
        bias_files = files.get("bias", [])
        if len(bias_files) == 0:
            pytest.skip("No bias files for this instrument")

        pipe = Pipeline(instr, str(tmp_path), arm=arm, config=settings).bias(
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
        import numpy as np

        fake_bias = (np.zeros((100, 100)), {})

        pipe = Pipeline("UVES", str(tmp_path))
        pipe.load("bias", fake_bias)

        assert "bias" in pipe._data
        assert pipe._data["bias"] is fake_bias
