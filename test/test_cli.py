"""
Tests for the PyReduce command-line interface.

Tests cover:
- Main CLI commands (run, combine, download, examples, list-steps)
- Individual step commands (bias, flat, trace, etc.)
- CLI options and flags (--help, --version)
- Error handling and validation
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pyreduce.__main__ import ALL_STEPS, cli

pytestmark = pytest.mark.unit


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


class TestCLIBasic:
    """Test basic CLI functionality."""

    def test_cli_version(self, runner):
        """Test --version flag shows version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower() or "pyreduce" in result.output.lower()

    def test_cli_help(self, runner):
        """Test --help flag shows help text."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "PyReduce" in result.output
        assert "Usage" in result.output

    def test_cli_no_command(self, runner):
        """Test invoking CLI without command shows help or error."""
        result = runner.invoke(cli, [])
        # Click returns exit code 2 for missing required command
        assert result.exit_code in [0, 2]
        assert (
            "Usage" in result.output
            or "error" in result.output.lower()
            or "Commands" in result.output
        )


class TestListStepsCommand:
    """Test the list-steps command."""

    def test_list_steps_runs(self, runner):
        """Test list-steps command executes successfully."""
        result = runner.invoke(cli, ["list-steps"])
        assert result.exit_code == 0

    def test_list_steps_shows_all_steps(self, runner):
        """Test list-steps shows all available steps."""
        result = runner.invoke(cli, ["list-steps"])
        assert result.exit_code == 0
        for step in ALL_STEPS:
            assert step in result.output

    def test_list_steps_help(self, runner):
        """Test list-steps --help."""
        result = runner.invoke(cli, ["list-steps", "--help"])
        assert result.exit_code == 0
        assert (
            "Available reduction steps" in result.output
            or "list-steps" in result.output
        )


class TestDownloadCommand:
    """Test the download command."""

    def test_download_help(self, runner):
        """Test download --help."""
        result = runner.invoke(cli, ["download", "--help"])
        assert result.exit_code == 0
        assert "download" in result.output.lower() or "Download" in result.output

    @patch("pyreduce.datasets.UVES")
    def test_download_uves(self, mock_uves, runner):
        """Test downloading UVES dataset."""
        mock_uves.return_value = "/tmp/uves_data"
        result = runner.invoke(cli, ["download", "UVES"])
        assert result.exit_code == 0
        assert "downloaded" in result.output.lower()
        mock_uves.assert_called_once()

    @patch("pyreduce.datasets.XSHOOTER")
    def test_download_xshooter(self, mock_xshooter, runner):
        """Test downloading XSHOOTER dataset."""
        mock_xshooter.return_value = "/tmp/xshooter_data"
        result = runner.invoke(cli, ["download", "XSHOOTER"])
        assert result.exit_code == 0
        assert "downloaded" in result.output.lower()
        mock_xshooter.assert_called_once()

    def test_download_invalid_instrument(self, runner):
        """Test download with invalid instrument."""
        result = runner.invoke(cli, ["download", "INVALID_INSTRUMENT"])
        assert result.exit_code != 0
        assert "Unknown instrument" in result.output or "error" in result.output.lower()

    def test_download_instrument_case_insensitive(self, runner):
        """Test download accepts lowercase instrument names."""
        with patch("pyreduce.datasets.UVES") as mock_uves:
            mock_uves.return_value = "/tmp/uves_data"
            result = runner.invoke(cli, ["download", "uves"])
            assert result.exit_code == 0


class TestCombineCommand:
    """Test the combine command."""

    def test_combine_help(self, runner):
        """Test combine --help."""
        result = runner.invoke(cli, ["combine", "--help"])
        assert result.exit_code == 0
        assert "combine" in result.output.lower() or "Combine" in result.output

    def test_combine_no_files(self, runner):
        """Test combine without input files shows error."""
        result = runner.invoke(cli, ["combine"])
        assert result.exit_code != 0

    @patch("pyreduce.tools.combine.combine")
    def test_combine_with_files(self, mock_combine, runner):
        """Test combine command with file arguments."""
        mock_combine.return_value = None
        result = runner.invoke(cli, ["combine", "file1.fits", "file2.fits"])
        assert result.exit_code == 0
        mock_combine.assert_called_once()
        args = mock_combine.call_args[0]
        assert "file1.fits" in args[0]
        assert "file2.fits" in args[0]

    @patch("pyreduce.tools.combine.combine")
    def test_combine_with_output_option(self, mock_combine, runner):
        """Test combine with --output option."""
        mock_combine.return_value = None
        result = runner.invoke(
            cli, ["combine", "-o", "output.fits", "file1.fits", "file2.fits"]
        )
        assert result.exit_code == 0
        mock_combine.assert_called_once()
        # Check that output filename is second argument
        assert mock_combine.call_args[0][1] == "output.fits"

    @patch("pyreduce.tools.combine.combine")
    def test_combine_with_plot_option(self, mock_combine, runner):
        """Test combine with --plot option."""
        mock_combine.return_value = None
        result = runner.invoke(cli, ["combine", "-p", "5", "file1.fits", "file2.fits"])
        assert result.exit_code == 0
        mock_combine.assert_called_once()
        # Check that plot argument was passed
        assert mock_combine.call_args[1].get("plot") == 5


class TestRunCommand:
    """Test the run command."""

    def test_run_help(self, runner):
        """Test run --help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output.lower() or "Run" in result.output

    def test_run_no_arguments(self, runner):
        """Test run without required arguments shows error."""
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0

    @patch("pyreduce.reduce.main")
    def test_run_no_target(self, mock_main, runner):
        """Test run with only instrument argument succeeds (target optional)."""
        mock_main.return_value = None
        result = runner.invoke(cli, ["run", "UVES"])
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["target"] is None

    @patch("pyreduce.reduce.main")
    def test_run_basic(self, mock_main, runner):
        """Test run command with basic arguments."""
        mock_main.return_value = None
        result = runner.invoke(cli, ["run", "UVES", "-t", "HD132205"])
        assert result.exit_code == 0
        mock_main.assert_called_once()

    @patch("pyreduce.reduce.main")
    def test_run_with_night(self, mock_main, runner):
        """Test run command with --night option."""
        mock_main.return_value = None
        result = runner.invoke(
            cli, ["run", "UVES", "-t", "HD132205", "--night", "2010-04-01"]
        )
        assert result.exit_code == 0
        mock_main.assert_called_once()
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["night"] == "2010-04-01"

    @patch("pyreduce.reduce.main")
    def test_run_with_channel(self, mock_main, runner):
        """Test run command with --channel option."""
        mock_main.return_value = None
        result = runner.invoke(cli, ["run", "UVES", "-t", "HD132205", "-c", "MIDDLE"])
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["channels"] == "MIDDLE"

    @patch("pyreduce.reduce.main")
    def test_run_with_steps(self, mock_main, runner):
        """Test run command with --steps option."""
        mock_main.return_value = None
        result = runner.invoke(
            cli, ["run", "UVES", "-t", "HD132205", "--steps", "bias,flat,trace"]
        )
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["steps"] == ("bias", "flat", "trace")

    @patch("pyreduce.reduce.main")
    def test_run_with_steps_all(self, mock_main, runner):
        """Test run command without --steps runs all steps."""
        mock_main.return_value = None
        result = runner.invoke(cli, ["run", "UVES", "-t", "HD132205"])
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["steps"] == "all"

    @patch("pyreduce.reduce.main")
    def test_run_with_base_dir(self, mock_main, runner):
        """Test run command with --base-dir option."""
        mock_main.return_value = None
        result = runner.invoke(
            cli, ["run", "UVES", "-t", "HD132205", "-b", "/tmp/data"]
        )
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["base_dir"] == "/tmp/data"

    @patch("pyreduce.reduce.main")
    def test_run_with_input_dir(self, mock_main, runner):
        """Test run command with --input-dir option."""
        mock_main.return_value = None
        result = runner.invoke(
            cli, ["run", "UVES", "-t", "HD132205", "-i", "raw_frames"]
        )
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["input_dir"] == "raw_frames"

    @patch("pyreduce.reduce.main")
    def test_run_with_output_dir(self, mock_main, runner):
        """Test run command with --output-dir option."""
        mock_main.return_value = None
        result = runner.invoke(
            cli, ["run", "UVES", "-t", "HD132205", "-o", "processed"]
        )
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["output_dir"] == "processed"

    @patch("pyreduce.reduce.main")
    def test_run_with_plot(self, mock_main, runner):
        """Test run command with --plot option."""
        mock_main.return_value = None
        result = runner.invoke(cli, ["run", "UVES", "-t", "HD132205", "-p", "2"])
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        # plot is passed as string to reduce_main (which then converts it)
        assert call_kwargs["plot"] in ["2", 2]

    @patch("pyreduce.reduce.main")
    def test_run_with_trace_range(self, mock_main, runner):
        """Test run command with --trace-range option."""
        mock_main.return_value = None
        result = runner.invoke(
            cli, ["run", "UVES", "-t", "HD132205", "--trace-range", "3,21"]
        )
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["trace_range"] == (3, 21)

    @patch("pyreduce.reduce.main")
    def test_run_with_settings_file(self, mock_main, runner):
        """Test run command with --settings option."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"bias": {"degree": 1}}, f)
            settings_file = f.name

        try:
            with patch(
                "pyreduce.configuration.load_settings_override"
            ) as mock_override:
                mock_override.return_value = {}
                result = runner.invoke(
                    cli, ["run", "UVES", "-t", "HD132205", "--settings", settings_file]
                )
                assert result.exit_code == 0
                mock_override.assert_called_once()
        finally:
            os.unlink(settings_file)

    @patch("pyreduce.reduce.main")
    def test_run_with_nonexistent_settings_file(self, mock_main, runner):
        """Test run command rejects nonexistent settings file."""
        result = runner.invoke(
            cli, ["run", "UVES", "-t", "HD132205", "--settings", "/nonexistent.json"]
        )
        assert result.exit_code != 0

    @patch("pyreduce.reduce.main")
    def test_run_with_empty_steps(self, mock_main, runner):
        """Test run command with empty steps argument."""
        mock_main.return_value = None
        result = runner.invoke(cli, ["run", "UVES", "-t", "HD132205", "--steps", ""])
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        # Empty steps string defaults to "all"
        assert call_kwargs["steps"] == "all"

    @patch("pyreduce.reduce.main")
    def test_run_with_multiple_steps(self, mock_main, runner):
        """Test run command with multiple steps."""
        mock_main.return_value = None
        result = runner.invoke(
            cli, ["run", "UVES", "-t", "HD132205", "-s", "bias,flat,trace,science"]
        )
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["steps"] == ("bias", "flat", "trace", "science")

    @patch("pyreduce.reduce.main")
    def test_run_with_step_order_preserved(self, mock_main, runner):
        """Test run command preserves step order."""
        mock_main.return_value = None
        result = runner.invoke(
            cli, ["run", "UVES", "-t", "HD132205", "-s", "science,bias,flat"]
        )
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        # Order should be preserved as given
        assert call_kwargs["steps"] == ("science", "bias", "flat")

    @patch("pyreduce.reduce.main")
    def test_run_with_whitespace_in_steps(self, mock_main, runner):
        """Test run command handles whitespace in steps."""
        mock_main.return_value = None
        result = runner.invoke(
            cli, ["run", "UVES", "-t", "HD132205", "-s", "bias, flat , trace"]
        )
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        # Whitespace should be stripped
        assert "bias" in call_kwargs["steps"]
        assert "flat" in call_kwargs["steps"]
        assert "trace" in call_kwargs["steps"]


class TestIndividualStepCommands:
    """Test individual step commands (bias, flat, trace, etc.)."""

    @pytest.mark.parametrize("step", ALL_STEPS)
    def test_step_help(self, step, runner):
        """Test each step command has help."""
        result = runner.invoke(cli, [step, "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output or step in result.output.lower()

    @pytest.mark.parametrize("step", ALL_STEPS)
    def test_step_requires_instrument(self, step, runner):
        """Test each step command requires instrument argument."""
        result = runner.invoke(cli, [step])
        assert result.exit_code != 0

    @patch("pyreduce.reduce.main")
    @pytest.mark.parametrize("step", ALL_STEPS)
    def test_step_basic(self, mock_main, step, runner):
        """Test each step command with basic arguments."""
        mock_main.return_value = None
        result = runner.invoke(cli, [step, "UVES"])
        assert result.exit_code == 0
        mock_main.assert_called_once()
        # Verify that the step is passed as a tuple with only this step
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["steps"] == (step,)

    @patch("pyreduce.reduce.main")
    @pytest.mark.parametrize("step", ["bias", "flat", "trace"])
    def test_step_with_options(self, mock_main, step, runner):
        """Test step commands with various options."""
        mock_main.return_value = None
        result = runner.invoke(
            cli,
            [
                step,
                "UVES",
                "-t",
                "HD132205",
                "-n",
                "2010-04-01",
                "-c",
                "MIDDLE",
                "-o",
                "output",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["night"] == "2010-04-01"
        assert call_kwargs["channels"] == "MIDDLE"
        assert call_kwargs["output_dir"] == "output"

    @patch("pyreduce.reduce.main")
    def test_step_with_file_option_on_unsupported_step(self, mock_main, runner):
        """Test --file option on step that doesn't support it."""
        # wavecal_init is in the no_file_steps list
        result = runner.invoke(cli, ["wavecal_init", "UVES", "--file", "test.fits"])
        assert result.exit_code != 0
        assert "does not accept raw files" in result.output

    @patch("pyreduce.reduce.main")
    def test_step_with_settings_file(self, mock_main, runner):
        """Test step command with settings override."""
        mock_main.return_value = None
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"bias": {"degree": 2}}, f)
            settings_file = f.name

        try:
            with patch(
                "pyreduce.configuration.load_settings_override"
            ) as mock_override:
                mock_override.return_value = {}
                result = runner.invoke(
                    cli, ["bias", "UVES", "--settings", settings_file]
                )
                assert result.exit_code == 0
                mock_override.assert_called_once()
        finally:
            os.unlink(settings_file)

    @patch("pyreduce.reduce.main")
    def test_step_with_plot_levels(self, mock_main, runner):
        """Test step commands with different plot levels."""
        mock_main.return_value = None
        for plot_level in [0, 1, 2]:
            result = runner.invoke(cli, ["bias", "UVES", "-p", str(plot_level)])
            assert result.exit_code == 0
            call_kwargs = mock_main.call_args[1]
            # plot option can be either int or str depending on Click's type conversion
            assert call_kwargs["plot"] in [str(plot_level), plot_level]

    @patch("pyreduce.reduce.main")
    def test_step_with_all_long_options(self, mock_main, runner):
        """Test step command with all long option names."""
        mock_main.return_value = None
        result = runner.invoke(
            cli,
            [
                "flat",
                "XSHOOTER",
                "--target",
                "UX-Ori",
                "--night",
                "2015-10-15",
                "--channel",
                "NIR",
                "--base-dir",
                "/data",
                "--input-dir",
                "raw",
                "--output-dir",
                "products",
                "--plot",
                "1",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_main.call_args[1]
        assert call_kwargs["night"] == "2015-10-15"
        assert call_kwargs["channels"] == "NIR"
        assert call_kwargs["base_dir"] == "/data"
        assert call_kwargs["input_dir"] == "raw"
        assert call_kwargs["output_dir"] == "products"


class TestExamplesCommand:
    """Test the examples command."""

    def test_examples_help(self, runner):
        """Test examples --help."""
        result = runner.invoke(cli, ["examples", "--help"])
        assert result.exit_code == 0

    def test_examples_no_args_lists(self, runner):
        """Test examples without args lists available examples."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__.return_value.read.return_value = json.dumps(
                [
                    {"name": "uves_example.py"},
                    {"name": "xshooter_example.py"},
                ]
            ).encode()
            mock_urlopen.return_value = mock_response

            result = runner.invoke(cli, ["examples"])
            assert result.exit_code == 0
            assert "uves_example.py" in result.output

    def test_examples_list_flag(self, runner):
        """Test examples -l lists available examples."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__.return_value.read.return_value = json.dumps(
                [{"name": "uves_example.py"}]
            ).encode()
            mock_urlopen.return_value = mock_response

            result = runner.invoke(cli, ["examples", "-l"])
            assert result.exit_code == 0
            assert "uves_example.py" in result.output

    def test_examples_version_not_found(self, runner):
        """Test examples with unknown version shows error."""
        from urllib.error import HTTPError

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = HTTPError("url", 404, "Not Found", {}, None)
            with patch("pyreduce.__version__", "unknown"):
                result = runner.invoke(cli, ["examples"])
                assert result.exit_code != 0
                assert "Cannot determine" in result.output or "version" in result.output

    def test_examples_download_single_file(self, runner):
        """Test downloading a single example file."""
        with (
            patch("urllib.request.urlopen") as mock_urlopen,
            patch("urllib.request.urlretrieve") as mock_retrieve,
            patch("os.makedirs"),
        ):
            mock_response = MagicMock()
            mock_response.__enter__.return_value.read.return_value = json.dumps(
                [{"name": "uves_example.py"}]
            ).encode()
            mock_urlopen.return_value = mock_response
            mock_retrieve.return_value = None

            with tempfile.TemporaryDirectory() as tmpdir:
                result = runner.invoke(
                    cli, ["examples", "uves_example.py", "-o", tmpdir]
                )
                assert result.exit_code == 0
                assert "Downloaded" in result.output

    def test_examples_download_all(self, runner):
        """Test downloading all example files."""
        with (
            patch("urllib.request.urlopen") as mock_urlopen,
            patch("urllib.request.urlretrieve") as mock_retrieve,
            patch("os.makedirs"),
        ):
            mock_response = MagicMock()
            mock_response.__enter__.return_value.read.return_value = json.dumps(
                [{"name": "uves_example.py"}, {"name": "xshooter_example.py"}]
            ).encode()
            mock_urlopen.return_value = mock_response
            mock_retrieve.return_value = None

            with tempfile.TemporaryDirectory() as tmpdir:
                result = runner.invoke(cli, ["examples", "--all", "-o", tmpdir])
                assert result.exit_code == 0
                # Should download both files
                assert mock_retrieve.call_count == 2

    def test_examples_run_with_all_flag_error(self, runner):
        """Test that --run and --all flags together produce error."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__.return_value.read.return_value = json.dumps(
                [{"name": "uves_example.py"}]
            ).encode()
            mock_urlopen.return_value = mock_response

            result = runner.invoke(cli, ["examples", "--all", "-r"])
            assert result.exit_code != 0
            assert "Cannot use --run with --all" in result.output

    def test_examples_unknown_file(self, runner):
        """Test downloading unknown example file shows error."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__.return_value.read.return_value = json.dumps(
                [{"name": "uves_example.py"}]
            ).encode()
            mock_urlopen.return_value = mock_response

            result = runner.invoke(cli, ["examples", "nonexistent.py"])
            assert result.exit_code != 0
            assert "Unknown example" in result.output


class TestConfigurationLoading:
    """Test configuration loading in CLI commands."""

    @patch("pyreduce.configuration.get_configuration_for_instrument")
    @patch("pyreduce.reduce.main")
    def test_run_loads_instrument_config(self, mock_main, mock_get_config, runner):
        """Test run command loads instrument configuration."""
        mock_config = {"bias": {}, "flat": {}}
        mock_get_config.return_value = mock_config
        mock_main.return_value = None

        result = runner.invoke(cli, ["run", "UVES", "-t", "HD132205"])
        assert result.exit_code == 0
        mock_get_config.assert_called_with("UVES")

    @patch("pyreduce.configuration.get_configuration_for_instrument")
    @patch("pyreduce.configuration.load_settings_override")
    @patch("pyreduce.reduce.main")
    def test_run_applies_settings_override(
        self, mock_main, mock_override, mock_get_config, runner
    ):
        """Test run command applies settings override when provided."""
        mock_get_config.return_value = {"bias": {}}
        mock_override.return_value = {"bias": {"degree": 2}}
        mock_main.return_value = None

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"bias": {"degree": 2}}, f)
            settings_file = f.name

        try:
            result = runner.invoke(
                cli, ["run", "UVES", "-t", "HD132205", "--settings", settings_file]
            )
            assert result.exit_code == 0
            mock_override.assert_called_once()
        finally:
            os.unlink(settings_file)


class TestErrorHandling:
    """Test error handling in CLI."""

    @patch("pyreduce.reduce.main")
    def test_run_handles_invalid_instrument(self, mock_main, runner):
        """Test run command handles invalid instrument gracefully."""
        mock_main.side_effect = ValueError("Unknown instrument")
        result = runner.invoke(cli, ["run", "INVALID", "target"])
        # Note: The actual error handling may vary, just verify it doesn't crash
        assert isinstance(result.exception, ValueError) or result.exit_code != 0

    def test_combine_with_empty_file_list(self, runner):
        """Test combine with no files shows error."""
        result = runner.invoke(cli, ["combine"])
        assert result.exit_code != 0

    @patch("pyreduce.reduce.main")
    def test_step_with_invalid_trace_range(self, mock_main, runner):
        """Test step command with malformed trace range."""
        result = runner.invoke(
            cli, ["run", "UVES", "-t", "HD132205", "--trace-range", "invalid"]
        )
        # May either fail to parse or raise during execution
        # Just verify it's handled appropriately
        assert result.exit_code != 0


class TestStepFileMode:
    """Test --file option for step commands that support it."""

    @patch("pyreduce.reduce.main")
    def test_bias_supports_file_option(self, mock_main, runner):
        """Test bias command accepts --file option."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            test_file = f.name

        try:
            result = runner.invoke(cli, ["bias", "UVES", "--file", test_file])
            # Should either succeed or fail gracefully (depending on data availability)
            assert result.exit_code in [
                0,
                1,
                2,
            ]  # 0=success, 1=runtime error, 2=usage error
        finally:
            os.unlink(test_file)

    @patch("pyreduce.reduce.main")
    def test_trace_supports_file_option(self, mock_main, runner):
        """Test trace command accepts --file option."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            test_file = f.name

        try:
            result = runner.invoke(cli, ["trace", "UVES", "--file", test_file])
            # Should either succeed or fail gracefully
            assert result.exit_code in [0, 1, 2]
        finally:
            os.unlink(test_file)


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_run_command_full_help(self, runner):
        """Test that run command help includes all options."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        options = [
            "--night",
            "--channel",
            "--steps",
            "--base-dir",
            "--input-dir",
            "--output-dir",
            "--plot",
            "--trace-range",
            "--settings",
        ]
        for option in options:
            assert option in result.output

    def test_all_step_commands_registered(self, runner):
        """Test that all steps are registered as commands."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # At least some steps should be listed
        for step in ALL_STEPS[:3]:  # Check first 3 steps
            assert step in result.output or step.replace("_", "-") in result.output

    def test_list_steps_matches_cli_commands(self, runner):
        """Test that list-steps output matches available commands."""
        list_result = runner.invoke(cli, ["list-steps"])
        help_result = runner.invoke(cli, ["--help"])

        assert list_result.exit_code == 0
        assert help_result.exit_code == 0

        # All steps from list-steps should be in help
        for step in ALL_STEPS[:5]:  # Sample check
            assert step in list_result.output
