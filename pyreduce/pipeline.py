"""
Fluent Pipeline API for PyReduce.

Provides a cleaner interface for building and running reduction pipelines.
Wraps the existing Step classes internally for backward compatibility.

Example usage:
    from pyreduce.pipeline import Pipeline

    # Simple: auto-discover files for an instrument
    result = Pipeline.from_instrument(
        instrument="UVES",
        target="HD132205",
        night="2010-04-01",
        channel="middle",
        base_dir="/data",
    ).run()

    # Or build manually with explicit files:
    result = (
        Pipeline("UVES", output_dir, config=settings)
        .bias(bias_files)
        .flat(flat_files)
        .trace_orders(order_files)
        .extract(science_files)
        .run()
    )
"""

from __future__ import annotations

import logging
import os
from os.path import join
from typing import TYPE_CHECKING

from . import util
from .configuration import load_config
from .instruments.instrument_info import load_instrument
from .reduce import (
    BackgroundScatter,
    Bias,
    ContinuumNormalization,
    Finalize,
    Flat,
    LaserFrequencyCombFinalize,
    LaserFrequencyCombMaster,
    Mask,
    NormalizeFlatField,
    OrderTracing,
    RectifyImage,
    ScienceExtraction,
    SlitCurvatureDetermination,
    WavelengthCalibrationFinalize,
    WavelengthCalibrationInitialize,
    WavelengthCalibrationMaster,
)

if TYPE_CHECKING:
    from .instruments.common import Instrument

logger = logging.getLogger(__name__)


class Pipeline:
    """Fluent API for building reduction pipelines."""

    STEP_CLASSES = {
        "mask": Mask,
        "bias": Bias,
        "flat": Flat,
        "orders": OrderTracing,
        "scatter": BackgroundScatter,
        "norm_flat": NormalizeFlatField,
        "wavecal_master": WavelengthCalibrationMaster,
        "wavecal_init": WavelengthCalibrationInitialize,
        "wavecal": WavelengthCalibrationFinalize,
        "freq_comb_master": LaserFrequencyCombMaster,
        "freq_comb": LaserFrequencyCombFinalize,
        "curvature": SlitCurvatureDetermination,
        "science": ScienceExtraction,
        "continuum": ContinuumNormalization,
        "finalize": Finalize,
        "rectify": RectifyImage,
    }

    STEP_ORDER = {
        "mask": 5,
        "bias": 10,
        "flat": 20,
        "orders": 30,
        "curvature": 40,
        "scatter": 45,
        "norm_flat": 50,
        "wavecal_master": 60,
        "wavecal_init": 64,
        "wavecal": 67,
        "freq_comb_master": 70,
        "freq_comb": 72,
        "rectify": 75,
        "science": 80,
        "continuum": 90,
        "finalize": 100,
    }

    def __init__(
        self,
        instrument: Instrument | str,
        output_dir: str,
        target: str = "",
        channel: str = "",
        night: str = "",
        config: dict | None = None,
        order_range: tuple[int, int] | None = None,
        plot: int = 0,
        plot_dir: str | None = None,
    ):
        """Initialize a reduction pipeline.

        Parameters
        ----------
        instrument : Instrument or str
            Instrument instance or name to load
        output_dir : str
            Directory for output files
        target : str, optional
            Target name for output file naming
        channel : str, optional
            Instrument channel (e.g., "RED", "BLUE")
        night : str, optional
            Observation night string
        config : dict, optional
            Configuration dict with step-specific settings
        order_range : tuple, optional
            (first, last+1) orders to process
        plot : int, optional
            Plot level (0=off, 1=basic, 2=detailed). Default 0.
        plot_dir : str, optional
            Directory to save plots as PNG files. If None, plots are shown interactively.
        """
        if isinstance(instrument, str):
            instrument = load_instrument(instrument)

        self.instrument = instrument
        self.output_dir = output_dir.format(
            instrument=instrument.name.upper(),
            target=target,
            night=night,
            channel=channel,
        )
        self.target = target
        self.channel = channel
        self.night = night
        self.config = config or {}
        self.order_range = order_range
        self.plot = plot
        self.plot_dir = plot_dir

        # Set global plot directory for util.show_or_save()
        util.set_plot_dir(plot_dir)

        self._steps: list[tuple[str, list | None]] = []
        self._data: dict = {}
        self._files: dict = {}

    def _add_step(self, name: str, files: list | None = None) -> Pipeline:
        """Add a step to the pipeline."""
        self._steps.append((name, files))
        if files is not None:
            self._files[name] = files
        return self

    # Step methods - fluent API

    def mask(self) -> Pipeline:
        """Load or create bad pixel mask."""
        return self._add_step("mask")

    def bias(self, files: list[str]) -> Pipeline:
        """Combine bias frames into master bias."""
        return self._add_step("bias", files)

    def flat(self, files: list[str]) -> Pipeline:
        """Combine flat frames into master flat."""
        return self._add_step("flat", files)

    def trace_orders(self, files: list[str] | None = None) -> Pipeline:
        """Trace echelle orders on flat field.

        If files not provided, uses flat from previous step.
        """
        return self._add_step("orders", files)

    def curvature(self, files: list[str] | None = None) -> Pipeline:
        """Determine slit curvature (tilt/shear)."""
        return self._add_step("curvature", files)

    def scatter(self, files: list[str] | None = None) -> Pipeline:
        """Fit background scatter model."""
        return self._add_step("scatter", files)

    def normalize_flat(self) -> Pipeline:
        """Normalize flat field, extract blaze function."""
        return self._add_step("norm_flat")

    def wavecal_master(self, files: list[str]) -> Pipeline:
        """Extract wavelength calibration spectrum."""
        return self._add_step("wavecal_master", files)

    def wavecal_init(self) -> Pipeline:
        """Initialize wavelength solution from line atlas."""
        return self._add_step("wavecal_init")

    def wavecal(self) -> Pipeline:
        """Finalize wavelength calibration."""
        return self._add_step("wavecal")

    def wavelength_calibration(self, files: list[str]) -> Pipeline:
        """Full wavelength calibration (master + init + finalize)."""
        return self.wavecal_master(files).wavecal_init().wavecal()

    def freq_comb_master(self, files: list[str]) -> Pipeline:
        """Extract laser frequency comb spectrum."""
        return self._add_step("freq_comb_master", files)

    def freq_comb(self) -> Pipeline:
        """Finalize frequency comb calibration."""
        return self._add_step("freq_comb")

    def extract(self, files: list[str]) -> Pipeline:
        """Extract science spectra."""
        return self._add_step("science", files)

    def continuum(self) -> Pipeline:
        """Normalize continuum."""
        return self._add_step("continuum")

    def finalize(self) -> Pipeline:
        """Write final output files."""
        return self._add_step("finalize")

    def rectify(self) -> Pipeline:
        """Rectify 2D image."""
        return self._add_step("rectify")

    # Loading intermediate results

    def load(self, step: str, data=None) -> Pipeline:
        """Load intermediate result instead of computing.

        Parameters
        ----------
        step : str
            Name of step whose output to load
        data : any, optional
            Data to use directly instead of loading from disk
        """
        if data is not None:
            self._data[step] = data
        else:
            # Will be loaded during run()
            self._data[step] = None  # Marker to load
        return self

    # Execution

    def _get_step_inputs(self) -> tuple:
        """Get the standard inputs for Step classes."""
        return (
            self.instrument,
            self.channel,
            self.target,
            self.night,
            self.output_dir,
            self.order_range,
        )

    def _run_step(self, name: str, files: list | None, load_only: bool = False):
        """Run or load a single step."""
        step_class = self.STEP_CLASSES[name]
        step_config = self.config.get(name, {}).copy()
        step_config["plot"] = self.plot  # Runtime plot setting
        step = step_class(*self._get_step_inputs(), **step_config)

        # Get dependencies
        deps = step.loadDependsOn if load_only else step.dependsOn
        for dep in deps:
            if dep not in self._data:
                self._ensure_dependency(dep)
        dep_args = {d: self._data[d] for d in deps}

        if load_only:
            try:
                logger.info("Loading data from step '%s'", name)
                return step.load(**dep_args)
            except FileNotFoundError:
                logger.warning(
                    "Intermediate files for step '%s' not found, running instead.",
                    name,
                )
                return self._run_step(name, files, load_only=False)

        logger.info("Running step '%s'", name)
        if files is not None:
            dep_args["files"] = files
        return step.run(**dep_args)

    def _ensure_dependency(self, name: str):
        """Ensure a dependency is available (load if needed)."""
        if name in self._data:
            return

        # 'config' is a special dependency - it's the full config dict, not a step
        if name == "config":
            self._data["config"] = self.config
            return

        files = self._files.get(name)
        self._data[name] = self._run_step(name, files, load_only=True)

    def run(self, skip_existing: bool = False) -> dict:
        """Execute all queued steps.

        Parameters
        ----------
        skip_existing : bool
            If True, skip steps whose output files already exist

        Returns
        -------
        dict
            Results keyed by step name
        """
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Sort steps by execution order
        sorted_steps = sorted(self._steps, key=lambda x: self.STEP_ORDER.get(x[0], 999))

        for name, files in sorted_steps:
            # Check if already computed
            if name in self._data and self._data[name] is not None:
                continue

            result = self._run_step(name, files)
            self._data[name] = result

        return self._data

    @property
    def results(self) -> dict:
        """Access results after run()."""
        return self._data

    @classmethod
    def from_files(
        cls,
        files: dict,
        output_dir: str,
        target: str,
        instrument,
        channel: str,
        night: str,
        config: dict,
        order_range=None,
        steps="all",
        plot: int = 0,
        plot_dir: str | None = None,
    ) -> Pipeline:
        """Create pipeline from a files dict and run specified steps.

        This provides a simpler interface similar to the legacy Reducer class.

        Parameters
        ----------
        files : dict
            Files for each step (bias, flat, orders, wavecal, science, etc.)
        output_dir : str
            Output directory
        target : str
            Target name
        instrument : Instrument or str
            Instrument instance or name
        channel : str
            Instrument channel
        night : str
            Observation night
        config : dict
            Configuration dict
        order_range : tuple, optional
            Order range to process
        steps : list or "all"
            Steps to run
        plot : int, optional
            Plot level (0=off, 1=basic, 2=detailed). Default 0.
        plot_dir : str, optional
            Directory to save plots as PNG files. If None, plots are shown interactively.

        Returns
        -------
        Pipeline
            Configured pipeline ready to run
        """
        pipe = cls(
            instrument=instrument,
            output_dir=output_dir,
            target=target,
            channel=channel,
            night=night,
            config=config,
            order_range=order_range,
            plot=plot,
            plot_dir=plot_dir,
        )

        if steps == "all":
            steps = list(cls.STEP_ORDER.keys())

        # Register files for steps that may be needed as dependencies
        # (even if the step itself isn't in the steps list)
        for key in [
            "bias",
            "flat",
            "orders",
            "curvature",
            "scatter",
            "wavecal_master",
            "freq_comb_master",
            "science",
        ]:
            if key in files and len(files.get(key, [])):
                pipe._files[key] = files[key]

        # Map step names to pipeline methods
        # Use len() for truth checks since files can be numpy arrays
        step_map = {
            "bias": lambda: pipe.bias(files.get("bias", []))
            if len(files.get("bias", []))
            else pipe,
            "flat": lambda: pipe.flat(files.get("flat", []))
            if len(files.get("flat", []))
            else pipe,
            "orders": lambda: pipe.trace_orders(files.get("orders")),
            "curvature": lambda: pipe.curvature(files.get("curvature")),
            "scatter": lambda: pipe.scatter(files.get("scatter")),
            "norm_flat": lambda: pipe.normalize_flat(),
            "wavecal_master": lambda: pipe.wavecal_master(
                files.get("wavecal_master", [])
            )
            if len(files.get("wavecal_master", []))
            else pipe,
            "wavecal_init": lambda: pipe.wavecal_init(),
            "wavecal": lambda: pipe.wavecal(),
            "freq_comb_master": lambda: pipe.freq_comb_master(
                files.get("freq_comb_master", [])
            )
            if len(files.get("freq_comb_master", []))
            else pipe,
            "freq_comb": lambda: pipe.freq_comb(),
            "rectify": lambda: pipe.rectify(),
            "science": lambda: pipe.extract(files.get("science", []))
            if len(files.get("science", []))
            else pipe,
            "continuum": lambda: pipe.continuum(),
            "finalize": lambda: pipe.finalize(),
        }

        for step in steps:
            if step in step_map:
                step_map[step]()

        return pipe

    @classmethod
    def from_instrument(
        cls,
        instrument: str,
        target: str,
        night: str | None = None,
        channel: str | None = None,
        steps: tuple | list | str = "all",
        base_dir: str | None = None,
        input_dir: str | None = None,
        output_dir: str | None = None,
        configuration: dict | None = None,
        order_range: tuple[int, int] | None = None,
        allow_calibration_only: bool = False,
        plot: int = 0,
        plot_dir: str | None = None,
    ) -> Pipeline:
        """Create pipeline from instrument name with automatic file discovery.

        This is the recommended entry point for running reductions. It handles
        loading the instrument, finding and sorting files, and setting up
        the pipeline with the correct configuration.

        Parameters
        ----------
        instrument : str
            Instrument name (e.g., "UVES", "HARPS", "XSHOOTER")
        target : str
            Target name or regex pattern to match in headers
        night : str, optional
            Observation night (YYYY-MM-DD format or regex)
        channel : str, optional
            Instrument channel (e.g., "RED", "BLUE", "middle"). If None,
            uses all available channels for the instrument.
        steps : tuple, list, or "all"
            Steps to run. Default "all" runs all applicable steps.
        base_dir : str, optional
            Base directory for data. Default: $REDUCE_DATA or ~/REDUCE_DATA
        input_dir : str, optional
            Input directory relative to base_dir. Default: from config
        output_dir : str, optional
            Output directory relative to base_dir. Default: from config
        configuration : dict, optional
            Configuration overrides. Default: instrument defaults
        order_range : tuple, optional
            (first, last+1) orders to process
        allow_calibration_only : bool
            If True, allow running without science files
        plot : int
            Plot level (0=off, 1=basic, 2=detailed)
        plot_dir : str, optional
            Directory to save plots. If None, shows interactively.

        Returns
        -------
        Pipeline
            Configured pipeline ready to call .run()

        Example
        -------
        >>> result = Pipeline.from_instrument(
        ...     instrument="UVES",
        ...     target="HD132205",
        ...     night="2010-04-01",
        ...     channel="middle",
        ...     steps=("bias", "flat", "orders", "science"),
        ... ).run()
        """
        # Environment variable overrides for plot
        if "PYREDUCE_PLOT" in os.environ:
            plot = int(os.environ["PYREDUCE_PLOT"])
        if "PYREDUCE_PLOT_DIR" in os.environ:
            plot_dir = os.environ["PYREDUCE_PLOT_DIR"]

        # Set global plot directory
        util.set_plot_dir(plot_dir)

        # Load configuration
        config = load_config(configuration, instrument, 0)

        # Load instrument
        inst = load_instrument(instrument)
        info = inst.info

        # Get directories from config if not specified
        if base_dir is None:
            base_dir = config["reduce"]["base_dir"]
        if input_dir is None:
            input_dir = config["reduce"]["input_dir"]
        if output_dir is None:
            output_dir = config["reduce"]["output_dir"]

        full_input_dir = join(base_dir, input_dir)
        full_output_dir = join(base_dir, output_dir)

        # Get channels to process
        if channel is None:
            channels = info["channels"]
        else:
            channels = [channel] if isinstance(channel, str) else channel

        # Find and sort files
        files = inst.sort_files(
            full_input_dir,
            target,
            night,
            channel=channels[0] if len(channels) == 1 else channels[0],
            **config["instrument"],
            allow_calibration_only=allow_calibration_only,
        )

        if len(files) == 0:
            logger.warning(
                "No files found for instrument: %s, target: %s, night: %s, channel: %s",
                instrument,
                target,
                night,
                channel,
            )
            raise FileNotFoundError(
                f"No files found for {instrument} / {target} / {night} / {channel}"
            )

        # Use the first file set (for single channel)
        k, f = files[0]
        logger.info("Pipeline settings:")
        for key, value in k.items():
            logger.info("  %s: %s", key, value)

        # Create pipeline
        pipe = cls.from_files(
            files=f,
            output_dir=full_output_dir,
            target=k.get("target", target),
            instrument=inst,
            channel=channels[0],
            night=k.get("night", night or ""),
            config=config,
            order_range=order_range,
            steps=steps,
            plot=plot,
            plot_dir=plot_dir,
        )

        return pipe
