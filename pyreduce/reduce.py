"""
REDUCE script for spectrograph data

Authors
-------
Ansgar Wehrhahn  (ansgar.wehrhahn@physics.uu.se)
Thomas Marquart  (thomas.marquart@physics.uu.se)
Alexis Lavail    (alexis.lavail@physics.uu.se)
Nikolai Piskunov (nikolai.piskunov@physics.uu.se)

Version
-------
1.0 - Initial PyReduce

License
--------
...

"""

import logging
import os
import warnings
from itertools import product
from os.path import join

import numpy as np

from . import instruments, util
from .configuration import load_config

# Step classes live in pyreduce.steps; re-exported here for backwards compatibility
from .steps import (
    BackgroundScatter,
    Bias,
    CalibrationStep,
    ContinuumNormalization,
    ExtractionStep,
    Finalize,
    FitsIOStep,
    Flat,
    LaserFrequencyCombFinalize,
    LaserFrequencyCombMaster,
    Mask,
    NormalizeFlatField,
    RectifyImage,
    ScienceExtraction,
    SlitCurvatureDetermination,
    Step,
    Trace,
    WavelengthCalibrationFinalize,
    WavelengthCalibrationInitialize,
    WavelengthCalibrationMaster,
    wavelengths_from_traces,
)

__all__ = [
    "BackgroundScatter",
    "Bias",
    "CalibrationStep",
    "ContinuumNormalization",
    "ExtractionStep",
    "Finalize",
    "FitsIOStep",
    "Flat",
    "LaserFrequencyCombFinalize",
    "LaserFrequencyCombMaster",
    "Mask",
    "NormalizeFlatField",
    "RectifyImage",
    "ScienceExtraction",
    "SlitCurvatureDetermination",
    "Step",
    "Trace",
    "WavelengthCalibrationFinalize",
    "WavelengthCalibrationInitialize",
    "WavelengthCalibrationMaster",
    "main",
    "wavelengths_from_traces",
]

logger = logging.getLogger(__name__)


def main(
    instrument,
    target,
    night=None,
    channels=None,
    steps="all",
    base_dir=None,
    input_dir=None,
    output_dir=None,
    configuration=None,
    trace_range=None,
    skip_existing=False,
    plot=0,
    plot_dir=None,
    use_groups=None,
):
    r"""
    Main entry point for REDUCE scripts.

    Default values can be changed as required if reduce is used as a script.
    Finds input directories, and loops over observation nights and instrument channels.

    .. deprecated::
        Use :meth:`Pipeline.from_instrument` instead.

    Parameters
    ----------
    instrument : str, list[str]
        instrument used for the observation (e.g. UVES, HARPS)
    target : str, list[str]
        the observed star, as named in the folder structure/fits headers
    night : str, list[str]
        the observation nights to reduce, as named in the folder structure. Accepts bash wildcards (i.e. \*, ?), but then relies on the folder structure for restricting the nights
    channels : str, list[str], dict[{instrument}:list], None, optional
        the instrument channels to use, if None will use all known channels for the current instrument. See instruments for possible options
    steps : tuple(str), "all", optional
        which steps of the reduction process to perform
        the possible steps are: "bias", "flat", "trace", "norm_flat", "wavecal", "science"
        alternatively set steps to "all", which is equivalent to setting all steps
        Note that the later steps require the previous intermediary products to exist and raise an exception otherwise
    base_dir : str, optional
        base data directory that Reduce should work in, is prefixxed on input_dir and output_dir (default: use settings_pyreduce.json)
    input_dir : str, optional
        input directory containing raw files. Can contain placeholders {instrument}, {target}, {night}, {channel} as well as wildcards. If relative will use base_dir as root (default: use settings_pyreduce.json)
    output_dir : str, optional
        output directory for intermediary and final results. Can contain placeholders {instrument}, {target}, {night}, {channel}, but no wildcards. If relative will use base_dir as root (default: use settings_pyreduce.json)
    configuration : dict[str:obj], str, list[str], dict[{instrument}:dict,str], optional
        configuration file for the current run, contains parameters for different parts of reduce. Can be a path to a json file, or a dict with configurations for the different instruments. When a list, the order must be the same as instruments (default: settings_{instrument.upper()}.json)
    """
    warnings.warn(
        "pyreduce.reduce.main() is deprecated. Use Pipeline.from_instrument() instead:\n"
        "    from pyreduce.pipeline import Pipeline\n"
        "    result = Pipeline.from_instrument(instrument, target, ...).run()",
        DeprecationWarning,
        stacklevel=2,
    )

    if target is None or np.isscalar(target):
        target = [target]
    if night is None or np.isscalar(night):
        night = [night]

    output = []

    # Loop over everything

    # settings: default settings of PyReduce
    # config: paramters for the current reduction
    # info: constant, instrument specific parameters
    config = load_config(configuration, instrument, 0)

    # Environment variable overrides for plot (useful for headless runs)
    if "PYREDUCE_PLOT" in os.environ:
        plot = int(os.environ["PYREDUCE_PLOT"])
    if "PYREDUCE_PLOT_DIR" in os.environ:
        plot_dir = os.environ["PYREDUCE_PLOT_DIR"]
    plot_show = os.environ.get("PYREDUCE_PLOT_SHOW", "block")

    # Set global plot settings for util.show_or_save()
    util.set_plot_dir(plot_dir)
    util.set_plot_show(plot_show, plot_level=plot)

    if isinstance(instrument, str):
        instrument = instruments.instrument_info.load_instrument(instrument)
    info = instrument.info

    if use_groups is not None:
        fibers = getattr(instrument.config, "fibers", None)
        if fibers is not None:
            fibers.use = {"default": use_groups}
        else:
            logger.warning("--use ignored: instrument has no fiber config")

    # load default settings from settings_pyreduce.json
    # $REDUCE_DATA overrides config for base_dir (but "" means use relative paths)
    if base_dir is None:
        base_dir = os.environ.get("REDUCE_DATA") or config["reduce"]["base_dir"]
    if input_dir is None:
        input_dir = config["reduce"]["input_dir"]
    if output_dir is None:
        output_dir = config["reduce"]["output_dir"]

    # Validate base_dir exists (skip if empty, allows absolute input/output paths)
    if base_dir and not os.path.isdir(base_dir):
        source = "$REDUCE_DATA" if os.environ.get("REDUCE_DATA") else "config"
        raise FileNotFoundError(
            f"Base directory does not exist: {base_dir} (from {source})"
        )

    input_dir = join(base_dir, input_dir)
    output_dir = join(base_dir, output_dir)

    # Validate input_dir exists
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}\n"
            f"  base_dir={base_dir}, input_dir setting={config['reduce']['input_dir']}"
        )

    if channels is None:
        channels = info.get("channels") or instrument.discover_channels(input_dir)
    if np.isscalar(channels):
        channels = [channels]
    for c in channels:
        instrument.validate_channel(c)

    for t, n, c in product(target, night, channels):
        log_file = join(
            base_dir.format(instrument=str(instrument), channel=channels, target=t),
            f"logs/{t}.log",
        )
        util.start_logging(log_file)
        # find input files and sort them by type
        steps_list = list(steps) if steps != "all" else None
        files = instrument.sort_files(
            input_dir,
            t,
            n,
            channel=c,
            steps=steps_list,
            **config["instrument"],
        )
        if len(files) == 0:
            logger.warning(
                "No files found for instrument: %s, target: %s, night: %s, channel: %s in folder: %s",
                instrument,
                t,
                n,
                c,
                input_dir,
            )
            continue
        for k, f in files:
            logger.info("Settings:")
            for key, value in k.items():
                logger.info("%s: %s", key, value)
            logger.debug("Files:\n%s", f)

            from .pipeline import Pipeline

            pipe = Pipeline.from_files(
                files=f,
                output_dir=output_dir,
                target=k.get("target"),
                instrument=instrument,
                channel=c,
                night=k.get("night") or "",
                config=config,
                trace_range=trace_range,
                steps=steps,
                plot=plot,
                plot_dir=plot_dir,
            )
            try:
                data = pipe.run(skip_existing=skip_existing)
                output.append(data)
            except ValueError as e:
                if "does not contain data for this channel" in str(e):
                    logger.warning("Skipping channel %s: %s", c, e)
                    continue
                raise
    return output
