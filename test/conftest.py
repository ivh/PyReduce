import os
import tempfile
from os.path import dirname, join
from shutil import rmtree

# Stop matplotlib from crashing if interactive plotting does not work
import matplotlib as mpl
import numpy as np
import pytest

mpl.use("agg")

from pyreduce import configuration, datasets, instruments
from pyreduce.reduce import (
    BackgroundScatter,
    Bias,
    Flat,
    Mask,
    NormalizeFlatField,
    ScienceExtraction,
    SlitCurvatureDetermination,
    Trace,
    WavelengthCalibrationFinalize,
    WavelengthCalibrationInitialize,
    WavelengthCalibrationMaster,
)


def pytest_addoption(parser):
    """Add custom command-line options for test configuration"""
    parser.addoption(
        "--instrument",
        action="store",
        default=None,
        help="Test specific instrument (e.g., UVES, XSHOOTER, NIRSPEC, JWST_NIRISS). If not specified, runs all instruments.",
    )
    parser.addoption(
        "--target",
        action="store",
        default=None,
        help="Test specific target (e.g., HD132205, UX-Ori, GJ1214). Requires --instrument.",
    )


@pytest.fixture(scope="function")
def tempfiles():
    n = 10
    files = [tempfile.NamedTemporaryFile(delete=False) for _ in range(n)]
    files = [f.name for f in files]
    yield files

    for f in files:
        try:
            os.remove(f)
        except:
            pass


# TODO Add more datasets
# Default datasets for parametrized testing
_DEFAULT_DATASETS = [
    ("UVES", "HD[ -]?132205"),
    ("XSHOOTER", "UX-Ori"),
    ("NIRSPEC", "GJ1214"),
    ("JWST_NIRISS", ""),
]
_DEFAULT_IDS = ["UVES_HD132205", "XSHOOTER_UxOri", "NIRSPEC_GJ1214", "JWST_NIRISS"]


@pytest.fixture
def dataset(request):
    """Get dataset from CLI args or use parametrized defaults

    If --instrument is specified on command line, uses that single instrument/target.
    Otherwise, runs parametrized tests across all default datasets.
    """
    # Check for command-line overrides
    instrument_arg = request.config.getoption("--instrument")
    target_arg = request.config.getoption("--target")

    if instrument_arg is not None:
        # Use CLI-specified instrument/target
        # Map instrument to default target if not specified
        target_defaults = {
            "UVES": "HD[ -]?132205",
            "XSHOOTER": "UX-Ori",
            "NIRSPEC": "GJ1214",
            "JWST_NIRISS": "",
        }
        target = (
            target_arg
            if target_arg is not None
            else target_defaults.get(instrument_arg, "")
        )
        return (instrument_arg, target)

    # Use parametrized dataset (for full test matrix)
    # This requires the fixture to be parametrized when called without CLI args
    if hasattr(request, "param"):
        return request.param

    # Fallback to first dataset if neither CLI nor parametrization provided
    return _DEFAULT_DATASETS[0]


# Parametrize the dataset fixture for tests that don't use CLI args
def pytest_generate_tests(metafunc):
    """Parametrize dataset fixture unless overridden by CLI"""
    if "dataset" in metafunc.fixturenames:
        # Only parametrize if --instrument was not specified
        if metafunc.config.getoption("--instrument") is None:
            metafunc.parametrize(
                "dataset", _DEFAULT_DATASETS, ids=_DEFAULT_IDS, indirect=True
            )
        # Otherwise, dataset fixture will use CLI args via request.config


@pytest.fixture
def instrument(dataset):
    """Observing instrument of a given dataset

    Parameters
    ----------
    dataset : tuple(str, str)
        instrument, target

    Returns
    -------
    instrument : str
        Observing instrument
    """
    return dataset[0]


@pytest.fixture
def instr(instrument):
    return instruments.instrument_info.load_instrument(instrument)


@pytest.fixture
def target(dataset):
    """Observation target of a given dataset

    Parameters
    ----------
    dataset : tuple(str, str)
        instrument, target

    Returns
    -------
    target : str
        Observation target
    """

    return dataset[1]


@pytest.fixture
def night(dataset):
    """The observation night of each dataset

    Parameters
    ----------
    dataset : tuple(str, str)
        (instrument, target)

    Returns
    -------
    night : str
        Observation Night
    """

    _, target = dataset
    if target == "HD132205":
        return "2010-04-01"
    if target == "UX-Ori":
        return "2009-10-04"
    return ""


@pytest.fixture
def channel(instrument, info):
    """
    Instrument channel (detector/channel)

    Parameters
    ----------
    dataset : tuple(str, str)
        instrument, target

    Returns
    -------
    channel : str
        Instrument channel
    """

    if instrument == "UVES":
        return "MIDDLE"
    elif instrument == "XSHOOTER":
        return "NIR"

    channels = info["channels"]
    if isinstance(channels, list):
        return channels[0]
    return channels


@pytest.fixture
def info(instr):
    """
    Static instrument information

    Parameters
    ----------
    instrument : str
        Instrument Name

    Returns
    -------
    info : dict(str:obj)
        Instrument information
    """

    return instr.info


@pytest.fixture
def trace_range(instrument):
    if instrument == "JWST_NIRISS":
        return (0, 2)
    else:
        return (3, 5)


@pytest.fixture
def data(dataset, settings, target, night, channel):
    """
    Load dataset data from the web if necessary, and return data folder

    #TODO: remove data afterwards?

    Parameters
    ----------
    dataset : tuple(str, str)
        instrument, target

    Returns
    -------
    directory : str
        data directory
    """

    instrument, target = dataset
    folder = join(dirname(__file__), "datasets")
    if instrument == "UVES":
        folder = datasets.UVES(folder)
    elif instrument == "XSHOOTER":
        folder = datasets.XSHOOTER(folder)
    elif instrument == "JWST_NIRISS":
        folder = datasets.JWST_NIRISS(folder)
    elif instrument == "NIRSPEC":
        folder = datasets.KECK_NIRSPEC(folder)
    else:
        raise ValueError("Dataset not recognised")
    yield folder

    odir = _odir(folder, settings, instrument, target, night, channel)
    rmtree(odir, ignore_errors=True)


@pytest.fixture
def settings(instrument):
    """Combine run specific configuration with default settings

    Parameters
    ----------
    config : dict(str:obj)
        run specific settings

    Returns
    -------
    settings : dict(str:obj)
        updated settings
    """

    settings = configuration.get_configuration_for_instrument(
        instrument, plot=False, manual=False
    )
    return settings


@pytest.fixture
def input_dir(data, target, instrument, settings, night, channel):
    """Input data directory

    Parameters
    ----------
    data : str
        data directory
    target : str
        observation target name

    Returns
    -------
    input_dir : str
        Input data directory
    """

    odir = settings["reduce"]["input_dir"]
    odir = odir.format(
        instrument=instrument, target=target, night=night, channel=channel
    )
    return join(data, odir)


def _odir(data, settings, instrument, target, night, channel):
    odir = settings["reduce"]["output_dir"]
    odir = odir.format(
        instrument=instrument, target=target, night=night, channel=channel
    )
    odir = join(data, odir)
    return odir


@pytest.fixture
def output_dir(data, settings, instrument, target, night, channel):
    """Output data directory
    Also creates that directory if necessary

    Parameters
    ----------
    data : str
        data directory
    settings : dict(str:obj)
        run settings
    instrument : str
        instrument name
    target : str
        observation target
    night : str
        observation night
    channel : str
        instrument channel

    Returns
    -------
    output_dir : str
        output directory
    """
    odir = _odir(data, settings, instrument, target, night, channel)
    os.makedirs(odir, exist_ok=True)
    return odir


@pytest.fixture
def files(input_dir, instrument, target, night, channel, settings, instr):
    """Find and sort all files for this dataset

    Parameters
    ----------
    input_dir : str
        input data directory
    instrument : str
        instrument name
    target : str
        observation target name
    night : str
        observing night
    channel : str
        instrument channel
    settings : dict(str:obj)
        run settings

    Returns
    -------
    files : dict(str:str)
        filenames sorted by usecase (e.g. wavelength calibration files)
    """

    print(input_dir, target, night, instrument, channel, *settings["instrument"])
    files = instr.sort_files(
        input_dir, target, night, channel, **settings["instrument"]
    )
    files = files[0][1]
    return files


@pytest.fixture
def prefix(instrument, channel):
    """Prefix for the output files

    Parameters
    ----------
    instrument : str
        instrument name
    channel : str
        instrument channel

    Returns
    -------
    prefix : str
        instrument_arm
    """

    prefix = f"{instrument.lower()}_{channel.lower()}"
    return prefix


@pytest.fixture
def step_args(instr, channel, target, night, output_dir, trace_range):
    return instr, channel, target, night, output_dir, trace_range


@pytest.fixture
def mask(step_args, settings):
    """Load the bad pixel mask for this instrument/channel

    Parameters
    ----------
    instrument : str
        instrument name
    channel : str
        instrument channel

    Returns
    -------
    mask : array(bool) of size (ncol, nrow)
        Bad pixel mask
    """

    name = "mask"
    settings = settings[name]
    settings["plot"] = False

    step = Mask(*step_args, **settings)

    try:
        mask = step.load()
    except FileNotFoundError:
        mask = step.run()
    return mask


@pytest.fixture
def bias(step_args, settings, files, mask):
    """Load or if necessary create the bias calibration

    Parameters
    ----------
    instrument : str
        instrument name
    channel : str
        instrument channel
    files : dict(str:str)
        calibration files
    extension : int
        fits extension
    mask : array(bool)
        Bad pixel mask
    output_dir : str
        directory conatining the bias data

    Returns
    -------
    bias : array(float)
        bias calibration data
    bhead : fits.header
        bias information
    """

    name = "bias"
    files = files[name]
    settings = settings[name]
    settings["plot"] = False

    step = Bias(*step_args, **settings)

    bias = step.load(mask)
    if bias[0] is None:
        try:
            bias = step.run(files, mask=mask)
        except FileNotFoundError:
            # No input data for this instrument
            bias = (None, None)

    return bias


@pytest.fixture
def flat(step_args, settings, files, bias, mask):
    """Load or if necessary create the flat field calibration data"""

    name = "flat"
    files = files[name]
    settings = settings[name]
    settings["plot"] = False

    step = Flat(*step_args, **settings)

    flat = step.load(mask)
    if flat[0] is None:
        try:
            flat = step.run(files, bias=bias, mask=mask)
        except FileNotFoundError:
            flat = (None, None)

    return flat


@pytest.fixture
def traces(step_args, settings, files, mask, bias):
    """Load or calculate the order traces.

    Returns list[TraceData] - the new standard format for traces.
    """
    name = "trace"
    settings = settings[name]
    files = files[name]
    settings["plot"] = False
    settings["manual"] = False

    step = Trace(*step_args, **settings)

    try:
        trace_list = step.load()
    except FileNotFoundError:
        trace_list = step.run(files, mask=mask, bias=bias)
    return trace_list


@pytest.fixture
def orders(traces):
    """Legacy fixture: returns (orders_arr, column_range) tuple from traces.

    Use 'traces' fixture directly for new code.
    """
    # Build arrays from Trace objects for legacy test compatibility
    orders_arr = np.array([t.pos for t in traces])
    column_range = np.array([t.column_range for t in traces])
    return orders_arr, column_range


@pytest.fixture
def scatter(step_args, settings, files, mask, bias, traces):
    name = "scatter"
    settings = settings[name]
    settings["plot"] = False
    files = files[name]

    step = BackgroundScatter(*step_args, **settings)

    scatter = step.load()
    if scatter is None:
        try:
            scatter = step.run(files, traces, mask=mask, bias=bias)
        except FileNotFoundError:
            scatter = None
    return scatter


@pytest.fixture
def normflat(step_args, settings, flat, traces, scatter):
    """Load or create the normalized flat field

    Parameters
    ----------
    flat : array(float)
        flat field calibration data and header
    traces : list[TraceData]
        Trace objects from trace step (curvature is embedded in traces)
    settings : dict(str:obj)
        run settings
    output_dir : str
        output data folder

    Returns
    -------
    norm : array(float) of size (nrow, ncol)
        normalized flat field data
    blaze : array(float) of size (norders, ncol)
        extracted blaze for each order
    """

    name = "norm_flat"
    settings = settings[name]
    settings["plot"] = False

    step = NormalizeFlatField(*step_args, **settings)

    norm, blaze, *_ = step.load()

    if norm is None:
        try:
            norm, blaze, *_ = step.run(flat, traces, scatter=scatter)
        except FileNotFoundError:
            norm, blaze = None, None
    return norm, blaze


@pytest.fixture
def curvature(step_args, settings, files, traces, mask):
    name = "curvature"
    files = files[name]
    settings = settings[name]
    settings["plot"] = False

    step = SlitCurvatureDetermination(*step_args, **settings)

    try:
        curv = step.load()
    except FileNotFoundError:
        curv = step.run(files, traces, mask=mask)
    return curv


@pytest.fixture
def wave_master(step_args, settings, files, traces, mask, bias, normflat):
    """Load or create wavelength calibration files

    Parameters
    ----------
    files : dict(str:str)
        calibration file names
    traces : list[TraceData]
        Trace objects from trace step (curvature is embedded in traces)
    mask : array(bool)
        Bad pixel mask
    settings : dict(str:obj)
        run settings
    bias : tuple
        Master bias
    normflat : tuple
        Normalized flat field

    Returns
    -------
    wavecal : array(float) of size (norder, ncol)
        Extracted wavelength calibration spectrum
    whead : FITS header
        Header of wavelength calibration image
    """
    name = "wavecal_master"
    files = files[name]
    settings[name]["plot"] = False

    step = WavelengthCalibrationMaster(*step_args, **settings[name])

    try:
        results = step.load()
    except FileNotFoundError:
        try:
            results = step.run(
                files,
                traces,
                mask=mask,
                bias=bias,
                norm_flat=normflat,
            )
        except FileNotFoundError:
            return None, None
    # Extract first group's data for backwards compatibility
    first_group = next(iter(results))
    wavecal, whead = results[first_group]
    return wavecal, whead


@pytest.fixture
def wave_init(step_args, settings, wave_master):
    name = "wavecal_init"
    settings[name]["plot"] = False

    step = WavelengthCalibrationInitialize(*step_args, **settings[name])

    try:
        linelist = step.load(settings, wave_master)
    except:
        linelist = None
    return linelist


@pytest.fixture
def wlen(step_args, settings, wave_master, wave_init):
    """Load or create wavelength calibration files

    Returns
    -------
    wlen : array(float) of size (norder, ncol)
        Wavelength along the spectral orders
    """
    name = "wavecal"
    settings[name]["plot"] = False

    step = WavelengthCalibrationFinalize(*step_args, **settings[name])

    try:
        wlen, wave, linelist = step.load()
    except FileNotFoundError:
        try:
            wlen, wave, linelist = step.run(wave_master, wave_init)
        except Exception:
            wlen = None
    return wlen


@pytest.fixture
def wave(wlen):
    """Alias for wlen fixture for test_continuum.py compatibility."""
    return wlen


@pytest.fixture
def spec(step_args, settings, files, bias, traces, normflat, scatter, mask):
    """Load or create science spectrum

    Returns
    -------
    spec : array(float) of size (norders, ncol)
        extracted science spectra
    sigma : array(float) of size (norders, ncol)
        uncertainty on the extracted science spectra
    """
    name = "science"
    settings = settings[name]
    settings["plot"] = False

    step = ScienceExtraction(*step_args, **settings)
    step.files = files

    try:
        heads, specs, sigmas, slitfus, column_ranges = step.load()
        return specs[0], sigmas[0]
    except FileNotFoundError:
        files_list = files[name][:1]
        heads, all_spectra = step.run(
            files_list,
            traces,
            bias=bias,
            norm_flat=normflat,
            scatter=scatter,
            mask=mask,
        )
        # Convert list[Spectrum] to arrays
        import numpy as np

        spectra = all_spectra[0]
        specs = np.array([s.spec for s in spectra])
        sigmas = np.array([s.sig for s in spectra])
        return specs, sigmas
