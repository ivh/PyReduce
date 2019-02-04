import os
from os.path import dirname, join

import numpy as np
from scipy.io import readsav
from astropy.io import fits
import tempfile
import pickle

import json
import pytest
from pyreduce import datasets, util, instruments
from pyreduce.combine_frames import combine_bias, combine_flat
from pyreduce.trace_orders import mark_orders
from pyreduce.normalize_flat import normalize_flat
from pyreduce.extract import extract
from pyreduce.wavelength_calibration import wavecal
from pyreduce import echelle


@pytest.fixture(scope="function")
def tempfiles():
    n = 10
    files = [tempfile.NamedTemporaryFile(delete=False) for _ in range(n)]
    files = [f.name for f in files]
    yield files

    for f in files:
        os.remove(f)


# TODO Add more datasets
@pytest.fixture(params=[("UVES", "HD132205")], ids=["UVES_HD132205"])
def dataset(request):
    return request.param


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
        return "2010-04-02"


@pytest.fixture
def mode(dataset):
    """
    Observation mode of the instrument

    Parameters
    ----------
    dataset : tuple(str, str)
        instrument, target

    Returns
    -------
    mode : str
        Observation mode
    """

    instrument, target = dataset
    if instrument == "UVES":
        return "middle"


@pytest.fixture
def info(instrument):
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

    i = instruments.instrument_info.get_instrument_info(instrument)
    return i


@pytest.fixture
def extension(info, mode):
    """Fits Extension to use

    Parameters
    ----------
    info : dict(str:obj)
        instrument information
    mode : str
        observing mode

    Returns
    -------
    extension : int
        fits extension
    """

    imode = util.find_first_index(info["modes"], mode)
    ext = info["extension"][imode]
    return ext


@pytest.fixture
def order_range(dataset):
    return (0, 1)


@pytest.fixture
def data(dataset):
    """
    Load dataset data from the web if necessary, and return data folder

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
    folder = dirname(__file__)
    if instrument == "UVES" and target == "HD132205":
        folder = datasets.UVES_HD132205(folder)
    return folder


@pytest.fixture
def config(dataset):
    """
    Configuration settings for this run of PyReduce

    Parameters
    ----------
    dataset : tuple(str, str)
        instrument, target

    Returns
    -------
    config : dict(str:obj)
        run specific configuration
    """

    instrument, target = dataset
    folder = dirname(__file__)
    filename = join(folder, "settings", f"settings_{instrument.upper()}.json")

    with open(filename) as f:
        conf = json.load(f)

    return conf


@pytest.fixture
def settings(config):
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

    setti = util.read_config()
    setti.update(config)
    return setti


@pytest.fixture
def input_dir(data, target):
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

    return join(data, target)


@pytest.fixture
def output_dir(data, settings, instrument, target, night, mode):
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
    mode : str
        observing mode
    
    Returns
    -------
    output_dir : str
        output directory
    """

    odir = settings["reduce.output_dir"]
    odir = odir.format(instrument=instrument, target=target, night=night, mode=mode)
    odir = join(data, "reduced", odir)

    os.makedirs(odir, exist_ok=True)

    return odir


@pytest.fixture
def files(input_dir, instrument, target, night, mode, settings):
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
    mode : str
        observing mode
    settings : dict(str:obj)
        run settings

    Returns
    -------
    files : dict(str:str)
        filenames sorted by usecase (e.g. wavelength calibration files)
    """

    files, _ = instruments.instrument_info.sort_files(
        input_dir, target, night, instrument, mode, **settings
    )
    files = files[0][list(files[0].keys())[0]]
    return files


@pytest.fixture
def mask(instrument, mode):
    """Load the bad pixel mask for this instrument/mode

    Parameters
    ----------
    instrument : str
        instrument name
    mode : str
        observing mode

    Returns
    -------
    mask : array(bool) of size (ncol, nrow)
        Bad pixel mask
    """

    mask_dir = os.path.dirname(__file__)
    mask_dir = os.path.join(mask_dir, "../pyreduce", "masks")
    mask_file = join(mask_dir, "mask_%s_%s.fits.gz" % (instrument.lower(), mode))

    mask, _ = util.load_fits(mask_file, instrument, mode, extension=0)
    mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks
    return mask


@pytest.fixture
def prefix(instrument, mode):
    """Prefix for the output files

    Parameters
    ----------
    instrument : str
        instrument name
    mode : str
        observing mode

    Returns
    -------
    prefix : str
        instrument_mode
    """

    prefix = "%s_%s" % (instrument.lower(), mode.lower())
    return prefix


@pytest.fixture
def bias(instrument, mode, files, extension, mask, output_dir):
    """Load or if necessary create the bias calibration

    Parameters
    ----------
    instrument : str
        instrument name
    mode : str
        observing mode
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

    biasfile = os.path.join(output_dir, "test_bias.fits")
    try:
        bias = fits.open(biasfile)
        bhead, bias = bias[0].header, bias[0].data
        bias = np.ma.masked_array(bias, mask=mask)
    except FileNotFoundError:
        bias, bhead = combine_bias(
            files["bias"], instrument, mode, extension=extension, window=50, mask=mask
        )
        fits.writeto(biasfile, data=bias.data, header=bhead, overwrite=True)

    return bias, bhead


@pytest.fixture
def flat(instrument, mode, files, extension, mask, bias, output_dir):
    """Load or if necessary create the flat field calibration data

    Parameters
    ----------
    instrument : str
        instrument name
    mode : str
        observing mode
    files : dict(str:str)
        calibration file names
    extension : int
        fits data extension
    mask : array(bool)
        Bad pixel mask
    bias : array(float)
        bias calibration data
    output_dir : str
        output directory

    Returns
    -------
    flat : array(float)
        flat field calibration data
    fhead : fits.header
        flat information
    """

    flatfile = os.path.join(output_dir, "test_flat.fits")
    try:
        flat = fits.open(flatfile)
        fhead, flat = flat[0].header, flat[0].data
        flat = np.ma.masked_array(flat, mask=mask)
    except FileNotFoundError:
        bias, _ = bias
        flat, fhead = combine_flat(
            files["flat"],
            instrument,
            mode,
            extension=extension,
            bias=bias,
            window=50,
            mask=mask,
        )
        fits.writeto(flatfile, data=flat.data, header=fhead, overwrite=True)

    return flat, fhead


@pytest.fixture
def orders(instrument, mode, extension, files, settings, mask, output_dir):
    """Load or if necessary calculate the order traces

    Parameters
    ----------
    instrument : str
        instrument name
    mode : str
        observing mode
    extension : int
        fits data extension
    files : dict(str:str)
        calibration data files
    settings : dict(str:obj)
        settings for this run
    mask : array(bool)
        Bad pixel map
    output_dir : str
        output file directory

    Returns
    -------
    orders : array(float) of size (norders, ndegree+1)
        polynomial coefficients of the order tracing
    column_range : array(int) of size (norders, 2)
        valid columns that include traces/data
    """

    orderfile = os.path.join(output_dir, "test_orders.pkl")
    try:
        with open(orderfile, "rb") as file:
            orders, column_range = pickle.load(file)
    except FileNotFoundError:
        files = files["order"][0]
        order_img, _ = util.load_fits(files, instrument, mode, extension, mask=mask)

        orders, column_range = mark_orders(
            order_img,
            min_cluster=settings["orders.min_cluster"],
            filter_size=settings["orders.filter_size"],
            noise=settings["orders.noise"],
            opower=settings["orders.fit_degree"],
            border_width=settings["orders.border_width"],
            manual=False,
            plot=False,
        )
        with open(orderfile, "wb") as file:
            pickle.dump((orders, column_range), file)
    return orders, column_range


@pytest.fixture
def normflat(flat, orders, settings, output_dir, mask, order_range):
    """Load or create the normalized flat field

    Parameters
    ----------
    flat : array(float)
        flat field calibration data and header
    orders : tuple(array, array)
        order polynomials, and column ranges
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

    normflatfile = os.path.join(output_dir, "test_normflat.fits")
    blazefile = os.path.join(output_dir, "test_blaze.pkl")

    if os.path.exists(normflatfile) and os.path.exists(blazefile):
        norm = fits.open(normflatfile)[0]
        norm, fhead = norm.data, norm.header
        norm = np.ma.masked_array(norm, mask=mask)

        with open(blazefile, "rb") as file:
            blaze = pickle.load(file)
    else:
        flat, fhead = flat
        orders, column_range = orders

        norm, blaze = normalize_flat(
            flat,
            orders,
            gain=fhead["e_gain"],
            readnoise=fhead["e_readn"],
            dark=fhead["e_drk"],
            column_range=column_range,
            order_range=order_range,
            extraction_width=settings["normflat.extraction_width"],
            degree=settings["normflat.scatter_degree"],
            threshold=settings["normflat.threshold"],
            lambda_sf=settings["normflat.smooth_slitfunction"],
            lambda_sp=settings["normflat.smooth_spectrum"],
            swath_width=settings["normflat.swath_width"],
            plot=False,
        )
        with open(blazefile, "wb") as file:
            pickle.dump(blaze, file)
        fits.writeto(normflatfile, data=norm.data, header=fhead, overwrite=True)

    return norm, blaze


@pytest.fixture
def wave(files, instrument, mode, extension, mask, orders, settings, output_dir):
    """Load or create wavelength calibration files

    Parameters
    ----------
    files : dict(str:str)
        calibration file names
    instrument : str
        instrument name
    mode : str
        observing mode
    extension : int
        fits data extension
    mask : array(bool)
        Bad pixel mask
    orders : tuple(array, array)
        order tracing polynomials and column ranges
    settings : dict(str:obj)
        run settings
    output_dir : str
        output data directory

    Returns
    -------
    wave : array(float) of size (norder, ncol)
        Wavelength along the spectral orders
    """
    orders, column_range = orders
    wavefile = os.path.join(output_dir, "test_wavecal.thar.ech")

    if os.path.exists(wavefile):
        thar = echelle.read(wavefile, raw=True)
        wave = thar.wave
        thar = thar.spec

        mask = np.full(wave.shape, True)
        for iord in range(wave.shape[0]):
            cr = column_range[iord]
            mask[iord, cr[0] : cr[1]] = False

        wave = np.ma.array(wave, mask=mask)
    else:
        files = files["wave"][0]
        orig, thead = util.load_fits(files, instrument, mode, extension, mask=mask)
        thead["obase"] = (0, "base order number")

        # Extract wavecal spectrum
        thar, _ = extract(
            orig,
            orders,
            gain=thead["e_gain"],
            readnoise=thead["e_readn"],
            dark=thead["e_drk"],
            extraction_type="arc",
            column_range=column_range,
            extraction_width=settings["wavecal.extraction_width"],
            osample=settings["wavecal.oversampling"],
            plot=False,
        )

        reference = instruments.instrument_info.get_wavecal_filename(
            thead, instrument, mode
        )
        reference = readsav(reference)
        cs_lines = reference["cs_lines"]
        wave = wavecal(thar, cs_lines, plot=False, manual=False)

        echelle.save(wavefile, thead, spec=thar, wave=wave)

    return wave


@pytest.fixture
def spec(
    files,
    instrument,
    mode,
    mask,
    extension,
    bias,
    normflat,
    orders,
    settings,
    output_dir,
    order_range,
):
    """Load or create science spectrum

    Parameters
    ----------
    files : dict(str:str)
        raw input files
    instrument : str
        instrument name
    mode : str
        observing mode
    mask : array(bool)
        Bad pixel mask
    extension : int
        fits data extension
    bias : array(float)
        bias calibration data
    normflat : array(float)
        normalized flat field
    orders : array(float)
        order tracing polynomials and column_ranges
    settings : dict(str:obj)
        run settings
    output_dir : str
        output data directory

    Returns
    -------
    spec : array(float) of size (norders, ncol)
        extracted science spectra
    sigma : array(float) of size (norders, ncol)
        uncertainty on the extracted science spectra
    """
    orders, column_range = orders
    specfile = os.path.join(output_dir, "test_spec.ech")

    try:
        science = echelle.read(specfile, raw=True)
        head = science.head
        spec = science.spec
        sigma = science.sig

        mask = np.full(spec.shape, True)
        for iord in range(spec.shape[0]):
            cr = column_range[iord]
            mask[iord, cr[0] : cr[1]] = False
        spec = np.ma.array(spec, mask=mask)
        sigma = np.ma.array(sigma, mask=mask)
    except FileNotFoundError:
        flat, blaze = normflat
        bias, _ = bias

        # Fix column ranges
        for i in range(blaze.shape[0]):
            column_range[i] = np.where(blaze[i] != 0)[0][[0, -1]]

        f = files["spec"][0]

        im, head = util.load_fits(
            f, instrument, mode, extension, mask=mask, dtype=np.float32
        )
        # Correct for bias and flat field
        im -= bias
        im /= flat

        # Optimally extract science spectrum
        spec, sigma = extract(
            im,
            orders,
            gain=head["e_gain"],
            readnoise=head["e_readn"],
            dark=head["e_drk"],
            column_range=column_range,
            order_range=order_range,
            extraction_width=settings["science.extraction_width"],
            lambda_sf=settings["science.smooth_slitfunction"],
            lambda_sp=settings["science.smooth_spectrum"],
            osample=settings["science.oversampling"],
            swath_width=settings["science.swath_width"],
            plot=False,
        )
        echelle.save(specfile, head, spec=spec, sig=sigma)

    return spec, sigma
