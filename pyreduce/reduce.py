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

import joblib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning

warnings.simplefilter("ignore", category=VerifyWarning, append=True)
warnings.simplefilter("ignore", category=AstropyUserWarning, append=True)


from tqdm import tqdm

# PyReduce subpackages
from . import __version__, instruments, util
from .combine_frames import (
    combine_bias,
    combine_calibrate,
    combine_polynomial,
)
from .configuration import load_config
from .continuum_normalization import continuum_normalize, splice_orders
from .estimate_background_scatter import estimate_background_scatter
from .extract import extract, extract_normalize
from .rectify import merge_images, rectify_image
from .slit_curve import Curvature as CurvatureModule
from .spectra import ExtractionParams, Spectra, Spectrum
from .trace import (
    group_fibers,
    select_traces_for_step,
)
from .trace import trace as mark_orders
from .trace_model import (
    Trace as TraceData,
)
from .trace_model import (
    load_traces,
    save_traces,
)
from .wavelength_calibration import LineList, WavelengthCalibrationComb
from .wavelength_calibration import WavelengthCalibration as WavelengthCalibrationModule
from .wavelength_calibration import (
    WavelengthCalibrationInitialize as WavelengthCalibrationInitializeModule,
)

# TODO Naming of functions and modules
# TODO License

# TODO automatic determination of the extraction width
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


def wavelengths_from_traces(traces: list, ncol: int = None) -> np.ndarray:
    """Compute wavelength array from trace objects.

    Parameters
    ----------
    traces : list[TraceData]
        Trace objects with .wave polynomial coefficients set
    ncol : int, optional
        Number of columns. If not provided, uses max column_range.

    Returns
    -------
    wlen : ndarray of shape (ntrace, ncol)
        Wavelength for each pixel, or None if no wavelength data
    """
    if not traces:
        return None

    # Check if any trace has wavelength data
    if not any(t.wave is not None for t in traces):
        return None

    # Determine ncol from traces if not provided
    if ncol is None:
        max_col = max(t.column_range[1] for t in traces)
        ncol = int(2 ** np.ceil(np.log2(max_col)))  # Round up to power of 2

    x = np.arange(ncol)
    wlen = np.array(
        [t.wlen(x) if t.wave is not None else np.full(ncol, np.nan) for t in traces]
    )
    return wlen


class Step:
    """Parent class for all steps"""

    def __init__(
        self, instrument, channel, target, night, output_dir, trace_range, **config
    ):
        self._dependsOn = []
        self._loadDependsOn = []
        #:dict: Input files dict, set by pipeline before load()
        self.files = None
        #:str: Name of the instrument
        self.instrument = instrument
        #:str: Name of the instrument channel
        self.channel = channel
        #:str: Name of the observation target
        self.target = target
        #:str: Date of the observation (as a string)
        self.night = night
        #:tuple(int, int): First and Last(+1) trace to process
        self.trace_range = trace_range
        #:bool: Whether to plot the results or the progress of this step
        self.plot = config.get("plot", False)
        #:str: Title used in the plots, if any
        self.plot_title = config.get("plot_title", None)
        self._output_dir = output_dir

    def run(self, files, *args):  # pragma: no cover
        """Execute the current step

        This should fail if files are missing or anything else goes wrong.
        If the user does not want to run this step, they should not specify it in steps.

        Parameters
        ----------
        files : list(str)
            data files required for this step

        Raises
        ------
        NotImplementedError
            needs to be implemented for each step
        """
        raise NotImplementedError

    def save(self, *args):  # pragma: no cover
        """Save the results of this step

        Parameters
        ----------
        *args : obj
            things to save

        Raises
        ------
        NotImplementedError
            Needs to be implemented for each step
        """
        raise NotImplementedError

    def load(self):  # pragma: no cover
        """Load results from a previous execution

        If this raises a FileNotFoundError, run() will be used instead
        For calibration steps it is preferred however to print a warning
        and return None. Other modules can then use a default value instead.

        Raises
        ------
        NotImplementedError
            Needs to be implemented for each step
        """
        raise NotImplementedError

    @property
    def dependsOn(self):
        """list(str): Steps that are required before running this step"""
        return list(set(self._dependsOn))

    @property
    def loadDependsOn(self):
        """list(str): Steps that are required before loading data from this step"""
        return list(set(self._loadDependsOn))

    @property
    def output_dir(self):
        """str: output directory, may contain tags {instrument}, {night}, {target}, {channel}"""
        return self._output_dir.format(
            instrument=self.instrument.name.upper(),
            target=self.target,
            night=self.night,
            channel=self.channel,
        )

    @property
    def prefix(self):
        """str: temporary file prefix"""
        i = self.instrument.name.lower()
        if self.channel is not None and self.channel != "":
            c = self.channel.lower()
            return f"{i}_{c}"
        else:
            return i

    def _select_traces(
        self, trace_objects: list[TraceData], step_name: str
    ) -> dict[str, list[TraceData]]:
        """Apply fiber selection to traces based on instrument config.

        Parameters
        ----------
        trace_objects : list[TraceData]
            Trace objects from Tracing step
        step_name : str
            Name of this step for fibers.use lookup

        Returns
        -------
        selected : dict[str, list[TraceData]]
            {group_name: [traces]} for each selected group
        """
        fibers_config = getattr(self.instrument.config, "fibers", None)
        return select_traces_for_step(trace_objects, fibers_config, step_name)


class CalibrationStep(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["mask", "bias"]

        #:{'number_of_files', 'exposure_time', 'mean', 'median', 'none'}: how to adjust for diferences between the bias and flat field exposure times
        self.bias_scaling = config["bias_scaling"]
        #:{'divide', 'none'}: how to apply the normalized flat field
        self.norm_scaling = config["norm_scaling"]

    def calibrate(
        self,
        files,
        mask,
        bias=None,
        norm_flat=None,
        traces=None,
        extraction_height=None,
    ):
        bias, bhead = bias if bias is not None else (None, None)
        norm, blaze, *_ = norm_flat if norm_flat is not None else (None, None, None)
        orig, thead = combine_calibrate(
            files,
            self.instrument,
            self.channel,
            mask,
            bias=bias,
            bhead=bhead,
            norm=norm,
            bias_scaling=self.bias_scaling,
            norm_scaling=self.norm_scaling,
            plot=self.plot,
            plot_title=self.plot_title,
            traces=traces,
            extraction_height=extraction_height,
        )

        return orig, thead


class ExtractionStep(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += [
            "trace",
        ]

        #:{'simple', 'optimal'}: Extraction method to use
        self.extraction_method = config["extraction_method"]
        if self.extraction_method in (
            "simple",
            "arc",
        ):  # "arc" for backwards compatibility
            #:dict: arguments for the extraction
            self.extraction_kwargs = {
                "extraction_height": config["extraction_height"],
                "collapse_function": config["collapse_function"],
            }
        elif self.extraction_method == "optimal":
            self.extraction_kwargs = {
                "extraction_height": config["extraction_height"],
                "lambda_sf": config["smooth_slitfunction"],
                "lambda_sp": config["smooth_spectrum"],
                "osample": config["oversampling"],
                "swath_width": config["swath_width"],
                "maxiter": config["maxiter"],
                "reject_threshold": config.get("extraction_reject", 6),
            }
        else:
            raise ValueError(
                f"Extraction method {self.extraction_method} not supported for step 'wavecal'"
            )

    def extract_to_arrays(self, img, head, trace_list: list[TraceData], scatter=None):
        """Extract spectra and return as arrays (for wavecal compatibility)."""
        extraction_kwargs = dict(self.extraction_kwargs)
        default_height = extraction_kwargs.pop("extraction_height", 0.5)

        # Apply trace_range if specified
        if self.trace_range is not None:
            trace_list = trace_list[self.trace_range[0] : self.trace_range[1]]

        spectra = extract(
            img,
            trace_list,
            extraction_height=default_height,
            extraction_type=self.extraction_method,
            gain=head["e_gain"],
            readnoise=head["e_readn"],
            dark=head["e_drk"],
            scatter=scatter,
            plot=self.plot,
            plot_title=self.plot_title,
            **extraction_kwargs,
        )

        # Convert Spectrum objects back to arrays
        data = np.array([s.spec for s in spectra])
        unc = np.array([s.sig for s in spectra])
        slitfu = [s.slitfu for s in spectra]
        cr = np.array(
            [list(trace_list[i].column_range) for i in range(len(spectra))],
            dtype=np.int32,
        )

        return data, unc, slitfu, cr


class FitsIOStep(Step):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loadDependsOn += ["mask"]
        self.allow_failure = True

    def save(self, data, head, dtype=None):
        """
        Save the data to a FITS file

        Parameters
        ----------
        data : array of shape (nrow, ncol)
            bias data
        head : FITS header
            bias header
        """
        if dtype is not None:
            data = np.asarray(data, dtype=np.float32)

        fits.writeto(
            self.savefile,
            data=data,
            header=head,
            overwrite=True,
            output_verify="silentfix+ignore",
        )
        logger.info("Created data file: %s", self.savefile)

    def load(self, mask):
        """
        Load the master bias from a previous run

        Parameters
        ----------
        mask : array of shape (nrow, ncol)
            Bad pixel mask

        Returns
        -------
        data : masked array of shape (nrow, ncol)
            master bias data, with the bad pixel mask applied
        head : FITS header
            header of the master bias
        """
        try:
            with fits.open(self.savefile, memmap=False) as hdu:
                data, head = hdu[0].data, hdu[0].head
            data = np.ma.masked_array(data, mask=mask)
            logger.info("Data file: %s", self.savefile)
        except FileNotFoundError as ex:
            if self.allow_failure:
                logger.warning("No data file found")
                data, head = None, None
            else:
                raise ex
        return data, head


class Mask(Step):
    """Load the bad pixel mask for the given instrument/channel"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)

    def run(self):
        """Load the mask file from disk

        Returns
        -------
        mask : array of shape (nrow, ncol)
            Bad pixel mask for this setting
        """
        return self.load()

    def load(self):
        """Load the mask file from disk

        Returns
        -------
        mask : array of shape (nrow, ncol)
            Bad pixel mask for this setting
        """
        mask_file = self.instrument.get_mask_filename(channel=self.channel)
        try:
            mask, _ = self.instrument.load_fits(mask_file, self.channel, extension=0)
            mask = mask.data.astype(bool)  # 1 = bad/masked (numpy convention)
            logger.info("Bad pixel mask file: %s", mask_file)
        except (FileNotFoundError, ValueError):
            logger.error(
                "Bad Pixel Mask datafile %s not found. Using all pixels instead.",
                mask_file,
            )
            mask = False
        return mask


class Bias(Step):
    """Calculates the master bias"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["mask"]
        self._loadDependsOn += ["mask"]

        #:int: polynomial degree of the fit between exposure time and pixel values
        self.degree = config["degree"]

    @property
    def savefile(self):
        """str: Name of master bias fits file"""
        return join(self.output_dir, self.prefix + ".bias.fits")

    def run(self, files, mask=None):
        """Calculate the master bias

        Parameters
        ----------
        files : list(str)
            bias files
        mask : array of shape (nrow, ncol), optional
            bad pixel map

        Returns
        -------
        bias : masked array of shape (nrow, ncol)
            master bias data, with the bad pixel mask applied
        bhead : FITS header
            header of the master bias
        """
        logger.info("Bias Files: %s", files)

        if self.degree == 0:
            # If the degree is 0, we just combine all images into a single master bias
            # this works great if we assume there is no dark at exposure time 0
            bias, bhead = combine_bias(
                files,
                self.instrument,
                self.channel,
                mask=mask,
                plot=self.plot,
                plot_title=self.plot_title,
            )
        else:
            # Otherwise we fit a polynomial to each pixel in the image, with
            # the pixel value versus the exposure time. The constant coefficients
            # are then the bias, and the others are used to scale with the
            # exposure time
            bias, bhead = combine_polynomial(
                files,
                self.instrument,
                self.channel,
                mask=mask,
                degree=self.degree,
                plot=self.plot,
                plot_title=self.plot_title,
            )

        self.save(bias.data, bhead)
        return bias, bhead

    def save(self, bias, bhead):
        """Save the master bias to a FITS file

        Parameters
        ----------
        bias : array of shape (nrow, ncol)
            bias data
        bhead : FITS header
            bias header
        """
        bias = np.asarray(bias, dtype=np.float32)

        if self.degree == 0:
            hdus = [fits.PrimaryHDU(data=bias, header=bhead, scale_back=False)]
        else:
            hdus = [fits.PrimaryHDU(data=bias[0], header=bhead, scale_back=False)]
            for i in range(1, len(bias)):
                hdus += [fits.ImageHDU(data=bias[i])]
        hdus = fits.HDUList(hdus)

        hdus[0].header["BZERO"] = 0
        hdus.writeto(
            self.savefile,
            overwrite=True,
            output_verify="silentfix+ignore",
        )
        logger.info("Created master bias file: %s", self.savefile)

    def load(self, mask):
        """Load the master bias from a previous run

        Parameters
        ----------
        mask : array of shape (nrow, ncol)
            Bad pixel mask

        Returns
        -------
        bias : masked array of shape (nrow, ncol)
            master bias data, with the bad pixel mask applied
        bhead : FITS header
            header of the master bias
        """
        try:
            logger.info("Master bias file: %s", self.savefile)
            with fits.open(self.savefile, memmap=False) as hdu:
                degree = len(hdu) - 1
                if degree == 0:
                    bias, bhead = hdu[0].data, hdu[0].header
                    bias = np.ma.masked_array(bias, mask=mask)
                else:
                    bhead = hdu[0].header
                    bias = np.array([h.data for h in hdu])
                    bias = np.ma.masked_array(
                        bias, mask=[mask for _ in range(len(hdu))]
                    )
        except FileNotFoundError:
            logger.warning("No intermediate bias file found. Using Bias = 0 instead.")
            bias, bhead = None, None
        return bias, bhead


class Flat(CalibrationStep):
    """Calculates the master flat"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._loadDependsOn += ["mask"]

    @property
    def savefile(self):
        """str: Name of master bias fits file"""
        return join(self.output_dir, self.prefix + ".flat.fits")

    def save(self, flat, fhead):
        """Save the master flat to a FITS file

        Parameters
        ----------
        flat : array of shape (nrow, ncol)
            master flat data
        fhead : FITS header
            master flat header
        """
        flat = np.asarray(flat, dtype=np.float32)
        fits.writeto(
            self.savefile,
            data=flat,
            header=fhead,
            overwrite=True,
            output_verify="silentfix+ignore",
        )
        logger.info("Created master flat file: %s", self.savefile)

    def run(self, files, bias=None, mask=None):
        """Calculate the master flat, with the bias already subtracted

        Parameters
        ----------
        files : list(str)
            flat files
        bias : tuple(array of shape (nrow, ncol), FITS header), optional
            master bias and header
        mask : array of shape (nrow, ncol), optional
            Bad pixel mask

        Returns
        -------
        flat : masked array of shape (nrow, ncol)
            Master flat with bad pixel map applied
        fhead : FITS header
            Master flat FITS header
        """
        logger.info("Flat files: %s", files)
        # This is just the calibration of images
        flat, fhead = self.calibrate(files, mask, bias, None)
        # And then save it
        self.save(flat.data, fhead)
        return flat, fhead

    def load(self, mask):
        """Load master flat from disk

        Parameters
        ----------
        mask : array of shape (nrow, ncol)
            Bad pixel mask

        Returns
        -------
        flat : masked array of shape (nrow, ncol)
            Master flat with bad pixel map applied
        fhead : FITS header
            Master flat FITS header
        """
        try:
            with fits.open(self.savefile, memmap=False) as hdu:
                flat, fhead = hdu[0].data, hdu[0].header
            flat = np.ma.masked_array(flat, mask=mask)
            logger.info("Master flat file: %s", self.savefile)
        except FileNotFoundError:
            logger.warning(
                "No intermediate file for the flat field found. Using Flat = 1 instead"
            )
            flat, fhead = None, None
        return flat, fhead


class Trace(CalibrationStep):
    """Determine the polynomial fits describing the pixel locations of each trace"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)

        #:int: Minimum size of each cluster to be included in further processing
        self.min_cluster = config["min_cluster"]
        #:int, float: Minimum width of each cluster after mergin
        self.min_width = config["min_width"]
        #:int: Smoothing width along x-axis (dispersion direction)
        self.filter_x = config.get("filter_x", 0)
        #:int: Smoothing width along y-axis (cross-dispersion direction)
        self.filter_y = config["filter_y"]
        #:str: Type of smoothing filter (boxcar, gaussian, whittaker)
        self.filter_type = config.get("filter_type", "boxcar")
        #:int: Absolute background noise threshold
        self.noise = config.get("noise", 0)
        #:float: Relative background noise threshold (fraction of background)
        self.noise_relative = config.get("noise_relative", 0)
        #:int: Polynomial degree of the fit to each order
        self.fit_degree = config["degree"]

        self.degree_before_merge = config["degree_before_merge"]
        self.regularization = config["regularization"]
        self.closing_shape = config["closing_shape"]
        self.opening_shape = config["opening_shape"]
        self.auto_merge_threshold = config["auto_merge_threshold"]
        self.merge_min_threshold = config["merge_min_threshold"]
        self.sigma = config["split_sigma"]
        #:int: Number of pixels at the edge of the detector to ignore
        self.border_width = config["border_width"]
        #:bool: Whether to use manual alignment
        self.manual = config["manual"]

        # Per-trace heights (derived from trace_objects)
        self.heights = None

        # Trace objects - the canonical representation
        self.trace_objects: list[TraceData] = None

    @property
    def savefile(self):
        """str: Name of the tracing file (FITS format)"""
        return join(self.output_dir, self.prefix + ".traces.fits")

    def run(self, files, mask=None, bias=None):
        """Determine polynomial coefficients describing order locations

        Parameters
        ----------
        files : list(str)
            Observation used for order tracing (should only have one element)
        mask : array of shape (nrow, ncol), optional
            Bad pixel mask
        bias : tuple, optional
            Bias correction

        Returns
        -------
        list[TraceData]
            Trace objects with position, column_range, height, and identity.
        """

        logger.info("Tracing files: %s", files)

        # Load order_centers for m assignment if available
        order_centers = self._load_order_centers()

        # Check if we should trace file groups separately
        fibers_config = getattr(self.instrument.config, "fibers", None)
        trace_by = getattr(fibers_config, "trace_by", None) if fibers_config else None

        if trace_by and len(files) > 1:
            raw_traces = self._trace_by_groups(
                files, mask, bias, trace_by, order_centers
            )
        else:
            raw_traces = self._trace_single(files, mask, bias, order_centers)

        # Store heights for backward compatibility
        self.heights = np.array(
            [t.height if t.height is not None else np.nan for t in raw_traces]
        )

        # Group fibers if configured (creates new traces with group set)
        if fibers_config is not None and (
            fibers_config.groups is not None or fibers_config.bundles is not None
        ):
            self.trace_objects = group_fibers(
                raw_traces, fibers_config, degree=self.fit_degree
            )
        else:
            self.trace_objects = raw_traces

        self.save()

        return self.trace_objects

    def _load_order_centers(self) -> dict[int, float] | None:
        """Load order_centers from instrument config if available.

        Returns
        -------
        dict[int, float] or None
            Order number -> y-position mapping, or None if not configured.
        """
        fibers_config = getattr(self.instrument.config, "fibers", None)
        if fibers_config is None:
            return None

        # Check for inline order_centers
        if fibers_config.order_centers is not None:
            return fibers_config.order_centers

        # Check for order_centers_file
        if fibers_config.order_centers_file is None:
            return None

        from pathlib import Path

        import yaml

        centers_file = fibers_config.order_centers_file
        # Substitute {channel} template
        if self.channel and "{channel}" in centers_file:
            centers_file = centers_file.format(channel=self.channel.lower())

        inst_dir = getattr(self.instrument, "_inst_dir", None)
        path = Path(centers_file)
        if not path.is_absolute() and inst_dir:
            path = Path(inst_dir) / centers_file

        if not path.exists():
            logger.info("Order centers file not found: %s", path)
            return None

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            logger.info("Order centers file is empty: %s", path)
            return None

        if "order_centers" in data:
            data = data["order_centers"]

        order_centers = {int(k): float(v) for k, v in data.items()}
        logger.info("Loaded order centers from %s: %d orders", path, len(order_centers))
        return order_centers

    def _trace_by_groups(self, files, mask, bias, trace_by, order_centers):
        """Trace files grouped by header value, then merge traces.

        Parameters
        ----------
        files : list(str)
            Files to trace
        mask : array, optional
            Bad pixel mask
        bias : tuple, optional
            Bias correction
        trace_by : str
            Header keyword to group files by
        order_centers : dict[int, float] | None
            Order centers for m assignment

        Returns
        -------
        list[TraceData]
            Merged traces from all groups
        """
        # Group files by header value
        file_groups = {}
        for f in files:
            hdr = fits.getheader(f)
            group_key = hdr.get(trace_by, "unknown")
            if group_key not in file_groups:
                file_groups[group_key] = []
            file_groups[group_key].append(f)

        logger.info(
            "Tracing %d file groups separately (grouped by %s): %s",
            len(file_groups),
            trace_by,
            list(file_groups.keys()),
        )

        # Trace each group
        all_traces = []
        for group_key, group_files in file_groups.items():
            logger.info("Tracing group '%s': %d files", group_key, len(group_files))
            traces = self._trace_single(group_files, mask, bias, order_centers)
            logger.info("  Found %d traces", len(traces))
            all_traces.extend(traces)

        # Sort by y-position
        def sort_key(t):
            # Use middle of column range for y evaluation
            x_mid = sum(t.column_range) / 2
            return t.y_at_x(x_mid)

        all_traces.sort(key=sort_key)

        logger.info(
            "Merged %d total traces from %d groups", len(all_traces), len(file_groups)
        )

        return all_traces

    def _trace_single(self, files, mask, bias, order_centers):
        """Trace a single set of files.

        Returns
        -------
        list[TraceData]
            Trace objects with fiber_idx set
        """
        trace_img, ohead = self.calibrate(files, mask, bias, None)

        # Get fibers_per_order from instrument config for auto-pairing
        fibers_config = getattr(self.instrument.config, "fibers", None)
        fpo = (
            getattr(fibers_config, "fibers_per_order", None) if fibers_config else None
        )

        traces = mark_orders(
            trace_img,
            min_cluster=self.min_cluster,
            min_width=self.min_width,
            filter_x=self.filter_x,
            filter_y=self.filter_y,
            filter_type=self.filter_type,
            noise=self.noise,
            noise_relative=self.noise_relative,
            degree=self.fit_degree,
            degree_before_merge=self.degree_before_merge,
            regularization=self.regularization,
            closing_shape=self.closing_shape,
            opening_shape=self.opening_shape,
            border_width=self.border_width,
            manual=self.manual,
            auto_merge_threshold=self.auto_merge_threshold,
            merge_min_threshold=self.merge_min_threshold,
            sigma=self.sigma,
            plot=self.plot,
            plot_title=self.plot_title,
            order_centers=order_centers,
            fibers_per_order=fpo,
        )

        return traces

    def save(self):
        """Save tracing results to disk in FITS format."""
        os.makedirs(os.path.dirname(self.savefile), exist_ok=True)

        if self.trace_objects is None or len(self.trace_objects) == 0:
            logger.warning("No traces to save")
            return

        save_traces(self.savefile, self.trace_objects, steps=["trace"])
        logger.info("Created trace file: %s", self.savefile)

    def load(self):
        """Load tracing results from FITS format.

        Returns
        -------
        list[TraceData]
            Trace objects with position, column_range, height, and identity.
        """
        logger.info("Trace file: %s", self.savefile)
        self.trace_objects, header = load_traces(self.savefile)
        logger.info("Loaded %d traces", len(self.trace_objects))
        return self.trace_objects

    def get_traces_for_step(self, step_name: str) -> dict[str, list[TraceData]]:
        """Get traces appropriate for a specific reduction step.

        Uses the instrument's fibers.use config to select traces.

        Parameters
        ----------
        step_name : str
            Name of the reduction step (e.g., "science", "curvature")

        Returns
        -------
        dict[str, list[TraceData]]
            {group_name: [traces]} for each selected group
        """
        fibers_config = getattr(self.instrument.config, "fibers", None)
        return select_traces_for_step(self.trace_objects, fibers_config, step_name)


class BackgroundScatter(CalibrationStep):
    """Determine the background scatter"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["trace"]

        #:tuple(int, int): Polynomial degrees for the background scatter fit, in row, column direction
        self.scatter_degree = config["scatter_degree"]
        self.extraction_height = config["extraction_height"]
        self.sigma_cutoff = config["scatter_cutoff"]
        self.border_width = config["border_width"]

    @property
    def savefile(self):
        """str: Name of the scatter file"""
        return join(self.output_dir, self.prefix + ".scatter.npz")

    def run(self, files, trace: list[TraceData], mask=None, bias=None):
        logger.info("Background scatter files: %s", files)

        scatter_img, shead = self.calibrate(files, mask, bias)

        # Apply fiber selection based on instrument config
        selected = self._select_traces(trace, "scatter")
        # Flatten all selected groups
        trace_list = [t for traces in selected.values() for t in traces]

        scatter = estimate_background_scatter(
            scatter_img,
            trace_list,
            extraction_height=self.extraction_height,
            scatter_degree=self.scatter_degree,
            sigma_cutoff=self.sigma_cutoff,
            border_width=self.border_width,
            plot=self.plot,
            plot_title=self.plot_title,
        )

        self.save(scatter)
        return scatter

    def save(self, scatter):
        """Save scatter results to disk

        Parameters
        ----------
        scatter : array
            scatter coefficients
        """
        np.savez(self.savefile, scatter=scatter)
        logger.info("Created background scatter file: %s", self.savefile)

    def load(self):
        """Load scatter results from disk

        Returns
        -------
        scatter : array
            scatter coefficients
        """
        try:
            data = np.load(self.savefile, allow_pickle=True)
            logger.info("Background scatter file: %s", self.savefile)
        except FileNotFoundError:
            logger.warning(
                "No intermediate files found for the scatter. Using scatter = 0 instead."
            )
            data = {"scatter": None}
        scatter = data["scatter"]
        return scatter


class NormalizeFlatField(Step):
    """Calculate the 'normalized' flat field image"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["flat", "trace", "scatter"]

        #:{'normalize'}: Extraction method to use
        self.extraction_method = config["extraction_method"]
        if self.extraction_method == "normalize":
            #:dict: arguments for the extraction
            self.extraction_kwargs = {
                "extraction_height": config["extraction_height"],
                "lambda_sf": config["smooth_slitfunction"],
                "lambda_sp": config["smooth_spectrum"],
                "osample": config["oversampling"],
                "swath_width": config["swath_width"],
                "maxiter": config["maxiter"],
                "reject_threshold": config.get("extraction_reject", 6),
            }
        else:
            raise ValueError(
                f"Extraction method {self.extraction_method} not supported for step 'norm_flat'"
            )
        #:int: Threshold of the normalized flat field (values below this are just 1)
        self.threshold = config["threshold"]
        self.threshold_lower = config["threshold_lower"]

    @property
    def savefile(self):
        """str: Name of the blaze file"""
        return join(self.output_dir, self.prefix + ".flat_norm.npz")

    def run(self, flat, trace: list[TraceData], scatter=None):
        """Calculate the 'normalized' flat field

        Parameters
        ----------
        flat : tuple(array, header)
            Master flat, and its FITS header
        trace : list[TraceData]
            Trace objects from trace step
        scatter : array, optional
            Background scatter model

        Returns
        -------
        norm : array of shape (nrow, ncol)
            normalized flat field
        blaze : array of shape (ntrace, ncol)
            Continuum level as determined from the flat field for each order
        slitfunc : list of arrays
            Slit function for each order
        slitfunc_meta : dict
            Metadata for slitfunc (extraction_height, osample, trace_range)
        """
        flat, fhead = flat

        # Apply fiber selection based on instrument config
        selected = self._select_traces(trace, "norm_flat")
        trace_list = [t for traces in selected.values() for t in traces]

        # Apply trace_range if specified
        if self.trace_range is not None:
            trace_list = trace_list[self.trace_range[0] : self.trace_range[1]]

        extraction_kwargs = dict(self.extraction_kwargs)
        default_height = extraction_kwargs.pop("extraction_height", 0.5)

        # if threshold is smaller than 1, assume percentage value is given
        if self.threshold <= 1:
            threshold = np.percentile(flat, self.threshold * 100)
        else:
            threshold = self.threshold

        norm, _, blaze, slitfunc, column_range = extract_normalize(
            flat,
            trace_list,
            extraction_height=default_height,
            gain=fhead["e_gain"],
            readnoise=fhead["e_readn"],
            dark=fhead["e_drk"],
            scatter=scatter,
            threshold=threshold,
            threshold_lower=self.threshold_lower,
            plot=self.plot,
            plot_title=self.plot_title,
            **extraction_kwargs,
        )

        blaze = np.ma.filled(blaze, 0)
        norm = np.ma.filled(norm, 1)
        norm = np.nan_to_num(norm, nan=1)

        # Metadata for slitfunc
        n_traces = len(trace_list)
        slitfunc_meta = {
            "extraction_height": default_height,
            "osample": extraction_kwargs["osample"],
            "trace_range": (0, n_traces),
            "n_traces_selected": n_traces,
        }
        self.save(norm, blaze, slitfunc, slitfunc_meta)
        return norm, blaze, slitfunc, slitfunc_meta

    def save(self, norm, blaze, slitfunc, slitfunc_meta):
        """Save normalized flat field results to disk

        Parameters
        ----------
        norm : array of shape (nrow, ncol)
            normalized flat field
        blaze : array of shape (ntrace, ncol)
            Continuum level as determined from the flat field for each order
        slitfunc : list of arrays
            Slit function for each order
        slitfunc_meta : dict
            Metadata for slitfunc (extraction_height, osample, trace_range)
        """
        # Stack slitfunctions into 2D array if all same length, else save as object array
        try:
            slitfunc_arr = np.array(slitfunc)
        except ValueError:
            slitfunc_arr = np.array(slitfunc, dtype=object)
        np.savez(
            self.savefile,
            blaze=blaze,
            norm=norm,
            slitfunc=slitfunc_arr,
            slitfunc_meta=slitfunc_meta,
        )
        logger.info("Created normalized flat file: %s", self.savefile)

    def load(self):
        """Load normalized flat field results from disk

        Returns
        -------
        norm : array of shape (nrow, ncol)
            normalized flat field
        blaze : array of shape (ntrace, ncol)
            Continuum level as determined from the flat field for each order
        slitfunc : list of arrays, or None
            Slit function for each order (None if not available)
        slitfunc_meta : dict or None
            Metadata for slitfunc (extraction_height, osample, trace_range)
        """
        try:
            data = np.load(self.savefile, allow_pickle=True)
            logger.info("Normalized flat file: %s", self.savefile)
        except FileNotFoundError:
            logger.warning(
                "No intermediate files found for the normalized flat field. Using flat = 1 instead."
            )
            data = {"blaze": None, "norm": None}
        blaze = data["blaze"]
        norm = data["norm"]
        slitfunc = data.get("slitfunc", None)
        if slitfunc is not None:
            slitfunc = list(slitfunc)
        slitfunc_meta = data.get("slitfunc_meta", None)
        if slitfunc_meta is not None:
            slitfunc_meta = slitfunc_meta.item()  # unwrap 0-d array from npz
        return norm, blaze, slitfunc, slitfunc_meta


class WavelengthCalibrationMaster(CalibrationStep, ExtractionStep):
    """Create wavelength calibration master image"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["norm_flat", "bias"]

    def savefile_for_group(self, group: str) -> str:
        """Get savefile path for a specific group."""
        if group == "all":
            return join(self.output_dir, self.prefix + ".wavecal_master.fits")
        return join(self.output_dir, self.prefix + f"_{group}.wavecal_master.fits")

    @property
    def savefile(self):
        """str: Name of the wavelength echelle file (single-group compat)"""
        return self.savefile_for_group("all")

    def run(
        self,
        files,
        trace: list[TraceData],
        mask=None,
        bias=None,
        norm_flat=None,
    ):
        """Extract wavelength calibration spectra, per fiber group.

        Parameters
        ----------
        files : list(str)
            wavelength calibration files
        trace : list[TraceData]
            Trace objects from trace step
        mask : array of shape (nrow, ncol), optional
            Bad pixel mask
        bias : tuple, optional
            Master bias
        norm_flat : tuple, optional
            Normalized flat field

        Returns
        -------
        results : dict[str, tuple]
            {group: (wavecal_spec, thead)} for each fiber group
        """
        if len(files) == 0:
            raise FileNotFoundError("No files found for wavelength calibration")
        logger.info("Wavelength calibration files: %s", files)

        # Apply fiber selection based on instrument config
        selected = self._select_traces(trace, "wavecal_master")

        # Load wavecal image (same for all groups)
        orig, thead = self.calibrate(files, mask, bias, norm_flat)

        # Extract per group
        results = {}
        for group, trace_list in selected.items():
            if not trace_list:
                logger.warning("No traces for group '%s', skipping", group)
                continue
            logger.info(
                "Extracting wavecal for group '%s' (%d traces)", group, len(trace_list)
            )
            wavecal_spec, _, _, _ = self.extract_to_arrays(orig, thead, trace_list)
            results[group] = (wavecal_spec, thead)

        self.save(results)
        return results

    def save(self, results: dict):
        """Save the master wavelength calibration to FITS files.

        Parameters
        ----------
        results : dict[str, tuple]
            {group: (wavecal_spec, thead)} for each fiber group
        """
        for group, (wavecal_spec, thead) in results.items():
            wavecal_spec = np.asarray(wavecal_spec, dtype=np.float64)
            savefile = self.savefile_for_group(group)
            fits.writeto(
                savefile,
                data=wavecal_spec,
                header=thead,
                overwrite=True,
                output_verify="silentfix+ignore",
            )
            logger.info("Created wavelength calibration spectrum file: %s", savefile)

    def load(self):
        """Load master wavelength calibration from disk.

        Returns
        -------
        results : dict[str, tuple]
            {group: (wavecal_spec, thead)} for each fiber group
        """
        import glob

        # Find all wavecal_master files for this prefix
        # Naming: {prefix}.wavecal_master.fits (no group)
        #         {prefix}_{group}.wavecal_master.fits (with group)
        pattern = join(self.output_dir, self.prefix + "*.wavecal_master.fits")
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(f"No wavecal_master files found matching {pattern}")

        results = {}
        prefix_base = self.prefix
        for fpath in files:
            basename = os.path.basename(fpath)
            stem = basename.replace(".wavecal_master.fits", "")
            if stem == prefix_base:
                group = "all"
            elif stem.startswith(prefix_base + "_"):
                group = stem[len(prefix_base) + 1 :]
            else:
                continue

            with fits.open(fpath, memmap=False) as hdu:
                wavecal_spec, thead = hdu[0].data, hdu[0].header
            logger.info("Loaded wavelength calibration spectrum: %s", fpath)
            results[group] = (wavecal_spec, thead)

        return results


class WavelengthCalibrationInitialize(Step):
    """Create the initial wavelength solution file"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["wavecal_master"]
        self._loadDependsOn += ["config", "wavecal_master"]

        self.degree = config["degree"]
        self.resid_delta = config["resid_delta"]
        self.match_tolerance = config["match_tolerance"]
        self.iterations = config["iterations"]
        self.edge_margin = config["edge_margin"]
        self.width_min = config["width_min"]
        self.width_max = config["width_max"]
        self.atlas_name = config["atlas"]
        self.medium = config["medium"]
        self.smoothing = config["smoothing"]
        self.cutoff = config["cutoff"]

    def savefile_for_group(self, group: str) -> str:
        """Get savefile path for a specific group."""
        if group == "all":
            return join(self.output_dir, self.prefix + ".linelist.npz")
        return join(self.output_dir, self.prefix + f"_{group}.linelist.npz")

    @property
    def savefile(self):
        """str: Name of the linelist file (single-group compat)"""
        return self.savefile_for_group("all")

    def run(self, wavecal_master: dict):
        """Run iterative line matching for each fiber group.

        Parameters
        ----------
        wavecal_master : dict[str, tuple]
            {group: (wavecal_spec, thead)} from wavecal_master step

        Returns
        -------
        results : dict[str, LineList]
            {group: linelist} for each fiber group
        """
        results = {}
        for group, (wavecal_spec, thead) in wavecal_master.items():
            logger.info("Running wavecal_init for group '%s'", group)

            # Get the initial wavelength guess from the instrument
            wave_range = self.instrument.get_wavelength_range(thead, self.channel)
            if wave_range is None:
                raise ValueError(
                    "This instrument is missing an initial wavelength guess for wavecal_init"
                )

            module = WavelengthCalibrationInitializeModule(
                plot=self.plot,
                plot_title=f"{self.plot_title} [{group}]" if self.plot_title else group,
                degree=self.degree,
                resid_delta=self.resid_delta,
                match_tolerance=self.match_tolerance,
                iterations=self.iterations,
                edge_margin=self.edge_margin,
                width_min=self.width_min,
                width_max=self.width_max,
                atlas_name=self.atlas_name,
                atlas_search_dirs=[self.instrument._inst_dir],
                medium=self.medium,
                smoothing=self.smoothing,
                cutoff=self.cutoff,
            )
            linelist = module.execute(wavecal_spec, wave_range)
            results[group] = linelist

        self.save(results)
        return results

    def save(self, results: dict):
        """Save linelists for each fiber group."""
        for group, linelist in results.items():
            savefile = self.savefile_for_group(group)
            linelist.save(savefile)
            logger.info("Created wavelength calibration linelist file: %s", savefile)

    def load(self, config, wavecal_master: dict):
        """Load linelists for each fiber group.

        Falls back to instrument-provided wavecal file if custom not found.
        """

        results = {}

        # First try to load custom linelists matching wavecal_master groups
        for group in wavecal_master.keys():
            savefile = self.savefile_for_group(group)
            try:
                linelist = LineList.load(savefile)
                logger.info("Loaded linelist for group '%s': %s", group, savefile)
                results[group] = linelist
            except FileNotFoundError:
                pass

        # If we found custom linelists, use them
        if results:
            return results

        # Otherwise, fall back to instrument-provided wavecal file
        # (applies same linelist to all groups)
        first_group = next(iter(wavecal_master.keys()))
        _, thead = wavecal_master[first_group]
        reference = self.instrument.get_wavecal_filename(
            thead, self.channel, **config["instrument"]
        )
        linelist = LineList.load(reference)
        logger.info("Wavelength calibration linelist file: %s", reference)

        # Apply same linelist to all groups
        for group in wavecal_master.keys():
            results[group] = linelist

        return results


class WavelengthCalibrationFinalize(Step):
    """Perform wavelength calibration"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["wavecal_master", "wavecal_init", "trace"]

        #:tuple(int, int): Polynomial degree of the wavelength calibration in order, column direction
        self.degree = config["degree"]
        #:bool: Whether to use manual alignment instead of cross correlation
        self.manual = config["manual"]
        #:float: residual threshold in m/s
        self.threshold = config["threshold"]
        #:int: Number of iterations in the remove lines, auto id cycle
        self.iterations = config["iterations"]
        #:{'1D', '2D'}: Whether to use 1d or 2d polynomials
        self.dimensionality = config["dimensionality"]
        #:int: Number of detector offset steps, due to detector design
        self.nstep = config["nstep"]
        #:int: How many columns to use in the 2D cross correlation alignment. 0 means all pixels (slow).
        self.correlate_cols = config["correlate_cols"]
        #:float: fraction of columns, to allow individual orders to shift
        self.shift_window = config["shift_window"]
        #:str: name of the line atlas
        self.atlas_name = config["atlas"]
        #:str: medium of the detector, vac or air
        self.medium = config["medium"]

    def savefile_for_group(self, group: str) -> str:
        """Get savefile path for a specific group."""
        if group == "all":
            return join(self.output_dir, self.prefix + ".linelist.npz")
        return join(self.output_dir, self.prefix + f"_{group}.linelist.npz")

    @property
    def savefile(self):
        """str: Name of the linelist file (single-group compat)"""
        return self.savefile_for_group("all")

    def run(self, wavecal_master: dict, wavecal_init: dict, trace: list):
        """Perform wavelength calibration for each fiber group.

        Fits wavelength polynomials and updates trace objects in-place.
        Returns linelists for diagnostics.

        Parameters
        ----------
        wavecal_master : dict[str, tuple]
            {group: (wavecal_spec, thead)} from wavecal_master step
        wavecal_init : dict[str, LineList]
            {group: linelist} from wavecal_init step
        trace : list[TraceData]
            Trace objects to update with wavelength polynomials

        Returns
        -------
        results : dict[str, LineList]
            {group: linelist} for each fiber group (wavelengths are in traces)
        """
        results_for_save = {}
        results = {}

        for group in wavecal_master.keys():
            if group not in wavecal_init:
                logger.warning("No linelist for group '%s', skipping", group)
                continue

            wavecal_spec, thead = wavecal_master[group]
            linelist = wavecal_init[group]
            logger.info("Running wavecal finalize for group '%s'", group)

            module = WavelengthCalibrationModule(
                plot=self.plot,
                plot_title=f"{self.plot_title} [{group}]" if self.plot_title else group,
                manual=self.manual,
                degree=self.degree,
                threshold=self.threshold,
                iterations=self.iterations,
                dimensionality=self.dimensionality,
                nstep=self.nstep,
                correlate_cols=self.correlate_cols,
                shift_window=self.shift_window,
                atlas_name=self.atlas_name,
                atlas_search_dirs=[self.instrument._inst_dir],
                medium=self.medium,
            )
            wlen, wave, linelist = module.execute(wavecal_spec, linelist)
            results_for_save[group] = (wave, linelist)
            results[group] = linelist

        # Update trace objects in-place
        self._update_traces(trace, results_for_save)

        self.save(results_for_save, trace)
        return results

    def _update_traces(self, trace: list, results: dict):
        """Update trace objects with wavelength polynomials and order numbers.

        Modifies traces in-place.

        Parameters
        ----------
        trace : list[TraceData]
            All trace objects
        results : dict[str, tuple]
            {group: (wave_coef, linelist)} polynomial coefficients per group
        """
        # Group traces by their group attribute AND by fiber_idx
        traces_by_group = {}
        traces_by_fiber = {}
        for i, t in enumerate(trace):
            g = str(t.group) if t.group is not None else "all"
            if g not in traces_by_group:
                traces_by_group[g] = []
            traces_by_group[g].append((i, t))
            if t.fiber_idx is not None:
                fkey = f"fiber_{t.fiber_idx}"
                if fkey not in traces_by_fiber:
                    traces_by_fiber[fkey] = []
                traces_by_fiber[fkey].append((i, t))

        for group, (wave, linelist) in results.items():
            if group in traces_by_group:
                group_traces = traces_by_group[group]
            elif group in traces_by_fiber:
                group_traces = traces_by_fiber[group]
            elif "all" in traces_by_group:
                group_traces = traces_by_group["all"]
            else:
                logger.warning("No traces found for group '%s'", group)
                continue

            # Update trace.m from obase if not already set
            obase = linelist.obase
            if obase is not None:
                already_have_m = any(t.m is not None for _i, t in group_traces)
                if already_have_m:
                    logger.debug(
                        "Traces for group '%s' already have m values, skipping obase",
                        group,
                    )
                else:
                    for idx_in_group, (_i, t) in enumerate(group_traces):
                        t.m = obase + idx_in_group
                    logger.info(
                        "Updated trace order numbers for group '%s' with obase=%d",
                        group,
                        obase,
                    )

            # Store wavelength polynomial in each trace.
            if self.dimensionality == "1D":
                for idx_in_group, (_i, t) in enumerate(group_traces):
                    if idx_in_group < len(wave):
                        t.wave = wave[idx_in_group]
            else:
                # Evaluate 2D poly P(x, order_idx) at each trace's 0-based
                # index to get a 1D poly in x (np.polyfit convention).
                for idx_in_group, (_i, t) in enumerate(group_traces):
                    poly_1d = np.polynomial.polynomial.polyval(idx_in_group, wave.T)
                    t.wave = poly_1d[::-1]

    def save(self, results: dict, trace: list):
        """Save linelists and updated traces to disk.

        Parameters
        ----------
        results : dict[str, tuple]
            {group: (wave, linelist)} - wave polynomials and linelists
        trace : list[TraceData]
            Already-updated trace objects
        """
        for group, (_wave, linelist) in results.items():
            savefile = self.savefile_for_group(group)
            # Re-normalize order numbers to 0-based so the linelist can be
            # reloaded as a starting point without accumulating alignment offsets.
            if len(linelist) > 0:
                min_order = int(np.min(linelist["order"]))
                if min_order != 0:
                    linelist["order"] -= min_order
            linelist.save(savefile)
            logger.info("Updated linelist with refined positions: %s", savefile)

        trace_file = join(self.output_dir, self.prefix + ".traces.fits")
        try:
            # Read existing header to preserve metadata
            header = None
            if os.path.exists(trace_file):
                with fits.open(trace_file, memmap=False) as hdu:
                    header = hdu[0].header
            if header is None:
                header = fits.Header()
            steps = header.get("E_STEPS", "trace").split(",")
            if "wavecal" not in steps:
                steps.append("wavecal")
            save_traces(trace_file, trace, header, steps=steps)
            logger.info("Updated traces with wavelength data: %s", trace_file)
        except Exception as e:
            logger.warning("Could not update traces.fits with wavelength: %s", e)

    def load(self):
        """Load wavelength calibration linelists.

        Wavelength data is stored in traces.fits, not returned here.

        Returns
        -------
        results : dict[str, LineList]
            {group: linelist} for each fiber group
        """
        import glob

        old_wavecal_file = join(self.output_dir, self.prefix + ".wavecal.npz")

        # Find all linelist files
        # Naming: {prefix}.linelist.npz (no group)
        #         {prefix}_{group}.linelist.npz (with group)
        pattern = join(self.output_dir, self.prefix + "*.linelist.npz")
        linelist_files = glob.glob(pattern)

        if linelist_files:
            results = {}
            prefix_base = self.prefix
            for fpath in linelist_files:
                basename = os.path.basename(fpath)
                stem = basename.replace(".linelist.npz", "")
                if stem == prefix_base:
                    group = "all"
                elif stem.startswith(prefix_base + "_"):
                    group = stem[len(prefix_base) + 1 :]
                else:
                    continue

                linelist = LineList.load(fpath)
                results[group] = linelist
                logger.info("Loaded linelist for group '%s': %s", group, fpath)

            if results:
                return results

        # Fall back to old .wavecal.npz format
        if os.path.exists(old_wavecal_file):
            data = np.load(old_wavecal_file, allow_pickle=True)
            logger.info("Wavelength calibration file (legacy): %s", old_wavecal_file)
            linelist = data["linelist"]
            return {"all": linelist}

        raise FileNotFoundError(f"No wavelength calibration found: {self.savefile}")


class LaserFrequencyCombMaster(CalibrationStep, ExtractionStep):
    """Create a laser frequency comb (or similar) master image"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["norm_flat"]

    @property
    def savefile(self):
        """str: Name of the wavelength echelle file"""
        return join(self.output_dir, self.prefix + ".comb_master.fits")

    def run(
        self,
        files,
        trace: list[TraceData],
        mask=None,
        bias=None,
        norm_flat=None,
    ):
        """Improve the wavelength calibration with a laser frequency comb (or similar)

        Parameters
        ----------
        files : list(str)
            observation files
        trace : list[TraceData]
            Trace objects from trace step
        mask : array of shape (nrow, ncol), optional
            Bad pixel mask
        bias : tuple, optional
            results from the bias step
        norm_flat : tuple, optional
            results from the norm_flat step

        Returns
        -------
        comb : array of shape (ntrace, ncol)
            extracted frequency comb image
        chead : Header
            FITS header of the combined image
        """

        if len(files) == 0:
            raise FileNotFoundError("No files for Laser Frequency Comb found")
        logger.info("Frequency comb files: %s", files)

        # Combine the input files and calibrate
        orig, chead = self.calibrate(files, mask, bias, norm_flat)
        # Extract the spectrum
        comb, _, _, _ = self.extract_to_arrays(orig, chead, trace)
        self.save(comb, chead)
        return comb, chead

    def save(self, comb, chead):
        """Save the master comb to a FITS file

        Parameters
        ----------
        comb : array of shape (nrow, ncol)
            master comb data
        chead : FITS header
            master comb header
        """
        comb = np.asarray(comb, dtype=np.float64)
        fits.writeto(
            self.savefile,
            data=comb,
            header=chead,
            overwrite=True,
            output_verify="silentfix+ignore",
        )
        logger.info("Created frequency comb master spectrum: %s", self.savefile)

    def load(self):
        """Load master comb from disk

        Returns
        -------
        comb : masked array of shape (nrow, ncol)
            Master comb with bad pixel map applied
        chead : FITS header
            Master comb FITS header
        """
        with fits.open(self.savefile, memmap=False) as hdu:
            comb, chead = hdu[0].data, hdu[0].header
        logger.info("Frequency comb master spectrum: %s", self.savefile)
        return comb, chead


class LaserFrequencyCombFinalize(Step):
    """Improve the precision of the wavelength calibration with a laser frequency comb"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["freq_comb_master", "trace", "wavecal"]

        #:tuple(int, int): polynomial degree of the wavelength fit
        self.degree = config["degree"]
        #:float: residual threshold in m/s above which to remove lines
        self.threshold = config["threshold"]
        #:{'1D', '2D'}: Whether to use 1D or 2D polynomials
        self.dimensionality = config["dimensionality"]
        self.nstep = config["nstep"]
        #:int: Width of the peaks for finding them in the spectrum
        self.lfc_peak_width = config["lfc_peak_width"]

    def run(self, freq_comb_master, trace: list, wavecal: dict):
        """Improve the wavelength calibration with a laser frequency comb.

        Updates trace objects in-place with improved wavelength polynomial.

        Parameters
        ----------
        freq_comb_master : tuple
            extracted frequency comb spectrum and header
        trace : list[TraceData]
            Trace objects with wavelength polynomials from wavecal
        wavecal : dict[str, LineList]
            {group: linelist} from wavecal step (for diagnostics)
        """
        comb, chead = freq_comb_master

        # Get base wavelengths from traces
        wlen = wavelengths_from_traces(trace)
        if wlen is None:
            raise ValueError("No wavelength data in traces - run wavecal first")

        # Get linelist (use first group's linelist for now)
        linelist = next(iter(wavecal.values()))

        module = WavelengthCalibrationComb(
            plot=self.plot,
            plot_title=self.plot_title,
            degree=self.degree,
            threshold=self.threshold,
            dimensionality=self.dimensionality,
            nstep=self.nstep,
            lfc_peak_width=self.lfc_peak_width,
        )
        coef = module.execute(comb, wlen, linelist)

        # Evaluate the full wavelength image (handles step corrections)
        new_wave = module.make_wave(coef)

        # Fit per-trace 1D polynomials to the evaluated wavelengths
        ncol = new_wave.shape[1]
        x = np.arange(ncol)
        poly_degree = (
            self.degree[0] if isinstance(self.degree, (list, tuple)) else self.degree
        )
        for i, t in enumerate(trace):
            cr = t.column_range
            x_cr = x[cr[0] : cr[1]]
            w_cr = new_wave[i, cr[0] : cr[1]]
            deg = min(poly_degree, len(x_cr) - 1)
            t.wave = np.polyfit(x_cr, w_cr, deg=deg)

        self.save(trace)

    def save(self, trace: list):
        """Save updated traces to disk.

        Parameters
        ----------
        trace : list[TraceData]
            Already-updated trace objects
        """
        trace_file = join(self.output_dir, self.prefix + ".traces.fits")
        try:
            header = None
            if os.path.exists(trace_file):
                with fits.open(trace_file, memmap=False) as hdu:
                    header = hdu[0].header
            if header is None:
                header = fits.Header()
            steps = header.get("E_STEPS", "trace").split(",")
            if "freq_comb" not in steps:
                steps.append("freq_comb")
            save_traces(trace_file, trace, header, steps=steps)
            logger.info("Updated traces with freq_comb wavelength: %s", trace_file)
        except Exception as e:
            logger.warning("Could not update traces.fits with freq_comb: %s", e)

    def load(self):
        """Load is a no-op - wavelengths are in traces.fits."""
        # Nothing to load - downstream steps get wavelengths from traces
        pass


class SlitCurvatureDetermination(CalibrationStep, ExtractionStep):
    """Determine the curvature of the slit"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        # No additional dependencies beyond CalibrationStep and ExtractionStep

        #:float: how many sigma of bad lines to cut away
        self.sigma_cutoff = config["curvature_cutoff"]
        #:float: extraction height for peak finding spectrum
        self.extraction_height = config["extraction_height"]
        #:float: height of the 2D cutout for curvature fitting
        self.curve_height = config["curve_height"]
        #:int: Polynomial degree of the overall fit
        self.fit_degree = config["degree"]
        #:int: Orders of the curvature to fit, currently supports only 1 and 2
        self.curve_degree = config["curve_degree"]
        #:{'1D', '2D'}: Whether to use 1d or 2d polynomials
        self.curvature_mode = config["dimensionality"]
        #:float: peak finding noise threshold
        self.peak_threshold = config["peak_threshold"]
        #:int: peak width
        self.peak_width = config["peak_width"]
        #:float: window width to search for peak in each row
        self.window_width = config["window_width"]
        #:str: Function shape that is fit to individual peaks
        self.peak_function = config["peak_function"]

    def run(self, files, trace: list[TraceData], mask=None, bias=None):
        """Determine the curvature of the slit

        Parameters
        ----------
        files : list(str)
            files to use for this
        trace : list[TraceData]
            Trace objects from trace step
        mask : array of shape (nrow, ncol), optional
            Bad pixel mask
        bias : tuple, optional
            Master bias

        Returns
        -------
        curvature : SlitCurvature
            Slit curvature data including polynomial coefficients
        """

        logger.info("Slit curvature files: %s", files)

        orig, thead = self.calibrate(files, mask, bias, None)

        # Apply fiber selection based on instrument config
        selected = self._select_traces(trace, "curvature")
        trace_list = [t for traces in selected.values() for t in traces]

        module = CurvatureModule(
            trace_list,
            curve_height=self.curve_height,
            extraction_height=self.extraction_height,
            trace_range=self.trace_range,
            fit_degree=self.fit_degree,
            curve_degree=self.curve_degree,
            sigma_cutoff=self.sigma_cutoff,
            mode=self.curvature_mode,
            peak_threshold=self.peak_threshold,
            peak_width=self.peak_width,
            window_width=self.window_width,
            peak_function=self.peak_function,
            plot=self.plot,
            plot_title=self.plot_title,
        )
        curvature = module.execute(orig)

        # Update traces in-place with curvature data
        fitted_coeffs = curvature["fitted_coeffs"]
        slitdeltas = curvature["slitdeltas"]
        for i, t in enumerate(trace_list):
            if fitted_coeffs is not None and i < fitted_coeffs.shape[0]:
                t.slit = fitted_coeffs[i]
            if slitdeltas is not None and i < slitdeltas.shape[0]:
                t.slitdelta = slitdeltas[i]

        self.save(trace_list)
        return trace_list

    def save(self, traces):
        """Save curvature results by updating traces.fits.

        Parameters
        ----------
        traces : list[Trace]
            Traces with updated slit/slitdelta data
        """
        trace_file = join(self.output_dir, self.prefix + ".traces.fits")
        if os.path.exists(trace_file):
            try:
                trace_objects, header = load_traces(trace_file)

                # Update each trace with slit data from fitted traces
                # Match by (m, group) since traces may be a filtered subset
                fitted = {(t.m, t.group): t for t in traces}
                for t in trace_objects:
                    match = fitted.get((t.m, t.group))
                    if match is not None:
                        t.slit = match.slit
                        t.slitdelta = match.slitdelta

                # Save updated traces
                steps = header.get("E_STEPS", "trace").split(",")
                if "curvature" not in steps:
                    steps.append("curvature")
                save_traces(trace_file, trace_objects, header, steps=steps)
                logger.info("Updated traces with curvature data: %s", trace_file)
            except Exception as e:
                logger.warning("Could not update traces.fits with curvature: %s", e)

    def load(self):
        """Curvature is now stored in traces, not separate files."""
        return None


class RectifyImage(Step):
    """Create a 2D image of the rectified orders"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["files", "trace", "mask"]
        # self._loadDependsOn += []

        self.extraction_height = config["extraction_height"]
        self.input_files = config["input_files"]

    def filename(self, name):
        if self.channel:
            ext = f".{self.channel.lower()}.rectify.fits"
        else:
            ext = ".rectify.fits"
        return util.swap_extension(name, ext, path=self.output_dir)

    def run(self, files, trace: list[TraceData], mask=None):
        # Get wavelengths from traces (includes freq_comb improvements if run)
        wave = wavelengths_from_traces(trace)

        files = files[self.input_files]

        rectified = {}
        for fname in tqdm(files, desc="Files"):
            img, head = self.instrument.load_fits(
                fname, self.channel, mask=mask, dtype="f8"
            )

            images, cr, xwd = rectify_image(
                img,
                trace,
                self.extraction_height,
                self.trace_range,
            )
            wavelength, image = merge_images(images, wave, cr, xwd)

            self.save(fname, image, wavelength, header=head)
            rectified[fname] = (wavelength, image)

        return rectified

    def save(self, fname, image, wavelength, header=None):
        # Change filename
        fname = self.filename(fname)
        # Create HDU List, one extension per order
        primary = fits.PrimaryHDU(header=header)
        secondary = fits.ImageHDU(data=image)
        column = fits.Column(name="wavelength", array=wavelength, format="D")
        tertiary = fits.BinTableHDU.from_columns([column])
        hdus = fits.HDUList([primary, secondary, tertiary])
        # Save data to file
        hdus.writeto(fname, overwrite=True, output_verify="silentfix")

    def load(self, files):
        files = files[self.input_files]

        rectified = {}
        for orig_fname in files:
            fname = self.filename(orig_fname)
            with fits.open(fname, memmap=False) as hdu:
                img = hdu[1].data
                wave = hdu[2].data["wavelength"]
            rectified[orig_fname] = (wave, img)

        return rectified


class ScienceExtraction(CalibrationStep, ExtractionStep):
    """Extract the science spectra"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["norm_flat", "scatter"]

    def science_file(self, name):
        """Name of the science file in disk, based on the input file

        Parameters
        ----------
        name : str
            name of the observation file

        Returns
        -------
        name : str
            science file name
        """
        if self.channel:
            ext = f".{self.channel.lower()}.science.fits"
        else:
            ext = ".science.fits"
        return util.swap_extension(name, ext, path=self.output_dir)

    def run(
        self,
        files,
        trace: list[TraceData],
        bias=None,
        norm_flat=None,
        scatter=None,
        mask=None,
    ):
        """Extract Science spectra from observation

        Parameters
        ----------
        files : list(str)
            list of observations
        trace : list[TraceData]
            Trace objects from trace step
        bias : tuple, optional
            results from master bias step
        norm_flat : tuple, optional
            results from flat normalization
        scatter : array, optional
            background scatter model
        mask : array of shape (nrow, ncol), optional
            bad pixel map

        Returns
        -------
        heads : list(FITS header)
            FITS headers of each observation
        spectra_list : list(list[Spectrum])
            extracted spectra (one list per file)
        """
        # Apply fiber selection based on instrument config
        selected = self._select_traces(trace, "science")
        trace_list = [t for traces in selected.values() for t in traces]

        # Apply trace_range if specified
        if self.trace_range is not None:
            trace_list = trace_list[self.trace_range[0] : self.trace_range[1]]

        # Extraction parameters
        extraction_kwargs = dict(self.extraction_kwargs)
        default_height = extraction_kwargs.pop("extraction_height", 0.5)

        heads, all_spectra = [], []
        for fname in tqdm(files, desc="Files"):
            logger.info("Science file: %s", fname)

            # Calibrate the input image
            im, head = self.calibrate(
                [fname],
                mask,
                bias,
                norm_flat,
                traces=trace_list,
                extraction_height=default_height,
            )

            # Extract science spectrum - returns list[Spectrum]
            spectra = extract(
                im,
                trace_list,
                extraction_height=default_height,
                extraction_type=self.extraction_method,
                gain=head["e_gain"],
                readnoise=head["e_readn"],
                dark=head["e_drk"],
                scatter=scatter,
                plot=self.plot,
                plot_title=self.plot_title,
                **extraction_kwargs,
            )

            # Save spectrum to disk
            self.save(fname, head, spectra)
            heads.append(head)
            all_spectra.append(spectra)

        return heads, all_spectra

    def save(self, fname, head, spectra: list[Spectrum]):
        """Save extracted spectra using Spectra format.

        Parameters
        ----------
        fname : str
            Original filename (used to derive output name)
        head : FITS header
            FITS header
        spectra : list[Spectrum]
            Extracted spectra from extract()
        """
        nameout = self.science_file(fname)

        # Create extraction params from settings
        params = ExtractionParams(
            osample=self.extraction_kwargs.get("oversampling", 10),
            lambda_sf=self.extraction_kwargs.get("smooth_slitfunction", 1.0),
            lambda_sp=self.extraction_kwargs.get("smooth_spectrum", 0.0),
            swath_width=self.extraction_kwargs.get("swath_width"),
        )

        spectra_container = Spectra(header=head, data=spectra, params=params)
        spectra_container.save(nameout, steps=["science"])
        logger.info("Created science file: %s", nameout)

    def load(self):
        """Load all science spectra from disk.

        Supports both new Spectra format (E_FMTVER >= 2) and legacy format.

        Returns
        -------
        heads : list(FITS header)
            FITS headers of each observation
        specs : list(array of shape (ntrace, ncol))
            extracted spectra
        sigmas : list(array of shape (ntrace, ncol))
            uncertainties of the extracted spectra
        slitfus : list or None
            slit functions (if available)
        columns : list(array of shape (ntrace, 2))
            column ranges for each spectra
        """
        files = self.files["science"]
        files = [self.science_file(fname) for fname in files]

        if len(files) == 0:
            raise FileNotFoundError("Science files are required to load them")

        logger.info("Science files: %s", files)

        heads, specs, sigmas, slitfus, columns = [], [], [], [], []
        for fname in files:
            # Spectra.read handles both new and legacy formats via E_FMTVER
            spectra = Spectra.read(
                fname,
                raw=True,
                continuum_normalization=False,
                barycentric_correction=False,
                radial_velocity_correction=False,
            )
            heads.append(spectra.header)

            # Stack arrays from Spectrum objects (NaN encodes masked pixels)
            spec_arr = np.ma.masked_invalid([s.spec for s in spectra.data])
            sig_arr = np.ma.masked_invalid([s.sig for s in spectra.data])
            specs.append(spec_arr)
            sigmas.append(sig_arr)

            # Extract column range from NaN masking
            ntrace, ncol = spec_arr.shape
            cr = np.zeros((ntrace, 2), dtype=np.int32)
            for i in range(ntrace):
                valid = ~np.isnan(spec_arr[i])
                if np.any(valid):
                    cr[i, 0] = np.argmax(valid)
                    cr[i, 1] = ncol - np.argmax(valid[::-1])
            columns.append(cr)

            # Extract slit functions
            has_slitfu = any(s.slitfu is not None for s in spectra.data)
            if has_slitfu:
                slitfus.append([s.slitfu for s in spectra.data])
            else:
                slitfus.append(None)

        return heads, specs, sigmas, slitfus, columns


class ContinuumNormalization(Step):
    """Determine the continuum to each observation"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["science", "norm_flat", "trace"]
        self._loadDependsOn += ["norm_flat", "science"]

    @property
    def savefile(self):
        """str: savefile name"""
        return join(self.output_dir, self.prefix + ".cont.npz")

    def run(self, science, norm_flat, trace: list):
        """Determine the continuum to each observation
        Also splices the orders together

        Parameters
        ----------
        science : tuple
            results from science step: (heads, list[list[Spectrum]])
        norm_flat : tuple
            results from the normalized flatfield step
        trace : list[TraceData]
            Trace objects with wavelength polynomials

        Returns
        -------
        heads : list(FITS header)
            FITS headers of each observation
        specs : list(array of shape (ntrace, ncol))
            extracted spectra
        sigmas : list(array of shape (ntrace, ncol))
            uncertainties of the extracted spectra
        conts : list(array of shape (ntrace, ncol))
            continuum for each spectrum
        columns : list(array of shape (ntrace, 2))
            column ranges for each spectra
        """
        norm, blaze, *_ = norm_flat

        # Select same traces as science step (fiber/group selection + trace_range)
        selected = self._select_traces(trace, "science")
        trace_list = [t for traces in selected.values() for t in traces]
        if self.trace_range is not None:
            trace_list = trace_list[self.trace_range[0] : self.trace_range[1]]

        # Handle both old format (5 elements from load) and new format (2 elements from run)
        if len(science) == 2:
            # New Spectrum-based format from science.run()
            heads, spectra_lists = science
            specs = []
            sigmas = []
            columns = []
            for spectra in spectra_lists:
                specs.append(np.ma.masked_invalid([s.spec for s in spectra]))
                sigmas.append(np.ma.masked_invalid([s.sig for s in spectra]))
                columns.append(np.array([[0, len(s.spec)] for s in spectra]))
        else:
            # Old array format from science.load()
            heads, specs, sigmas, _, columns = science

        nspec = specs[0].shape[0]

        # Filter out traces that extraction marked invalid
        valid = [t for t in trace_list if not t.invalid]
        if len(valid) == nspec:
            trace_list = valid
        wave = wavelengths_from_traces(trace_list)

        if wave is None:
            raise ValueError(
                "Continuum normalization requires wavelength data. "
                "Run wavecal or freq_comb steps first."
            )

        # Align all arrays to the smallest count (norm_flat may skip edge traces)
        nmin = min(nspec, len(blaze), len(wave) if wave is not None else nspec)
        if nspec > nmin:
            specs = [s[nspec - nmin :] for s in specs]
            sigmas = [s[nspec - nmin :] for s in sigmas]
            columns = [c[nspec - nmin :] for c in columns]
            nspec = nmin
        if wave is not None and len(wave) > nmin:
            wave = wave[len(wave) - nmin :]
        if len(blaze) > nmin:
            blaze = blaze[len(blaze) - nmin :]

        logger.info("Continuum normalization")
        conts = [None for _ in specs]
        for j, (spec, sigma) in enumerate(zip(specs, sigmas, strict=False)):
            logger.info("Splicing orders")
            specs[j], wave, blaze, sigmas[j] = splice_orders(
                spec,
                wave,
                blaze,
                sigma,
                scaling=True,
                plot=self.plot,
                plot_title=self.plot_title,
            )
            logger.info("Normalizing continuum")
            conts[j] = continuum_normalize(
                specs[j],
                wave,
                blaze,
                sigmas[j],
                plot=self.plot,
                plot_title=self.plot_title,
            )

        self.save(heads, specs, sigmas, conts, columns)
        return heads, specs, sigmas, conts, columns

    def save(self, heads, specs, sigmas, conts, columns):
        """Save the results from the continuum normalization

        Parameters
        ----------
        heads : list(FITS header)
            FITS headers of each observation
        specs : list(array of shape (ntrace, ncol))
            extracted spectra
        sigmas : list(array of shape (ntrace, ncol))
            uncertainties of the extracted spectra
        conts : list(array of shape (ntrace, ncol))
            continuum for each spectrum
        columns : list(array of shape (ntrace, 2))
            column ranges for each spectra
        """
        value = {
            "heads": heads,
            "specs": specs,
            "sigmas": sigmas,
            "conts": conts,
            "columns": columns,
        }
        joblib.dump(value, self.savefile)
        logger.info("Created continuum normalization file: %s", self.savefile)

    def load(self, norm_flat, science):
        """Load the results from the continuum normalization

        Returns
        -------
        heads : list(FITS header)
            FITS headers of each observation
        specs : list(array of shape (ntrace, ncol))
            extracted spectra
        sigmas : list(array of shape (ntrace, ncol))
            uncertainties of the extracted spectra
        conts : list(array of shape (ntrace, ncol))
            continuum for each spectrum
        columns : list(array of shape (ntrace, 2))
            column ranges for each spectra
        """
        try:
            data = joblib.load(self.savefile)
            logger.info("Continuum normalization file: %s", self.savefile)
        except FileNotFoundError:
            # Use science files instead
            logger.warning(
                "No continuum normalized data found. Using unnormalized results instead."
            )
            heads, specs, sigmas, columns = science
            norm, blaze, *_ = norm_flat
            conts = [blaze for _ in specs]
            data = {
                "heads": heads,
                "specs": specs,
                "sigmas": sigmas,
                "conts": conts,
                "columns": columns,
            }
        heads = data["heads"]
        specs = data["specs"]
        sigmas = data["sigmas"]
        conts = data["conts"]
        columns = data["columns"]
        return heads, specs, sigmas, conts, columns


class Finalize(Step):
    """Create the final output files"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["continuum", "trace", "config"]
        self.filename = config["filename"]

    def output_file(self, number, name):
        """str: output file name"""
        out = self.filename.format(
            instrument=self.instrument.name,
            night=self.night,
            channel=self.channel,
            number=number,
            input=name,
        )
        return join(self.output_dir, out)

    def save_config_to_header(self, head, config, prefix="PR"):
        for key, value in config.items():
            if isinstance(value, dict):
                head = self.save_config_to_header(
                    head, value, prefix=f"{prefix} {key.upper()}"
                )
            else:
                if key in ["plot", "$schema", "__skip_existing__"]:
                    # Skip values that are not relevant to the file product
                    continue
                if value is None:
                    value = "null"
                elif not np.isscalar(value):
                    value = str(value)
                head[f"HIERARCH {prefix} {key.upper()}"] = value
        return head

    def run(self, continuum, trace: list, config):
        """Create the final output files

        this is includes:
         - heliocentric corrections
         - creating one echelle file

        Parameters
        ----------
        continuum : tuple
            results from the continuum normalization
        trace : list[TraceData]
            Trace objects with wavelength polynomials
        config : dict
            Pipeline configuration
        """
        heads, specs, sigmas, conts, columns = continuum

        # Select same traces as science/continuum steps
        selected = self._select_traces(trace, "science")
        trace_list = [t for traces in selected.values() for t in traces]
        if self.trace_range is not None:
            trace_list = trace_list[self.trace_range[0] : self.trace_range[1]]
        valid = [t for t in trace_list if not t.invalid]
        nspec = specs[0].shape[0]
        if len(valid) == nspec:
            trace_list = valid
        wave = wavelengths_from_traces(trace_list)
        if wave is not None and len(wave) > nspec:
            wave = wave[len(wave) - nspec :]

        fnames = []
        # Combine science with wavecal and continuum
        for i, (head, spec, sigma, blaze, column) in enumerate(
            zip(heads, specs, sigmas, conts, columns, strict=False)
        ):
            head["e_erscle"] = ("absolute", "error scale")

            # Add heliocentric correction
            try:
                rv_corr, bjd = util.helcorr(
                    head["e_obslon"],
                    head["e_obslat"],
                    head["e_obsalt"],
                    head["e_ra"],
                    head["e_dec"],
                    head["e_jd"],
                )

                logger.debug("Heliocentric correction: %f km/s", rv_corr)
                logger.debug("Heliocentric Julian Date: %s", str(bjd))
            except KeyError:
                logger.warning("Could not calculate heliocentric correction")
                # logger.warning("Telescope is in space?")
                rv_corr = 0
                bjd = head["e_jd"]

            head["barycorr"] = rv_corr
            head["e_jd"] = bjd
            head["HIERARCH PR_version"] = __version__

            head = self.save_config_to_header(head, config)

            if self.plot:
                plt.figure()
                plt.plot(wave.T, (spec / blaze).T)
                if self.plot_title is not None:
                    plt.title(self.plot_title)
                util.show_or_save(f"finalize_{i}")

            fname = self.save(i, head, spec, sigma, blaze, wave, column)
            fnames.append(fname)
        return fnames

    def save(self, i, head, spec, sigma, cont, wave, columns):
        """Save one output spectrum to disk

        Parameters
        ----------
        i : int
            individual number of each file
        head : FITS header
            FITS header
        spec : array of shape (ntrace, ncol)
            final spectrum
        sigma : array of shape (ntrace, ncol)
            final uncertainties
        cont : array of shape (ntrace, ncol)
            final continuum scales
        wave : array of shape (ntrace, ncol)
            wavelength solution
        columns : array of shape (ntrace, 2)
            columns that carry signal

        Returns
        -------
        out_file : str
            name of the output file
        """
        original_name = os.path.splitext(head["e_input"])[0]
        out_file = self.output_file(i, original_name)

        ntrace = spec.shape[0]

        # Convert arrays to list[Spectrum], masking outside column range with NaN
        spectra_list = []
        for j in range(ntrace):
            spec_row = np.array(spec[j], dtype=np.float32)
            sig_row = np.array(sigma[j], dtype=np.float32)
            wave_row = np.array(wave[j], dtype=np.float64) if wave is not None else None
            cont_row = np.array(cont[j], dtype=np.float32) if cont is not None else None

            # Apply column mask as NaN
            if columns is not None:
                spec_row[: columns[j, 0]] = np.nan
                spec_row[columns[j, 1] :] = np.nan
                sig_row[: columns[j, 0]] = np.nan
                sig_row[columns[j, 1] :] = np.nan

            spectra_list.append(
                Spectrum(
                    m=j,
                    spec=spec_row,
                    sig=sig_row,
                    wave=wave_row,
                    cont=cont_row,
                )
            )

        spectra = Spectra(header=head, data=spectra_list)
        spectra.save(out_file, steps=["finalize"])
        logger.info("Final science file: %s", out_file)
        return out_file
