# -*- coding: utf-8 -*-
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

import glob
import logging
import os.path
from itertools import product
from os.path import dirname, join

import joblib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from genericpath import exists
from tqdm import tqdm

from pyreduce import wavelength_calibration

# PyReduce subpackages
from . import __version__, echelle, instruments, util
from .combine_frames import (
    combine_bias,
    combine_calibrate,
    combine_frames,
    combine_polynomial,
)
from .configuration import load_config
from .continuum_normalization import continuum_normalize, splice_orders
from .estimate_background_scatter import estimate_background_scatter
from .extract import extract
from .extraction_width import estimate_extraction_width
from .instruments.instrument_info import load_instrument
from .make_shear import Curvature as CurvatureModule
from .rectify import merge_images, rectify_image
from .trace_orders import mark_orders
from .wavelength_calibration import LineList
from .wavelength_calibration import WavelengthCalibration as WavelengthCalibrationModule
from .wavelength_calibration import WavelengthCalibrationComb
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
    modes=None,
    steps="all",
    base_dir=None,
    input_dir=None,
    output_dir=None,
    configuration=None,
    order_range=None,
    allow_calibration_only=False,
    skip_existing=False,
):
    r"""
    Main entry point for REDUCE scripts,
    default values can be changed as required if reduce is used as a script
    Finds input directories, and loops over observation nights and instrument modes

    Parameters
    ----------
    instrument : str, list[str]
        instrument used for the observation (e.g. UVES, HARPS)
    target : str, list[str]
        the observed star, as named in the folder structure/fits headers
    night : str, list[str]
        the observation nights to reduce, as named in the folder structure. Accepts bash wildcards (i.e. \*, ?), but then relies on the folder structure for restricting the nights
    modes : str, list[str], dict[{instrument}:list], None, optional
        the instrument modes to use, if None will use all known modes for the current instrument. See instruments for possible options
    steps : tuple(str), "all", optional
        which steps of the reduction process to perform
        the possible steps are: "bias", "flat", "orders", "norm_flat", "wavecal", "science"
        alternatively set steps to "all", which is equivalent to setting all steps
        Note that the later steps require the previous intermediary products to exist and raise an exception otherwise
    base_dir : str, optional
        base data directory that Reduce should work in, is prefixxed on input_dir and output_dir (default: use settings_pyreduce.json)
    input_dir : str, optional
        input directory containing raw files. Can contain placeholders {instrument}, {target}, {night}, {mode} as well as wildcards. If relative will use base_dir as root (default: use settings_pyreduce.json)
    output_dir : str, optional
        output directory for intermediary and final results. Can contain placeholders {instrument}, {target}, {night}, {mode}, but no wildcards. If relative will use base_dir as root (default: use settings_pyreduce.json)
    configuration : dict[str:obj], str, list[str], dict[{instrument}:dict,str], optional
        configuration file for the current run, contains parameters for different parts of reduce. Can be a path to a json file, or a dict with configurations for the different instruments. When a list, the order must be the same as instruments (default: settings_{instrument.upper()}.json)
    """
    if target is None or np.isscalar(target):
        target = [target]
    if night is None or np.isscalar(night):
        night = [night]

    isNone = {
        "modes": modes is None,
        "base_dir": base_dir is None,
        "input_dir": input_dir is None,
        "output_dir": output_dir is None,
    }
    output = []

    # Loop over everything

    # settings: default settings of PyReduce
    # config: paramters for the current reduction
    # info: constant, instrument specific parameters
    config = load_config(configuration, instrument, 0)
    if isinstance(instrument, str):
        instrument = instruments.instrument_info.load_instrument(instrument)
    info = instrument.info

    # load default settings from settings_pyreduce.json
    if base_dir is None:
        base_dir = config["reduce"]["base_dir"]
    if input_dir is None:
        input_dir = config["reduce"]["input_dir"]
    if output_dir is None:
        output_dir = config["reduce"]["output_dir"]

    input_dir = join(base_dir, input_dir)
    output_dir = join(base_dir, output_dir)

    if modes is None:
        modes = info["modes"]
    if np.isscalar(modes):
        modes = [modes]

    for t, n, m in product(target, night, modes):
        log_file = join(
            base_dir.format(instrument=str(instrument), mode=modes, target=t),
            "logs/%s.log" % t,
        )
        util.start_logging(log_file)
        # find input files and sort them by type
        files = instrument.sort_files(
            input_dir,
            t,
            n,
            mode=m,
            **config["instrument"],
            allow_calibration_only=allow_calibration_only,
        )
        if len(files) == 0:
            logger.warning(
                f"No files found for instrument: %s, target: %s, night: %s, mode: %s in folder: %s",
                instrument,
                t,
                n,
                m,
                input_dir,
            )
            continue
        for k, f in files:
            logger.info("Settings:")
            for key, value in k.items():
                logger.info("%s: %s", key, value)
            logger.debug("Files:\n%s", f)

            reducer = Reducer(
                f,
                output_dir,
                k.get("target"),
                instrument,
                m,
                k.get("night"),
                config,
                order_range=order_range,
                skip_existing=skip_existing,
            )
            # try:
            data = reducer.run_steps(steps=steps)
            output.append(data)
            # except Exception as e:
            #     logger.error("Reduction failed with error message: %s", str(e))
            #     logger.info("------------")
    return output


class Step:
    """Parent class for all steps"""

    def __init__(
        self, instrument, mode, target, night, output_dir, order_range, **config
    ):
        self._dependsOn = []
        self._loadDependsOn = []
        #:str: Name of the instrument
        self.instrument = instrument
        #:str: Name of the instrument mode
        self.mode = mode
        #:str: Name of the observation target
        self.target = target
        #:str: Date of the observation (as a string)
        self.night = night
        #:tuple(int, int): First and Last(+1) order to process
        self.order_range = order_range
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
        """str: output directory, may contain tags {instrument}, {night}, {target}, {mode}"""
        return self._output_dir.format(
            instrument=self.instrument.name.upper(),
            target=self.target,
            night=self.night,
            mode=self.mode,
        )

    @property
    def prefix(self):
        """str: temporary file prefix"""
        i = self.instrument.name.lower()
        if self.mode is not None and self.mode != "":
            m = self.mode.lower()
            return f"{i}_{m}"
        else:
            return i


class CalibrationStep(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["mask", "bias"]

        #:{'number_of_files', 'exposure_time', 'mean', 'median', 'none'}: how to adjust for diferences between the bias and flat field exposure times
        self.bias_scaling = config["bias_scaling"]
        #:{'divide', 'none'}: how to apply the normalized flat field
        self.norm_scaling = config["norm_scaling"]

    def calibrate(self, files, mask, bias=None, norm_flat=None):
        bias, bhead = bias if bias is not None else (None, None)
        norm, blaze = norm_flat if norm_flat is not None else (None, None)
        orig, thead = combine_calibrate(
            files,
            self.instrument,
            self.mode,
            mask,
            bias=bias,
            bhead=bhead,
            norm=norm,
            bias_scaling=self.bias_scaling,
            norm_scaling=self.norm_scaling,
            plot=self.plot,
            plot_title=self.plot_title,
        )

        return orig, thead


class ExtractionStep(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += [
            "orders",
        ]

        #:{'arc', 'optimal'}: Extraction method to use
        self.extraction_method = config["extraction_method"]
        if self.extraction_method == "arc":
            #:dict: arguments for the extraction
            self.extraction_kwargs = {
                "extraction_width": config["extraction_width"],
                "sigma_cutoff": config["extraction_cutoff"],
                "collapse_function": config["collapse_function"],
            }
        elif self.extraction_method == "optimal":
            self.extraction_kwargs = {
                "extraction_width": config["extraction_width"],
                "lambda_sf": config["smooth_slitfunction"],
                "lambda_sp": config["smooth_spectrum"],
                "osample": config["oversampling"],
                "swath_width": config["swath_width"],
                "sigma_cutoff": config["extraction_cutoff"],
                "maxiter": config["maxiter"],
            }
        else:
            raise ValueError(
                f"Extraction method {self.extraction_method} not supported for step 'wavecal'"
            )

    def extract(self, img, head, orders, curvature):
        orders, column_range = orders if orders is not None else (None, None)
        tilt, shear = curvature if curvature is not None else (None, None)

        data, unc, blaze, cr = extract(
            img,
            orders,
            gain=head["e_gain"],
            readnoise=head["e_readn"],
            dark=head["e_drk"],
            column_range=column_range,
            extraction_type=self.extraction_method,
            order_range=self.order_range,
            plot=self.plot,
            plot_title=self.plot_title,
            tilt=tilt,
            shear=shear,
            **self.extraction_kwargs,
        )
        return data, unc, blaze, cr


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
            data = fits.open(self.savefile)[0]
            data, head = data.data, data.header
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
    """Load the bad pixel mask for the given instrument/mode"""

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
        mask_file = self.instrument.get_mask_filename(mode=self.mode)
        try:
            mask, _ = self.instrument.load_fits(mask_file, self.mode, extension=0)
            mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks
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

    def run(self, files, mask):
        """Calculate the master bias

        Parameters
        ----------
        files : list(str)
            bias files
        mask : array of shape (nrow, ncol)
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
                self.mode,
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
                self.mode,
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
            hdus = fits.PrimaryHDU(data=bias, header=bhead)
        else:
            hdus = [fits.PrimaryHDU(data=bias[0], header=bhead)]
            for i in range(1, len(bias)):
                hdus += [fits.ImageHDU(data=bias[i])]
            hdus = fits.HDUList(hdus)

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
            hdu = fits.open(self.savefile)
            degree = len(hdu) - 1
            if degree == 0:
                bias, bhead = hdu[0].data, hdu[0].header
                bias = np.ma.masked_array(bias, mask=mask)
            else:
                bhead = hdu[0].header
                bias = np.array([h.data for h in hdu])
                bias = np.ma.masked_array(bias, mask=[mask for _ in range(len(hdu))])
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

    def run(self, files, bias, mask):
        """Calculate the master flat, with the bias already subtracted

        Parameters
        ----------
        files : list(str)
            flat files
        bias : tuple(array of shape (nrow, ncol), FITS header)
            master bias and header
        mask : array of shape (nrow, ncol)
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
            flat = fits.open(self.savefile)[0]
            flat, fhead = flat.data, flat.header
            flat = np.ma.masked_array(flat, mask=mask)
            logger.info("Master flat file: %s", self.savefile)
        except FileNotFoundError:
            logger.warning(
                "No intermediate file for the flat field found. Using Flat = 1 instead"
            )
            flat, fhead = None, None
        return flat, fhead


class OrderTracing(CalibrationStep):
    """Determine the polynomial fits describing the pixel locations of each order"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)

        #:int: Minimum size of each cluster to be included in further processing
        self.min_cluster = config["min_cluster"]
        #:int, float: Minimum width of each cluster after mergin
        self.min_width = config["min_width"]
        #:int: Size of the gaussian filter for smoothing
        self.filter_size = config["filter_size"]
        #:int: Background noise value threshold
        self.noise = config["noise"]
        #:int: Polynomial degree of the fit to each order
        self.fit_degree = config["degree"]

        self.degree_before_merge = config["degree_before_merge"]
        self.regularization = config["regularization"]
        self.closing_shape = config["closing_shape"]
        self.auto_merge_threshold = config["auto_merge_threshold"]
        self.merge_min_threshold = config["merge_min_threshold"]
        self.sigma = config["split_sigma"]
        #:int: Number of pixels at the edge of the detector to ignore
        self.border_width = config["border_width"]
        #:bool: Whether to use manual alignment
        self.manual = config["manual"]

    @property
    def savefile(self):
        """str: Name of the order tracing file"""
        return join(self.output_dir, self.prefix + ".ord_default.npz")

    def run(self, files, mask, bias):
        """Determine polynomial coefficients describing order locations

        Parameters
        ----------
        files : list(str)
            Observation used for order tracing (should only have one element)
        mask : array of shape (nrow, ncol)
            Bad pixel mask

        Returns
        -------
        orders : array of shape (nord, ndegree+1)
            polynomial coefficients for each order
        column_range : array of shape (nord, 2)
            first and last(+1) column that carries signal in each order
        """

        logger.info("Order tracing files: %s", files)

        order_img, ohead = self.calibrate(files, mask, bias, None)

        orders, column_range = mark_orders(
            order_img,
            min_cluster=self.min_cluster,
            min_width=self.min_width,
            filter_size=self.filter_size,
            noise=self.noise,
            opower=self.fit_degree,
            degree_before_merge=self.degree_before_merge,
            regularization=self.regularization,
            closing_shape=self.closing_shape,
            border_width=self.border_width,
            manual=self.manual,
            auto_merge_threshold=self.auto_merge_threshold,
            merge_min_threshold=self.merge_min_threshold,
            sigma=self.sigma,
            plot=self.plot,
            plot_title=self.plot_title,
        )

        self.save(orders, column_range)

        return orders, column_range

    def save(self, orders, column_range):
        """Save order tracing results to disk

        Parameters
        ----------
        orders : array of shape (nord, ndegree+1)
            polynomial coefficients
        column_range : array of shape (nord, 2)
            first and last(+1) column that carry signal in each order
        """
        np.savez(self.savefile, orders=orders, column_range=column_range)
        logger.info("Created order tracing file: %s", self.savefile)

    def load(self):
        """Load order tracing results

        Returns
        -------
        orders : array of shape (nord, ndegree+1)
            polynomial coefficients for each order
        column_range : array of shape (nord, 2)
            first and last(+1) column that carries signal in each order
        """
        logger.info("Order tracing file: %s", self.savefile)
        data = np.load(self.savefile, allow_pickle=True)
        orders = data["orders"]
        column_range = data["column_range"]
        return orders, column_range


class BackgroundScatter(CalibrationStep):
    """Determine the background scatter"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["orders"]

        #:tuple(int, int): Polynomial degrees for the background scatter fit, in row, column direction
        self.scatter_degree = config["scatter_degree"]
        self.extraction_width = config["extraction_width"]
        self.sigma_cutoff = config["scatter_cutoff"]
        self.border_width = config["border_width"]

    @property
    def savefile(self):
        """str: Name of the scatter file"""
        return join(self.output_dir, self.prefix + ".scatter.npz")

    def run(self, files, mask, bias, orders):
        logger.info("Background scatter files: %s", files)

        scatter_img, shead = self.calibrate(files, mask, bias)

        orders, column_range = orders
        scatter = estimate_background_scatter(
            scatter_img,
            orders,
            column_range=column_range,
            extraction_width=self.extraction_width,
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
        self._dependsOn += ["flat", "orders", "scatter", "curvature"]

        #:{'normalize'}: Extraction method to use
        self.extraction_method = config["extraction_method"]
        if self.extraction_method == "normalize":
            #:dict: arguments for the extraction
            self.extraction_kwargs = {
                "extraction_width": config["extraction_width"],
                "lambda_sf": config["smooth_slitfunction"],
                "lambda_sp": config["smooth_spectrum"],
                "osample": config["oversampling"],
                "swath_width": config["swath_width"],
                "sigma_cutoff": config["extraction_cutoff"],
                "maxiter": config["maxiter"],
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

    def run(self, flat, orders, scatter, curvature):
        """Calculate the 'normalized' flat field

        Parameters
        ----------
        flat : tuple(array, header)
            Master flat, and its FITS header
        orders : tuple(array, array)
            Polynomial coefficients for each order, and the first and last(+1) column containing signal

        Returns
        -------
        norm : array of shape (nrow, ncol)
            normalized flat field
        blaze : array of shape (nord, ncol)
            Continuum level as determined from the flat field for each order
        """
        flat, fhead = flat
        orders, column_range = orders
        tilt, shear = curvature

        # if threshold is smaller than 1, assume percentage value is given
        if self.threshold <= 1:
            threshold = np.percentile(flat, self.threshold * 100)
        else:
            threshold = self.threshold

        norm, _, blaze, _ = extract(
            flat,
            orders,
            gain=fhead["e_gain"],
            readnoise=fhead["e_readn"],
            dark=fhead["e_drk"],
            order_range=self.order_range,
            column_range=column_range,
            scatter=scatter,
            threshold=threshold,
            threshold_lower=self.threshold_lower,
            extraction_type=self.extraction_method,
            tilt=tilt,
            shear=shear,
            plot=self.plot,
            plot_title=self.plot_title,
            **self.extraction_kwargs,
        )

        blaze = np.ma.filled(blaze, 0)
        norm = np.nan_to_num(norm, nan=1)
        self.save(norm, blaze)
        return norm, blaze

    def save(self, norm, blaze):
        """Save normalized flat field results to disk

        Parameters
        ----------
        norm : array of shape (nrow, ncol)
            normalized flat field
        blaze : array of shape (nord, ncol)
            Continuum level as determined from the flat field for each order
        """
        np.savez(self.savefile, blaze=blaze, norm=norm)
        logger.info("Created normalized flat file: %s", self.savefile)

    def load(self):
        """Load normalized flat field results from disk

        Returns
        -------
        norm : array of shape (nrow, ncol)
            normalized flat field
        blaze : array of shape (nord, ncol)
            Continuum level as determined from the flat field for each order
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
        return norm, blaze


class WavelengthCalibrationMaster(CalibrationStep, ExtractionStep):
    """Create wavelength calibration master image"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["norm_flat", "curvature"]

    @property
    def savefile(self):
        """str: Name of the wavelength echelle file"""
        return join(self.output_dir, self.prefix + ".thar_master.fits")

    def run(self, files, orders, mask, curvature, bias, norm_flat):
        """Perform wavelength calibration

        This consists of extracting the wavelength image
        and fitting a polynomial the the known spectral lines

        Parameters
        ----------
        files : list(str)
            wavelength calibration files
        orders : tuple(array, array)
            Polynomial coefficients of each order, and columns with signal of each order
        mask : array of shape (nrow, ncol)
            Bad pixel mask

        Returns
        -------
        wave : array of shape (nord, ncol)
            wavelength for each point in the spectrum
        thar : array of shape (nrow, ncol)
            extracted wavelength calibration image
        coef : array of shape (*ndegrees,)
            polynomial coefficients of the wavelength fit
        linelist : record array of shape (nlines,)
            Updated line information for all lines
        """
        if len(files) == 0:
            raise FileNotFoundError("No files found for wavelength calibration")
        logger.info("Wavelength calibration files: %s", files)
        # Load wavecal image
        orig, thead = self.calibrate(files, mask, bias, norm_flat)
        # Extract wavecal spectrum
        thar, _, _, _ = self.extract(orig, thead, orders, curvature)
        self.save(thar, thead)
        return thar, thead

    def save(self, thar, thead):
        """Save the master wavelength calibration to a FITS file

        Parameters
        ----------
        thar : array of shape (nrow, ncol)
            master flat data
        thead : FITS header
            master flat header
        """
        thar = np.asarray(thar, dtype=np.float64)
        fits.writeto(
            self.savefile,
            data=thar,
            header=thead,
            overwrite=True,
            output_verify="silentfix+ignore",
        )
        logger.info("Created wavelength calibration spectrum file: %s", self.savefile)

    def load(self):
        """Load master wavelength calibration from disk

        Returns
        -------
        thar : masked array of shape (nrow, ncol)
            Master wavecal with bad pixel map applied
        thead : FITS header
            Master wavecal FITS header
        """
        thar = fits.open(self.savefile)[0]
        thar, thead = thar.data, thar.header
        logger.info("Wavelength calibration spectrum file: %s", self.savefile)
        return thar, thead


class WavelengthCalibrationInitialize(Step):
    """Create the initial wavelength solution file"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["wavecal_master"]
        self._loadDependsOn += ["config", "wavecal_master"]

        #:tuple(int, int): Polynomial degree of the wavelength calibration in order, column direction
        self.degree = config["degree"]
        #:float: wavelength range around the initial guess to explore
        self.wave_delta = config["wave_delta"]
        #:int: number of walkers in the MCMC
        self.nwalkers = config["nwalkers"]
        #:int: number of steps in the MCMC
        self.steps = config["steps"]
        #:float: resiudal range to accept as match between peaks and atlas in m/s
        self.resid_delta = config["resid_delta"]
        #:str: element for the atlas to use
        self.element = config["element"]
        #:str: medium the medium of the instrument, air or vac
        self.medium = config["medium"]
        #:float: Gaussian smoothing parameter applied to the observed spectrum in pixel scale, set to 0 to disable smoothing
        self.smoothing = config["smoothing"]
        #:float: Minimum height of spectral lines in the normalized spectrum, values of 1 and above are interpreted as percentiles of the spectrum, set to 0 to disable the cutoff
        self.cutoff = config["cutoff"]

    @property
    def savefile(self):
        """str: Name of the wavelength echelle file"""
        return join(self.output_dir, self.prefix + ".linelist.npz")

    def run(self, wavecal_master):
        thar, thead = wavecal_master

        # Get the initial wavelength guess from the instrument
        wave_range = self.instrument.get_wavelength_range(thead, self.mode)
        if wave_range is None:
            raise ValueError(
                "This instrument is missing an initial wavelength guess for wavecal_init"
            )

        module = WavelengthCalibrationInitializeModule(
            plot=self.plot,
            plot_title=self.plot_title,
            degree=self.degree,
            wave_delta=self.wave_delta,
            nwalkers=self.nwalkers,
            steps=self.steps,
            resid_delta=self.resid_delta,
            element=self.element,
            medium=self.medium,
            smoothing=self.smoothing,
            cutoff=self.cutoff,
        )
        linelist = module.execute(thar, wave_range)
        self.save(linelist)
        return linelist

    def save(self, linelist):
        linelist.save(self.savefile)
        logger.info("Created wavelength calibration linelist file: %s", self.savefile)

    def load(self, config, wavecal_master):
        thar, thead = wavecal_master
        try:
            # Try loading the custom reference file
            reference = self.savefile
            linelist = LineList.load(reference)
        except FileNotFoundError:
            # If that fails, load the file provided by PyReduce
            # It usually fails because we want to use this one
            reference = self.instrument.get_wavecal_filename(
                thead, self.mode, **config["instrument"]
            )

            # This should fail if there is no provided file by PyReduce
            linelist = LineList.load(reference)
        logger.info("Wavelength calibration linelist file: %s", reference)
        return linelist


class WavelengthCalibrationFinalize(Step):
    """Perform wavelength calibration"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["wavecal_master", "wavecal_init"]

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
        #:float: fraction of columns, to allow individual orders to shift
        self.shift_window = config["shift_window"]
        #:str: elements of the spectral lamp
        self.element = config["element"]
        #:str: medium of the detector, vac or air
        self.medium = config["medium"]

    @property
    def savefile(self):
        """str: Name of the wavelength echelle file"""
        return join(self.output_dir, self.prefix + ".thar.npz")

    def run(self, wavecal_master, wavecal_init):
        """Perform wavelength calibration

        This consists of extracting the wavelength image
        and fitting a polynomial the the known spectral lines

        Parameters
        ----------
        wavecal_master : tuple
            results of the wavecal_master step, containing the master wavecal image
            and its header
        wavecal_init : LineList
            the initial LineList guess with the positions and wavelengths of lines

        Returns
        -------
        wave : array of shape (nord, ncol)
            wavelength for each point in the spectrum
        coef : array of shape (*ndegrees,)
            polynomial coefficients of the wavelength fit
        linelist : record array of shape (nlines,)
            Updated line information for all lines
        """
        thar, thead = wavecal_master
        linelist = wavecal_init

        module = WavelengthCalibrationModule(
            plot=self.plot,
            plot_title=self.plot_title,
            manual=self.manual,
            degree=self.degree,
            threshold=self.threshold,
            iterations=self.iterations,
            dimensionality=self.dimensionality,
            nstep=self.nstep,
            shift_window=self.shift_window,
            element=self.element,
            medium=self.medium,
        )
        wave, coef = module.execute(thar, linelist)
        self.save(wave, coef, linelist)
        return wave, coef, linelist

    def save(self, wave, coef, linelist):
        """Save the results of the wavelength calibration

        Parameters
        ----------
        wave : array of shape (nord, ncol)
            wavelength for each point in the spectrum
        coef : array of shape (ndegrees,)
            polynomial coefficients of the wavelength fit
        linelist : record array of shape (nlines,)
            Updated line information for all lines
        """
        np.savez(self.savefile, wave=wave, coef=coef, linelist=linelist)
        logger.info("Created wavelength calibration file: %s", self.savefile)

    def load(self):
        """Load the results of the wavelength calibration

        Returns
        -------
        wave : array of shape (nord, ncol)
            wavelength for each point in the spectrum
        coef : array of shape (*ndegrees,)
            polynomial coefficients of the wavelength fit
        linelist : record array of shape (nlines,)
            Updated line information for all lines
        """
        data = np.load(self.savefile, allow_pickle=True)
        logger.info("Wavelength calibration file: %s", self.savefile)
        wave = data["wave"]
        coef = data["coef"]
        linelist = data["linelist"]
        return wave, coef, linelist


class LaserFrequencyCombMaster(CalibrationStep, ExtractionStep):
    """Create a laser frequency comb (or similar) master image"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["norm_flat", "curvature"]

    @property
    def savefile(self):
        """str: Name of the wavelength echelle file"""
        return join(self.output_dir, self.prefix + ".comb_master.fits")

    def run(self, files, orders, mask, curvature, bias, norm_flat):
        """Improve the wavelength calibration with a laser frequency comb (or similar)

        Parameters
        ----------
        files : list(str)
            observation files
        orders : tuple
            results from the order tracing step
        mask : array of shape (nrow, ncol)
            Bad pixel mask
        curvature : tuple
            results from the curvature step
        bias : tuple
            results from the bias step

        Returns
        -------
        comb : array of shape (nord, ncol)
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
        comb, _, _, _ = self.extract(orig, chead, orders, curvature)
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
        comb = fits.open(self.savefile)[0]
        comb, chead = comb.data, comb.header
        logger.info("Frequency comb master spectrum: %s", self.savefile)
        return comb, chead


class LaserFrequencyCombFinalize(Step):
    """Improve the precision of the wavelength calibration with a laser frequency comb"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["freq_comb_master", "wavecal"]
        self._loadDependsOn += ["wavecal"]

        #:tuple(int, int): polynomial degree of the wavelength fit
        self.degree = config["degree"]
        #:float: residual threshold in m/s above which to remove lines
        self.threshold = config["threshold"]
        #:{'1D', '2D'}: Whether to use 1D or 2D polynomials
        self.dimensionality = config["dimensionality"]
        self.nstep = config["nstep"]
        #:int: Width of the peaks for finding them in the spectrum
        self.lfc_peak_width = config["lfc_peak_width"]

    @property
    def savefile(self):
        """str: Name of the wavelength echelle file"""
        return join(self.output_dir, self.prefix + ".comb.npz")

    def run(self, freq_comb_master, wavecal):
        """Improve the wavelength calibration with a laser frequency comb (or similar)

        Parameters
        ----------
        files : list(str)
            observation files
        wavecal : tuple()
            results from the wavelength calibration step
        orders : tuple
            results from the order tracing step
        mask : array of shape (nrow, ncol)
            Bad pixel mask

        Returns
        -------
        wave : array of shape (nord, ncol)
            improved wavelength solution
        comb : array of shape (nord, ncol)
            extracted frequency comb image
        """
        comb, chead = freq_comb_master
        wave, coef, linelist = wavecal

        module = WavelengthCalibrationComb(
            plot=self.plot,
            plot_title=self.plot_title,
            degree=self.degree,
            threshold=self.threshold,
            dimensionality=self.dimensionality,
            nstep=self.nstep,
            lfc_peak_width=self.lfc_peak_width,
        )
        wave = module.execute(comb, wave, linelist)

        self.save(wave)
        return wave

    def save(self, wave):
        """Save the results of the frequency comb improvement

        Parameters
        ----------
        wave : array of shape (nord, ncol)
            improved wavelength solution
        """
        np.savez(self.savefile, wave=wave)
        logger.info("Created frequency comb wavecal file: %s", self.savefile)

    def load(self, wavecal):
        """Load the results of the frequency comb improvement if possible,
        otherwise just use the normal wavelength solution

        Parameters
        ----------
        wavecal : tuple
            results from the wavelength calibration step

        Returns
        -------
        wave : array of shape (nord, ncol)
            improved wavelength solution
        comb : array of shape (nord, ncol)
            extracted frequency comb image
        """
        try:
            data = np.load(self.savefile, allow_pickle=True)
            logger.info("Frequency comb wavecal file: %s", self.savefile)
        except FileNotFoundError:
            logger.warning(
                "No data for Laser Frequency Comb found, using regular wavelength calibration instead"
            )
            wave, coef, linelist = wavecal
            data = {"wave": wave}
        wave = data["wave"]
        return wave


class SlitCurvatureDetermination(CalibrationStep, ExtractionStep):
    """Determine the curvature of the slit"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)

        #:float: how many sigma of bad lines to cut away
        self.sigma_cutoff = config["curvature_cutoff"]
        #:float: width of the orders in the extraction
        self.extraction_width = config["extraction_width"]
        #:int: Polynomial degree of the overall fit
        self.fit_degree = config["degree"]
        #:int: Orders of the curvature to fit, currently supports only 1 and 2
        self.curv_degree = config["curv_degree"]
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

    @property
    def savefile(self):
        """str: Name of the tilt/shear save file"""
        return join(self.output_dir, self.prefix + ".shear.npz")

    def run(self, files, orders, mask, bias):
        """Determine the curvature of the slit

        Parameters
        ----------
        files : list(str)
            files to use for this
        orders : tuple
            results of the order tracing
        mask : array of shape (nrow, ncol)
            Bad pixel mask

        Returns
        -------
        tilt : array of shape (nord, ncol)
            first order slit curvature at each point
        shear : array of shape (nord, ncol)
            second order slit curvature at each point
        """

        logger.info("Slit curvature files: %s", files)

        orig, thead = self.calibrate(files, mask, bias, None)
        extracted, _, _, _ = self.extract(orig, thead, orders, None)

        orders, column_range = orders
        module = CurvatureModule(
            orders,
            column_range=column_range,
            extraction_width=self.extraction_width,
            order_range=self.order_range,
            fit_degree=self.fit_degree,
            curv_degree=self.curv_degree,
            sigma_cutoff=self.sigma_cutoff,
            mode=self.curvature_mode,
            peak_threshold=self.peak_threshold,
            peak_width=self.peak_width,
            window_width=self.window_width,
            peak_function=self.peak_function,
            plot=self.plot,
            plot_title=self.plot_title,
        )
        tilt, shear = module.execute(extracted, orig)
        self.save(tilt, shear)
        return tilt, shear

    def save(self, tilt, shear):
        """Save results from the curvature

        Parameters
        ----------
        tilt : array of shape (nord, ncol)
            first order slit curvature at each point
        shear : array of shape (nord, ncol)
            second order slit curvature at each point
        """
        np.savez(self.savefile, tilt=tilt, shear=shear)
        logger.info("Created slit curvature file: %s", self.savefile)

    def load(self):
        """Load the curvature if possible, otherwise return None, None, i.e. use vertical extraction

        Returns
        -------
        tilt : array of shape (nord, ncol)
            first order slit curvature at each point
        shear : array of shape (nord, ncol)
            second order slit curvature at each point
        """
        try:
            data = np.load(self.savefile, allow_pickle=True)
            logger.info("Slit curvature file: %s", self.savefile)
        except FileNotFoundError:
            logger.warning("No data for slit curvature found, setting it to 0.")
            data = {"tilt": None, "shear": None}

        tilt = data["tilt"]
        shear = data["shear"]
        return tilt, shear


class RectifyImage(Step):
    """Create a 2D image of the rectified orders"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["files", "orders", "curvature", "mask", "freq_comb"]
        # self._loadDependsOn += []

        self.extraction_width = config["extraction_width"]
        self.input_files = config["input_files"]

    def filename(self, name):
        return util.swap_extension(name, ".rectify.fits", path=self.output_dir)

    def run(self, files, orders, curvature, mask, freq_comb):
        orders, column_range = orders
        tilt, shear = curvature
        wave = freq_comb

        files = files[self.input_files]

        rectified = {}
        for fname in tqdm(files, desc="Files"):
            img, head = self.instrument.load_fits(
                fname, self.mode, mask=mask, dtype="f8"
            )

            images, cr, xwd = rectify_image(
                img,
                orders,
                column_range,
                self.extraction_width,
                self.order_range,
                tilt,
                shear,
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
            data = fits.open(fname)
            img = data[1].data
            wave = data[2].data["wavelength"]
            rectified[orig_fname] = (wave, img)

        return rectified


class ScienceExtraction(CalibrationStep, ExtractionStep):
    """Extract the science spectra"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["norm_flat", "curvature"]
        self._loadDependsOn += ["files"]

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
        return util.swap_extension(name, ".science.ech", path=self.output_dir)

    def run(self, files, bias, orders, norm_flat, curvature, mask):
        """Extract Science spectra from observation

        Parameters
        ----------
        files : list(str)
            list of observations
        bias : tuple
            results from master bias step
        orders : tuple
            results from order tracing step
        norm_flat : tuple
            results from flat normalization
        curvature : tuple
            results from slit curvature step
        mask : array of shape (nrow, ncol)
            bad pixel map

        Returns
        -------
        heads : list(FITS header)
            FITS headers of each observation
        specs : list(array of shape (nord, ncol))
            extracted spectra
        sigmas : list(array of shape (nord, ncol))
            uncertainties of the extracted spectra
        columns : list(array of shape (nord, 2))
            column ranges for each spectra
        """
        heads, specs, sigmas, columns = [], [], [], []
        for fname in tqdm(files, desc="Files"):
            logger.info("Science file: %s", fname)
            # Calibrate the input image
            im, head = self.calibrate([fname], mask, bias, norm_flat)
            # Optimally extract science spectrum
            spec, sigma, _, cr = self.extract(im, head, orders, curvature)

            # save spectrum to disk
            self.save(fname, head, spec, sigma, cr)
            heads.append(head)
            specs.append(spec)
            sigmas.append(sigma)
            columns.append(cr)

        return heads, specs, sigmas, columns

    def save(self, fname, head, spec, sigma, column_range):
        """Save the results of one extraction

        Parameters
        ----------
        fname : str
            filename to save to
        head : FITS header
            FITS header
        spec : array of shape (nord, ncol)
            extracted spectrum
        sigma : array of shape (nord, ncol)
            uncertainties of the extracted spectrum
        column_range : array of shape (nord, 2)
            range of columns that have spectrum
        """
        nameout = self.science_file(fname)
        echelle.save(nameout, head, spec=spec, sig=sigma, columns=column_range)
        logger.info("Created science file: %s", nameout)

    def load(self, files):
        """Load all science spectra from disk

        Returns
        -------
        heads : list(FITS header)
            FITS headers of each observation
        specs : list(array of shape (nord, ncol))
            extracted spectra
        sigmas : list(array of shape (nord, ncol))
            uncertainties of the extracted spectra
        columns : list(array of shape (nord, 2))
            column ranges for each spectra
        """
        files = files["science"]
        files = [self.science_file(fname) for fname in files]

        if len(files) == 0:
            raise FileNotFoundError("Science files are required to load them")

        logger.info("Science files: %s", files)

        heads, specs, sigmas, columns = [], [], [], []
        for fname in files:
            # fname = join(self.output_dir, fname)
            science = echelle.read(
                fname,
                continuum_normalization=False,
                barycentric_correction=False,
                radial_velociy_correction=False,
            )
            heads.append(science.header)
            specs.append(science["spec"])
            sigmas.append(science["sig"])
            columns.append(science["columns"])

        return heads, specs, sigmas, columns


class ContinuumNormalization(Step):
    """Determine the continuum to each observation"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["science", "freq_comb", "norm_flat"]
        self._loadDependsOn += ["norm_flat", "science"]

    @property
    def savefile(self):
        """str: savefile name"""
        return join(self.output_dir, self.prefix + ".cont.npz")

    def run(self, science, freq_comb, norm_flat):
        """Determine the continuum to each observation
        Also splices the orders together

        Parameters
        ----------
        science : tuple
            results from science step
        freq_comb : tuple
            results from freq_comb step (or wavecal if those don't exist)
        norm_flat : tuple
            results from the normalized flatfield step

        Returns
        -------
        heads : list(FITS header)
            FITS headers of each observation
        specs : list(array of shape (nord, ncol))
            extracted spectra
        sigmas : list(array of shape (nord, ncol))
            uncertainties of the extracted spectra
        conts : list(array of shape (nord, ncol))
            continuum for each spectrum
        columns : list(array of shape (nord, 2))
            column ranges for each spectra
        """
        wave = freq_comb
        heads, specs, sigmas, columns = science
        norm, blaze = norm_flat

        logger.info("Continuum normalization")
        conts = [None for _ in specs]
        for j, (spec, sigma) in enumerate(zip(specs, sigmas)):
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
        specs : list(array of shape (nord, ncol))
            extracted spectra
        sigmas : list(array of shape (nord, ncol))
            uncertainties of the extracted spectra
        conts : list(array of shape (nord, ncol))
            continuum for each spectrum
        columns : list(array of shape (nord, 2))
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
        specs : list(array of shape (nord, ncol))
            extracted spectra
        sigmas : list(array of shape (nord, ncol))
            uncertainties of the extracted spectra
        conts : list(array of shape (nord, ncol))
            continuum for each spectrum
        columns : list(array of shape (nord, 2))
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
            norm, blaze = norm_flat
            conts = [blaze for _ in specs]
            data = dict(
                heads=heads, specs=specs, sigmas=sigmas, conts=conts, columns=columns
            )
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
        self._dependsOn += ["continuum", "freq_comb", "config"]
        self.filename = config["filename"]

    def output_file(self, number, name):
        """str: output file name"""
        out = self.filename.format(
            instrument=self.instrument.name,
            night=self.night,
            mode=self.mode,
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

    def run(self, continuum, freq_comb, config):
        """Create the final output files

        this is includes:
         - heliocentric corrections
         - creating one echelle file

        Parameters
        ----------
        continuum : tuple
            results from the continuum normalization
        freq_comb : tuple
            results from the frequency comb step (or wavelength calibration)
        """
        heads, specs, sigmas, conts, columns = continuum
        wave = freq_comb

        fnames = []
        # Combine science with wavecal and continuum
        for i, (head, spec, sigma, blaze, column) in enumerate(
            zip(heads, specs, sigmas, conts, columns)
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
                plt.plot(wave.T, (spec / blaze).T)
                if self.plot_title is not None:
                    plt.title(self.plot_title)
                plt.show()

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
        spec : array of shape (nord, ncol)
            final spectrum
        sigma : array of shape (nord, ncol)
            final uncertainties
        cont : array of shape (nord, ncol)
            final continuum scales
        wave : array of shape (nord, ncol)
            wavelength solution
        columns : array of shape (nord, 2)
            columns that carry signal

        Returns
        -------
        out_file : str
            name of the output file
        """
        original_name = os.path.splitext(head["e_input"])[0]
        out_file = self.output_file(i, original_name)
        echelle.save(
            out_file, head, spec=spec, sig=sigma, cont=cont, wave=wave, columns=columns
        )
        logger.info("Final science file: %s", out_file)
        return out_file


class Reducer:

    step_order = {
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

    modules = {
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

    def __init__(
        self,
        files,
        output_dir,
        target,
        instrument,
        mode,
        night,
        config,
        order_range=None,
        skip_existing=False,
    ):
        """Reduce all observations from a single night and instrument mode

        Parameters
        ----------
        files: dict{str:str}
            Data files for each step
        output_dir : str
            directory to place output files in
        target : str
            observed targets as used in directory names/fits headers
        instrument : str
            instrument used for observations
        mode : str
            instrument mode used (e.g. "red" or "blue" for HARPS)
        night : str
            Observation night, in the same format as used in the directory structure/file sorting
        config : dict
            numeric reduction specific settings, like pixel threshold, which may change between runs
        info : dict
            fixed instrument specific values, usually header keywords for gain, readnoise, etc.
        skip_existing : bool
            Whether to skip reductions with existing output
        """
        #:dict(str:str): Filenames sorted by usecase
        self.files = files
        self.output_dir = output_dir.format(
            instrument=str(instrument), target=target, night=night, mode=mode
        )

        if isinstance(instrument, str):
            instrument = load_instrument(instrument)

        self.data = {"files": files, "config": config}
        self.inputs = (instrument, mode, target, night, output_dir, order_range)
        self.config = config
        self.skip_existing = skip_existing

    def run_module(self, step, load=False):
        # The Module this step is based on (An object of the Step class)
        module = self.modules[step](*self.inputs, **self.config.get(step, {}))

        # Load the dependencies necessary for loading/running this step
        dependencies = module.dependsOn if not load else module.loadDependsOn
        for dependency in dependencies:
            if dependency not in self.data.keys():
                self.data[dependency] = self.run_module(dependency, load=True)
        args = {d: self.data[d] for d in dependencies}

        # Try to load the data, if the step is not specifically given as necessary
        # If the intermediate data is not available, run it normally instead
        # But give a warning
        if load:
            try:
                logger.info("Loading data from step '%s'", step)
                data = module.load(**args)
            except FileNotFoundError:
                logger.warning(
                    "Intermediate File(s) for loading step %s not found. Running it instead.",
                    step,
                )
                data = self.run_module(step, load=False)
        else:
            logger.info("Running step '%s'", step)
            if step in self.files.keys():
                args["files"] = self.files[step]
            data = module.run(**args)

        self.data[step] = data
        return data

    def prepare_output_dir(self):
        # create output folder structure if necessary
        output_dir = self.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def run_steps(self, steps="all"):
        """
        Execute the steps as required

        Parameters
        ----------
        steps : {tuple(str), "all"}, optional
            which steps of the reduction process to perform
            the possible steps are: "bias", "flat", "orders", "norm_flat", "wavecal", "freq_comb",
            "curvature", "science", "continuum", "finalize"
            alternatively set steps to "all", which is equivalent to setting all steps
        """
        self.prepare_output_dir()

        if steps == "all":
            steps = list(self.step_order.keys())
        steps = list(steps)

        if self.skip_existing and "finalize" in steps:
            module = self.modules["finalize"](
                *self.inputs, **self.config.get("finalize", {})
            )
            exists = [False] * len(self.files["science"])
            data = {"finalize": [None] * len(self.files["science"])}
            for i, f in enumerate(self.files["science"]):
                fname_in = os.path.basename(f)
                fname_in = os.path.splitext(fname_in)[0]
                fname_out = module.output_file("?", fname_in)
                fname_out = glob.glob(fname_out)
                exists[i] = len(fname_out) != 0
                if exists[i]:
                    data["finalize"][i] = fname_out[0]
            if all(exists):
                logger.info("All science files already exist, skipping this set")
                logger.debug("--------------------------------")
                return data

        steps.sort(key=lambda x: self.step_order[x])

        for step in steps:
            self.run_module(step)

        logger.debug("--------------------------------")
        return self.data
