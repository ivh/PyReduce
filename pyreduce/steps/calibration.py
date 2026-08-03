"""Calibration steps: bad pixel mask, bias, flat, background scatter, flat normalization."""

import logging
from os.path import join

import numpy as np
from astropy.io import fits

# PyReduce subpackages
from ..combine_frames import (
    combine_bias,
    combine_polynomial,
)
from ..estimate_background_scatter import estimate_background_scatter
from ..extract import extract_normalize
from ..provenance import add_provenance
from ..trace_model import (
    Trace as TraceData,
)
from .base import (
    CalibrationStep,
    Step,
)

logger = logging.getLogger(__name__)


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
        bhead = add_provenance(bhead)

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
        fhead = add_provenance(fhead)
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
        if flat is None or (isinstance(flat, tuple) and flat[0] is None):
            logger.warning("No master flat available, skipping flat normalization")
            return None

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
