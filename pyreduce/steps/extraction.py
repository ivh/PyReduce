"""Image rectification and science extraction steps."""

import logging

import joblib
import numpy as np
from astropy.io import fits
from tqdm import tqdm

# PyReduce subpackages
from .. import util
from ..estimate_background_scatter import as_scatter_coeff
from ..extract import extract
from ..provenance import add_provenance
from ..rectify import merge_images, rectify_image
from ..spectra import ExtractionParams, Spectra, Spectrum
from ..trace_model import (
    Trace as TraceData,
)
from .base import (
    CalibrationStep,
    ExtractionStep,
    Step,
    wavelengths_from_traces,
)

logger = logging.getLogger(__name__)


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
        selected = self._select_traces(trace, "science")
        flat_traces = [t for group in selected.values() for t in group]

        # Get wavelengths from traces (includes freq_comb improvements if run)
        wave = wavelengths_from_traces(flat_traces)

        files = files[self.input_files]

        rectified = {}
        for fname in tqdm(files, desc="Files"):
            img, head = self.instrument.load_fits(
                fname, self.channel, mask=mask, dtype="f8"
            )

            images, cr, xwd = rectify_image(
                img,
                flat_traces,
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
        primary = fits.PrimaryHDU(header=add_provenance(header))
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
        #:int: number of files to extract in parallel (joblib semantics, -1 = all cores)
        self.n_jobs = config.get("n_jobs", 1)

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
        scatter : ScatterModel, optional
            background scatter model; re-estimated on each science frame
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

        if self.n_jobs != 1 and self.plot:
            # matplotlib does not survive worker processes
            logger.warning(
                "Disabling plots for parallel science extraction (n_jobs=%s)",
                self.n_jobs,
            )
            self.plot = False

        # All traces, not just the selected ones: the scatter fit must mask every
        # order on the detector, or it fits the flux of the ones it left out.
        args = (trace_list, trace, default_height, extraction_kwargs)
        calib = (bias, norm_flat, scatter, mask)
        if self.n_jobs == 1:
            results = [
                self._extract_file(fname, *args, *calib)
                for fname in tqdm(files, desc="Files")
            ]
        else:
            parallel = joblib.Parallel(n_jobs=self.n_jobs, return_as="generator")
            jobs = (
                joblib.delayed(self._extract_file)(fname, *args, *calib)
                for fname in files
            )
            results = list(tqdm(parallel(jobs), total=len(files), desc="Files"))

        heads = [head for head, _ in results]
        all_spectra = [spectra for _, spectra in results]
        return heads, all_spectra

    def _extract_file(
        self,
        fname,
        trace_list,
        all_traces,
        default_height,
        extraction_kwargs,
        bias,
        norm_flat,
        scatter,
        mask,
    ):
        """Calibrate, extract, and save a single observation."""
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

        # Scattered light scales with the illumination of this exposure, so it is
        # measured on this calibrated frame rather than inherited from the flat.
        scatter_coeff = as_scatter_coeff(scatter, im, all_traces, context=fname)

        # Extract science spectrum - returns list[Spectrum]
        spectra = extract(
            im,
            trace_list,
            extraction_height=default_height,
            extraction_type=self.extraction_method,
            gain=head["e_gain"],
            readnoise=head["e_readn"],
            dark=head["e_drk"],
            scatter=scatter_coeff,
            plot=self.plot,
            plot_title=self.plot_title,
            **extraction_kwargs,
        )

        # Save spectrum to disk
        self.save(fname, head, spectra)
        return head, spectra

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
