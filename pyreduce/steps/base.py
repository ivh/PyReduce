"""Base classes and shared helpers for reduction steps."""

import logging
import warnings

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning

# PyReduce subpackages
from ..combine_frames import (
    combine_calibrate,
)
from ..extract import extract
from ..provenance import add_provenance
from ..trace import (
    select_traces_for_step,
)
from ..trace_model import (
    Trace as TraceData,
)

warnings.simplefilter("ignore", category=VerifyWarning, append=True)
warnings.simplefilter("ignore", category=AstropyUserWarning, append=True)

logger = logging.getLogger(__name__)


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

        head = add_provenance(head)
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
