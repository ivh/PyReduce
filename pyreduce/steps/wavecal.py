"""Wavelength calibration and laser frequency comb steps."""

import json
import logging
import os
from os.path import join

import numpy as np
from astropy.io import fits

# PyReduce subpackages
from ..provenance import add_provenance
from ..trace_model import (
    Trace as TraceData,
)
from ..trace_model import (
    save_traces,
)
from ..wavelength_calibration import LineList, WavelengthCalibrationComb
from ..wavelength_calibration import (
    WavelengthCalibration as WavelengthCalibrationModule,
)
from ..wavelength_calibration import (
    WavelengthCalibrationInitialize as WavelengthCalibrationInitializeModule,
)
from .base import (
    CalibrationStep,
    ExtractionStep,
    Step,
    wavelengths_from_traces,
)

logger = logging.getLogger(__name__)


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

        # Load wavecal image (same for all groups) and overlay selected
        # traces on the diagnostic plot, like the science step does.
        all_selected = [t for group_traces in selected.values() for t in group_traces]
        orig, thead = self.calibrate(
            files,
            mask,
            bias,
            norm_flat,
            traces=all_selected if all_selected else None,
            extraction_height=self.extraction_kwargs.get("extraction_height"),
        )

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
            thead = add_provenance(thead)
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


def _is_single_order_multi_bundle(traces) -> bool:
    """True iff every trace is a bundle of a single-order spectrograph.

    Used to switch wavecal between (a) the multi-order indexing where each
    extracted row is a distinct spectral order, and (b) the MOSAIC-style
    case where every row is a different fiber bundle sharing one m.
    """
    return (
        len(traces) > 1
        and all(t.bundle is not None for t in traces)
        and all(t.m is None for t in traces)
    )


class WavelengthCalibrationInitialize(Step):
    """Create the initial wavelength solution file"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["wavecal_master", "trace"]
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
        self.wave_delta = config.get("wave_delta", 20)

    def savefile_for_group(self, group: str) -> str:
        """Get savefile path for a specific group."""
        if group == "all":
            return join(self.output_dir, self.prefix + ".linelist.npz")
        return join(self.output_dir, self.prefix + f"_{group}.linelist.npz")

    @property
    def savefile(self):
        """str: Name of the linelist file (single-group compat)"""
        return self.savefile_for_group("all")

    def run(self, wavecal_master: dict, trace: list):
        """Run iterative line matching for each fiber group.

        Parameters
        ----------
        wavecal_master : dict[str, tuple]
            {group: (wavecal_spec, thead)} from wavecal_master step
        trace : list[TraceData]
            All trace objects (used to detect single-order multi-bundle mode)

        Returns
        -------
        results : dict[str, LineList]
            {group: linelist} for each fiber group
        """
        selected = self._select_traces(trace, "wavecal_master")

        results = {}
        for group, (wavecal_spec, thead) in wavecal_master.items():
            logger.info("Running wavecal_init for group '%s'", group)

            group_traces = selected.get(group, [])
            single_order = _is_single_order_multi_bundle(group_traces)
            if single_order:
                logger.info(
                    "Group '%s': single-order multi-bundle mode (%d bundles)",
                    group,
                    len(group_traces),
                )

            # Get the initial wavelength guess from the instrument
            wave_range = self.instrument.get_wavelength_range(thead, self.channel)
            if wave_range is None:
                raise ValueError(
                    "This instrument is missing an initial wavelength guess for wavecal_init"
                )

            # Per-bundle guess (single-order multi-bundle): look up each spectrum
            # row's range by its trace bundle id. group_traces is in the same
            # order as the extracted wavecal_spec rows.
            per_bundle = self.instrument.get_wavelength_range_per_bundle(
                thead, self.channel
            )
            if single_order and per_bundle and group_traces:
                default = wave_range[0]
                wave_range = [per_bundle.get(t.bundle, default) for t in group_traces]

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
                wave_delta=self.wave_delta,
            )
            linelist = module.execute(
                wavecal_spec, wave_range, single_order=single_order
            )
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
        self.quality = {}

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

            metrics = module.quality_metrics(wave, linelist)
            self.quality[group] = metrics
            logger.info(
                "Wavecal quality for group '%s': rms=%.1f m/s, "
                "%d lines used, %d rejected",
                group,
                metrics["rms_mps"] if metrics["rms_mps"] is not None else float("nan"),
                metrics["nlines_used"],
                metrics["nlines_rejected"],
            )

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
        # Resolve traces with the SAME selection wavecal_master used, so the
        # wave rows line up with the right traces and in the same order. Using
        # t.group directly is wrong for single-order multi-bundle instruments
        # (MOSAIC): the trace list mixes bundle representatives with the raw
        # ungrouped fibers, and results is keyed "all".
        selected = self._select_traces(trace, "wavecal_master")

        for group, (wave, linelist) in results.items():
            # Prefer the same grouping wavecal_master used (handles MOSAIC, where
            # results is keyed "all" but the trace list mixes bundle reps with
            # raw fibers). Fall back to matching trace.group for named-group
            # instruments, then to all selected traces.
            group_traces = selected.get(group)
            if not group_traces:
                group_traces = [
                    t
                    for t in trace
                    if (str(t.group) if t.group is not None else "all") == group
                ]
            if not group_traces and group.startswith("fiber_"):
                try:
                    fidx = int(group.split("_", 1)[1])
                    group_traces = [t for t in trace if t.fiber_idx == fidx]
                except ValueError:
                    pass
            if not group_traces:
                group_traces = selected.get("all")
            if not group_traces:
                logger.warning("No traces found for group '%s'", group)
                continue

            # Update trace.m from obase if not already set.
            # Skip when traces are bundles of a single-order spectrograph --
            # there t.bundle is the meaningful spatial id and m must stay None.
            single_order = _is_single_order_multi_bundle(group_traces)
            obase = linelist.obase
            if obase is not None and not single_order:
                already_have_m = any(t.m is not None for t in group_traces)
                if already_have_m:
                    logger.debug(
                        "Traces for group '%s' already have m values, skipping obase",
                        group,
                    )
                else:
                    for idx_in_group, t in enumerate(group_traces):
                        t.m = obase + idx_in_group
                    logger.info(
                        "Updated trace order numbers for group '%s' with obase=%d",
                        group,
                        obase,
                    )

            # Store wavelength polynomial in each trace.
            if self.dimensionality == "1D":
                for idx_in_group, t in enumerate(group_traces):
                    if idx_in_group < len(wave):
                        t.wave = wave[idx_in_group]
            else:
                # Evaluate 2D poly P(x, order_idx) at each trace's 0-based
                # index to get a 1D poly in x (np.polyfit convention).
                for idx_in_group, t in enumerate(group_traces):
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

        if getattr(self, "quality", None):
            quality_file = join(self.output_dir, self.prefix + ".wavecal_quality.json")
            with open(quality_file, "w") as f:
                json.dump(self.quality, f, indent=2)
            logger.info("Saved wavecal quality metrics: %s", quality_file)

        trace_file = join(self.output_dir, self.prefix + ".traces.fits")
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
        chead = add_provenance(chead)
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

        selected = self._select_traces(trace, "wavecal")
        flat_traces = [t for group in selected.values() for t in group]

        # Get base wavelengths from selected traces
        wlen = wavelengths_from_traces(flat_traces)
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
        for i, t in enumerate(flat_traces):
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

    def load(self):
        """Load is a no-op - wavelengths are in traces.fits."""
        # Nothing to load - downstream steps get wavelengths from traces
        pass
