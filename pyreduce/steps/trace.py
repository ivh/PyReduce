"""Trace detection and slit curvature steps."""

import logging
import os
from os.path import join

import numpy as np
from astropy.io import fits

# PyReduce subpackages
from ..slit_curve import Curvature as CurvatureModule
from ..trace import (
    _compute_heights_inplace,
    group_fibers,
    select_traces_for_step,
)
from ..trace import trace as detect_traces
from ..trace_model import (
    Trace as TraceData,
)
from ..trace_model import (
    load_traces,
    save_traces,
)
from .base import (
    CalibrationStep,
    ExtractionStep,
)

logger = logging.getLogger(__name__)


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
        #:float: Maximum RMS of the order fit; clusters above this are discarded
        self.max_error = config.get("max_error", None)

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

        # Load order_centers for m assignment, and bundle_centers for
        # bundle assignment. They are independent: order_centers populates
        # t.m (spectral order), bundle_centers populates t.bundle (spatial
        # bundle id within an order).
        order_centers = self._load_order_centers()
        bundle_centers = self._load_bundle_centers()

        # Check if we should trace file groups separately
        fibers_config = getattr(self.instrument.config, "fibers", None)
        trace_by = getattr(fibers_config, "trace_by", None) if fibers_config else None

        if trace_by and len(files) > 1:
            raw_traces = self._trace_by_groups(
                files, mask, bias, trace_by, order_centers, bundle_centers
            )
        else:
            raw_traces = self._trace_single(
                files, mask, bias, order_centers, bundle_centers
            )

        # Store heights for backward compatibility
        self.heights = np.array(
            [t.height if t.height is not None else np.nan for t in raw_traces]
        )

        # Group fibers if configured (creates new traces with group set)
        if fibers_config is not None and (
            fibers_config.groups is not None or fibers_config.bundles is not None
        ):
            grouped = group_fibers(
                raw_traces,
                fibers_config,
                degree=self.fit_degree,
                bundle_centers=bundle_centers,
            )
            self.trace_objects = grouped + raw_traces
        else:
            self.trace_objects = raw_traces

        self.save()

        return self.trace_objects

    def _resolve_centers_file(self, centers_file, config_key: str):
        """Resolve a configured centers file to an existing path.

        Supports per-channel lists and the {channel} filename template,
        relative to the instrument directory. A configured file that does
        not exist is a config error and raises, because falling back to
        sequential order numbering would silently mis-assign traces.
        """
        from pathlib import Path

        if isinstance(centers_file, list):
            channels = self.instrument.config.channels or []
            ch_idx = channels.index(self.channel) if self.channel in channels else 0
            centers_file = (
                centers_file[ch_idx] if ch_idx < len(centers_file) else centers_file[0]
            )

        if self.channel and "{channel}" in centers_file:
            centers_file = centers_file.format(channel=self.channel.lower())

        inst_dir = getattr(self.instrument, "_inst_dir", None)
        path = Path(centers_file)
        if not path.is_absolute() and inst_dir:
            path = Path(inst_dir) / centers_file

        if not path.exists():
            raise FileNotFoundError(
                f"{config_key} is configured but the file does not exist: {path}"
            )
        return path

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

        import yaml

        path = self._resolve_centers_file(
            fibers_config.order_centers_file, "fibers.order_centers_file"
        )

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

    def _load_bundle_centers(self) -> dict[int, float] | None:
        """Load bundle_centers from fibers.bundles config as order_centers fallback."""
        fibers_config = getattr(self.instrument.config, "fibers", None)
        if fibers_config is None or fibers_config.bundles is None:
            return None

        bundles = fibers_config.bundles

        if bundles.bundle_centers is not None:
            logger.info(
                "Using inline bundle_centers: %d bundles", len(bundles.bundle_centers)
            )
            return bundles.bundle_centers

        if bundles.bundle_centers_file is None:
            return None

        import yaml

        path = self._resolve_centers_file(
            bundles.bundle_centers_file, "fibers.bundles.bundle_centers_file"
        )

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        if "bundle_centers" in data:
            data = data["bundle_centers"]

        result = {int(k): float(v) for k, v in data.items()}
        logger.info("Loaded bundle_centers from %s: %d bundles", path, len(result))
        return result

    def _trace_by_groups(
        self, files, mask, bias, trace_by, order_centers, bundle_centers=None
    ):
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
            traces = self._trace_single(
                group_files, mask, bias, order_centers, bundle_centers
            )
            logger.info("  Found %d traces", len(traces))
            all_traces.extend(traces)

        # Re-assign fiber_idx within each (m, bundle), since each trace_by
        # group assigned its own 1..N independently. Direction follows
        # fibers.numbering, matching assign_orders_and_fibers.
        from collections import defaultdict

        fibers_config = getattr(self.instrument.config, "fibers", None)
        top_down = (
            getattr(fibers_config, "numbering", "bottom_up") == "top_down"
            if fibers_config
            else False
        )

        traces_by_mb = defaultdict(list)
        for t in all_traces:
            traces_by_mb[(t.m, t.bundle)].append(t)

        for _key, order_traces in traces_by_mb.items():
            x_mid = sum(order_traces[0].column_range) / 2
            order_traces.sort(key=lambda t: t.y_at_x(x_mid), reverse=top_down)
            for idx, t in enumerate(order_traces, start=1):
                t.fiber_idx = idx

        # Recompute heights now that all groups are merged, so each trace
        # sees its true nearest neighbor (not just within its trace_by group).
        ncol = max(t.column_range[1] for t in all_traces)
        x_mid = ncol // 2
        all_traces.sort(key=lambda t: t.y_at_x(x_mid))
        _compute_heights_inplace(all_traces, ncol)

        # Sort by (m descending, fiber_idx)
        all_traces.sort(
            key=lambda t: (-t.m if t.m is not None else 0, t.fiber_idx or 0)
        )

        logger.info(
            "Merged %d total traces from %d groups", len(all_traces), len(file_groups)
        )

        return all_traces

    def _trace_single(self, files, mask, bias, order_centers, bundle_centers=None):
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
        top_down = (
            getattr(fibers_config, "numbering", "bottom_up") == "top_down"
            if fibers_config
            else False
        )

        traces = detect_traces(
            trace_img,
            min_cluster=self.min_cluster,
            min_width=self.min_width,
            filter_x=self.filter_x,
            filter_y=self.filter_y,
            filter_type=self.filter_type,
            noise=self.noise,
            noise_relative=self.noise_relative,
            degree=self.fit_degree,
            max_error=self.max_error,
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
            bundle_centers=bundle_centers,
            top_down=top_down,
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

    def load(self):
        """Curvature is now stored in traces, not separate files."""
        return None
