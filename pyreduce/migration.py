"""
Migration helpers for converting old file formats to new formats.

This module provides functions to detect and convert:
- Old NPZ trace files → new FITS trace files
- Old Echelle FITS files → new Spectra FITS files
"""

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

from .spectra import Spectra
from .trace_model import Trace, arrays_to_traces, save_traces

logger = logging.getLogger(__name__)


def detect_trace_format(path: str | Path) -> str:
    """Detect the format of a trace file.

    Parameters
    ----------
    path : str or Path
        Path to the trace file.

    Returns
    -------
    str
        Format identifier: "fits_v2" (new), "fits_v1" (old FITS), "npz" (legacy NPZ)
    """
    path = Path(path)

    if path.suffix == ".npz":
        return "npz"

    if path.suffix == ".fits":
        with fits.open(path) as hdu:
            fmtver = hdu[0].header.get("E_FMTVER", 1)
            if fmtver >= 2:
                return "fits_v2"
            return "fits_v1"

    raise ValueError(f"Unknown trace file format: {path}")


def detect_spectrum_format(path: str | Path) -> str:
    """Detect the format of a spectrum file.

    Parameters
    ----------
    path : str or Path
        Path to the spectrum file.

    Returns
    -------
    str
        Format identifier: "spectra_v2" (new), "echelle_v1" (legacy)
    """
    path = Path(path)

    with fits.open(path) as hdu:
        fmtver = hdu[0].header.get("E_FMTVER", 1)
        if fmtver >= 2:
            return "spectra_v2"
        return "echelle_v1"


def convert_trace_npz_to_fits(
    npz_path: str | Path,
    fits_path: str | Path = None,
    curvature_path: str | Path = None,
    wavecal_path: str | Path = None,
) -> list[Trace]:
    """Convert a legacy NPZ trace file to new FITS format.

    Parameters
    ----------
    npz_path : str or Path
        Path to the input NPZ trace file.
    fits_path : str or Path, optional
        Path for the output FITS file. If None, replaces .npz with .fits
    curvature_path : str or Path, optional
        Path to curvature NPZ file to merge into traces.
    wavecal_path : str or Path, optional
        Path to wavelength calibration file to merge into traces.

    Returns
    -------
    list[Trace]
        Converted trace objects.
    """
    npz_path = Path(npz_path)
    if fits_path is None:
        fits_path = npz_path.with_suffix(".fits")

    # Load from NPZ
    data = np.load(npz_path, allow_pickle=True)

    # Handle old key names
    if "traces" in data:
        trace_coeffs = data["traces"]
    elif "orders" in data:
        trace_coeffs = data["orders"]
    else:
        raise KeyError("Trace file missing 'traces' or 'orders' key")

    column_range = data["column_range"]
    heights = data.get("heights", None)
    if heights is not None and heights.ndim == 0:
        heights = None

    # Convert to Trace objects
    traces = arrays_to_traces(trace_coeffs, column_range, heights)

    # Load and merge curvature if provided
    if curvature_path is not None:
        curvature_path = Path(curvature_path)
        if curvature_path.exists():
            from .curvature_model import load_curvature

            curvature = load_curvature(curvature_path)
            if curvature is not None:
                logger.info("Merging curvature data from: %s", curvature_path)
                for i, t in enumerate(traces):
                    if i < curvature.coeffs.shape[0]:
                        t.slit = curvature.coeffs[i]
                        if curvature.slitdeltas is not None:
                            t.slitdelta = curvature.slitdeltas[i]

    # Save to FITS
    save_traces(fits_path, traces, steps=["trace"])
    logger.info("Converted trace file: %s -> %s", npz_path, fits_path)

    return traces


def convert_echelle_to_spectra(
    echelle_path: str | Path,
    spectra_path: str | Path = None,
) -> Spectra:
    """Convert a legacy Echelle FITS file to new Spectra format.

    Spectra.read() handles both formats, so this just reads and re-saves
    to upgrade the file format.

    Parameters
    ----------
    echelle_path : str or Path
        Path to the input Echelle FITS file.
    spectra_path : str or Path, optional
        Path for the output Spectra file. If None, appends "_v2" before extension.

    Returns
    -------
    Spectra
        Converted Spectra object.
    """
    echelle_path = Path(echelle_path)
    if spectra_path is None:
        spectra_path = echelle_path.with_stem(echelle_path.stem + "_v2")

    # Spectra.read() handles legacy format via _read_legacy_format()
    spectra = Spectra.read(
        echelle_path,
        raw=True,
        continuum_normalization=False,
        barycentric_correction=False,
        radial_velocity_correction=False,
    )

    # Save in new format
    spectra.save(spectra_path, steps=["migration"])
    logger.info("Converted Echelle file: %s -> %s", echelle_path, spectra_path)

    return spectra


def migrate_directory(
    directory: str | Path,
    dry_run: bool = True,
    backup: bool = True,
) -> dict:
    """Migrate all PyReduce files in a directory to new formats.

    Parameters
    ----------
    directory : str or Path
        Directory to scan for files to migrate.
    dry_run : bool
        If True, only report what would be done without making changes.
    backup : bool
        If True, keep original files with .bak extension.

    Returns
    -------
    dict
        Summary of migration results with keys:
        - traces: list of (old_path, new_path) for trace files
        - spectra: list of (old_path, new_path) for spectrum files
        - errors: list of (path, error_message)
    """
    directory = Path(directory)
    results = {"traces": [], "spectra": [], "errors": []}

    # Find trace NPZ files
    for npz_file in directory.glob("*.traces.npz"):
        fits_file = npz_file.with_suffix(".fits")
        if fits_file.exists():
            logger.info("Skipping %s (FITS already exists)", npz_file)
            continue

        if dry_run:
            logger.info("[DRY RUN] Would convert: %s -> %s", npz_file, fits_file)
            results["traces"].append((str(npz_file), str(fits_file)))
        else:
            try:
                convert_trace_npz_to_fits(npz_file, fits_file)
                if backup:
                    npz_file.rename(npz_file.with_suffix(".npz.bak"))
                results["traces"].append((str(npz_file), str(fits_file)))
            except Exception as e:
                logger.error("Failed to convert %s: %s", npz_file, e)
                results["errors"].append((str(npz_file), str(e)))

    # Find Echelle FITS files (science, final)
    for fits_file in directory.glob("*.science.fits"):
        fmt = detect_spectrum_format(fits_file)
        if fmt == "spectra_v2":
            logger.info("Skipping %s (already v2 format)", fits_file)
            continue

        if dry_run:
            logger.info("[DRY RUN] Would upgrade in place: %s", fits_file)
            results["spectra"].append((str(fits_file), str(fits_file)))
        else:
            try:
                convert_echelle_to_spectra(fits_file, fits_file)
                results["spectra"].append((str(fits_file), str(fits_file)))
            except Exception as e:
                logger.error("Failed to convert %s: %s", fits_file, e)
                results["errors"].append((str(fits_file), str(e)))

    return results
