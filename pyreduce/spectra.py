"""
Spectrum data model for PyReduce.

This module defines the Spectrum, ExtractionParams, and Spectra classes
for storing extracted spectral data.

Replaces the legacy Echelle class with cleaner semantics:
- NaN masking instead of COLUMNS + MASK redundancy
- Per-trace metadata (m, fiber, extraction_height)
- Un-normalized spectra with separate continuum
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import astropy.io.fits as fits
import numpy as np
import scipy.constants

if TYPE_CHECKING:
    from pyreduce.trace_model import Trace

logger = logging.getLogger(__name__)

# Format version for backwards compatibility detection
FORMAT_VERSION = 2


@dataclass
class ExtractionParams:
    """Global extraction parameters (same for all traces in a file).

    Attributes
    ----------
    osample : int
        Oversampling factor for slit function.
    lambda_sf : float
        Slitfunction smoothing parameter.
    lambda_sp : float
        Spectrum smoothing parameter.
    swath_width : int | None
        Swath width for extraction, or None for automatic.
    """

    osample: int
    lambda_sf: float
    lambda_sp: float
    swath_width: int | None = None

    def to_header(self, header: fits.Header) -> None:
        """Write extraction parameters to FITS header.

        Parameters
        ----------
        header : fits.Header
            Header to write to.
        """
        header["E_OSAMPLE"] = (self.osample, "Extraction oversampling")
        header["E_LAMBDASF"] = (self.lambda_sf, "Slitfunction smoothing")
        header["E_LAMBDASP"] = (self.lambda_sp, "Spectrum smoothing")
        if self.swath_width is not None:
            header["E_SWATHW"] = (self.swath_width, "Swath width")

    @classmethod
    def from_header(cls, header: fits.Header) -> ExtractionParams | None:
        """Read extraction parameters from FITS header.

        Parameters
        ----------
        header : fits.Header
            Header to read from.

        Returns
        -------
        ExtractionParams or None
            Extraction parameters, or None if not present.
        """
        osample = header.get("E_OSAMPLE")
        if osample is None:
            return None
        return cls(
            osample=int(osample),
            lambda_sf=float(header.get("E_LAMBDASF", 0)),
            lambda_sp=float(header.get("E_LAMBDASP", 0)),
            swath_width=header.get("E_SWATHW"),
        )


@dataclass
class Spectrum:
    """Output of extraction for one trace.

    Attributes
    ----------
    m : int | None
        Spectral order number. None if unknown.
    fiber : str | int
        Fiber identifier.
    spec : np.ndarray
        Flux values, un-normalized. NaN for masked pixels.
    sig : np.ndarray
        Uncertainty values. NaN for masked pixels.
    wave : np.ndarray | None
        Wavelength values (evaluated from polynomial). Same length as spec.
    cont : np.ndarray | None
        Continuum values (full array, not polynomial). Same length as spec.
    slitfu : np.ndarray | None
        Slit function (shape depends on osample: height * osample + 1).
    extraction_height : float | None
        Extraction aperture used for this trace.
    """

    # Identity (copied from Trace)
    m: int | None
    fiber: str | int

    # Extracted data (NaN for masked pixels)
    spec: np.ndarray
    sig: np.ndarray

    # Optional data
    wave: np.ndarray | None = None
    cont: np.ndarray | None = None
    slitfu: np.ndarray | None = None

    # Per-trace extraction param
    extraction_height: float | None = None

    @classmethod
    def from_trace(
        cls, trace: Trace, spec: np.ndarray, sig: np.ndarray, **kwargs
    ) -> Spectrum:
        """Factory method that copies identity from Trace.

        Parameters
        ----------
        trace : Trace
            Source trace for identity (m, fiber).
        spec : np.ndarray
            Extracted spectrum.
        sig : np.ndarray
            Spectrum uncertainty.
        **kwargs
            Additional fields (wave, cont, slitfu, extraction_height).

        Returns
        -------
        Spectrum
            New spectrum with identity copied from trace.
        """
        return cls(m=trace.m, fiber=trace.fiber, spec=spec, sig=sig, **kwargs)

    def normalized(self) -> tuple[np.ndarray, np.ndarray]:
        """Return continuum-normalized spectrum and uncertainty.

        Returns
        -------
        spec_norm : np.ndarray
            Spectrum divided by continuum.
        sig_norm : np.ndarray
            Uncertainty divided by continuum.

        Raises
        ------
        ValueError
            If no continuum is available.
        """
        if self.cont is None:
            raise ValueError("No continuum available for normalization")
        return self.spec / self.cont, self.sig / self.cont

    @property
    def mask(self) -> np.ndarray:
        """Boolean mask where True indicates invalid (masked) pixels."""
        return np.isnan(self.spec)


@dataclass
class Spectra:
    """Container for multiple spectra from one observation.

    Replaces the legacy Echelle class.

    Attributes
    ----------
    header : fits.Header
        FITS header with observation metadata.
    data : list[Spectrum]
        Individual spectra, one per trace.
    params : ExtractionParams | None
        Global extraction parameters (stored in header).
    """

    header: fits.Header
    data: list[Spectrum]
    params: ExtractionParams | None = None

    @property
    def ntrace(self) -> int:
        """Number of traces/spectra."""
        return len(self.data)

    @property
    def ncol(self) -> int:
        """Number of columns (pixels) in spectra."""
        if not self.data:
            return 0
        return len(self.data[0].spec)

    def select(
        self, m: int | None = None, fiber: str | int | None = None
    ) -> list[Spectrum]:
        """Filter spectra by order and/or fiber.

        Parameters
        ----------
        m : int, optional
            Select spectra with this spectral order number.
        fiber : str or int, optional
            Select spectra with this fiber identifier.

        Returns
        -------
        list[Spectrum]
            Matching spectra.
        """
        result = self.data
        if m is not None:
            result = [s for s in result if s.m == m]
        if fiber is not None:
            result = [s for s in result if s.fiber == fiber]
        return result

    def get_arrays(self) -> dict[str, np.ndarray]:
        """Get stacked arrays for all spectra.

        Returns
        -------
        dict
            Dictionary with keys 'spec', 'sig', 'wave', 'cont', 'm', 'fiber'.
            Arrays are stacked along axis 0 (one row per trace).
        """
        return {
            "spec": np.array([s.spec for s in self.data]),
            "sig": np.array([s.sig for s in self.data]),
            "wave": np.array([s.wave for s in self.data])
            if self.data[0].wave is not None
            else None,
            "cont": np.array([s.cont for s in self.data])
            if self.data[0].cont is not None
            else None,
            "m": np.array([s.m if s.m is not None else -1 for s in self.data]),
            "fiber": np.array([str(s.fiber) for s in self.data]),
        }

    @staticmethod
    def read(
        fname: str | Path,
        raw: bool = False,
        continuum_normalization: bool = False,
        barycentric_correction: bool = True,
        radial_velocity_correction: bool = True,
    ) -> Spectra:
        """Read spectra from a FITS file.

        Supports both new format (E_FMTVER >= 2) and legacy format.

        Parameters
        ----------
        fname : str or Path
            Input file path.
        raw : bool
            If True, skip all corrections.
        continuum_normalization : bool
            Apply continuum normalization (default False for new format).
        barycentric_correction : bool
            Apply barycentric correction to wavelength.
        radial_velocity_correction : bool
            Apply radial velocity correction to wavelength.

        Returns
        -------
        Spectra
            Loaded spectra.
        """
        with fits.open(fname, memmap=False) as hdu:
            header = hdu[0].header
            table = hdu[1].data

        fmtver = header.get("E_FMTVER", 1)

        if fmtver >= 2:
            return _read_new_format(
                fname,
                header,
                table,
                raw,
                continuum_normalization,
                barycentric_correction,
                radial_velocity_correction,
            )
        else:
            return _read_legacy_format(
                fname,
                header,
                table,
                raw,
                continuum_normalization,
                barycentric_correction,
                radial_velocity_correction,
            )

    def save(self, fname: str | Path, steps: list[str] = None) -> None:
        """Save spectra to a FITS file.

        Parameters
        ----------
        fname : str or Path
            Output file path.
        steps : list[str], optional
            Pipeline steps that have been run.
        """
        header = self.header.copy() if self.header else fits.Header()

        # Add format metadata
        header["E_FMTVER"] = (FORMAT_VERSION, "PyReduce format version")
        if steps:
            header["E_STEPS"] = (",".join(steps), "Pipeline steps run")

        # Add extraction parameters
        if self.params:
            self.params.to_header(header)

        ntrace = self.ntrace
        ncol = self.ncol

        # Stack arrays
        spec_arr = np.array([s.spec for s in self.data], dtype=np.float32)
        sig_arr = np.array([s.sig for s in self.data], dtype=np.float32)
        m_arr = np.array(
            [s.m if s.m is not None else -1 for s in self.data], dtype=np.int16
        )
        fiber_arr = np.array([str(s.fiber) for s in self.data], dtype="U16")
        height_arr = np.array(
            [
                s.extraction_height if s.extraction_height is not None else np.nan
                for s in self.data
            ],
            dtype=np.float32,
        )

        # Build columns
        columns = [
            fits.Column(name="SPEC", format=f"{ncol}E", array=spec_arr),
            fits.Column(name="SIG", format=f"{ncol}E", array=sig_arr),
            fits.Column(name="M", format="I", array=m_arr),
            fits.Column(name="FIBER", format="16A", array=fiber_arr),
            fits.Column(name="EXTR_H", format="E", array=height_arr),
        ]

        # Optional: wavelength
        has_wave = any(s.wave is not None for s in self.data)
        if has_wave:
            wave_arr = np.array(
                [
                    s.wave if s.wave is not None else np.full(ncol, np.nan)
                    for s in self.data
                ],
                dtype=np.float64,
            )
            columns.append(fits.Column(name="WAVE", format=f"{ncol}D", array=wave_arr))

        # Optional: continuum
        has_cont = any(s.cont is not None for s in self.data)
        if has_cont:
            cont_arr = np.array(
                [
                    s.cont if s.cont is not None else np.full(ncol, np.nan)
                    for s in self.data
                ],
                dtype=np.float32,
            )
            columns.append(fits.Column(name="CONT", format=f"{ncol}E", array=cont_arr))

        # Optional: slit function
        has_slitfu = any(s.slitfu is not None for s in self.data)
        if has_slitfu:
            max_slitfu_len = max(
                len(s.slitfu) if s.slitfu is not None else 0 for s in self.data
            )
            slitfu_arr = np.full((ntrace, max_slitfu_len), np.nan, dtype=np.float32)
            for i, s in enumerate(self.data):
                if s.slitfu is not None:
                    slitfu_arr[i, : len(s.slitfu)] = s.slitfu
            columns.append(
                fits.Column(
                    name="SLITFU", format=f"{max_slitfu_len}E", array=slitfu_arr
                )
            )

        # Create HDU list
        primary = fits.PrimaryHDU(header=header)
        table = fits.BinTableHDU.from_columns(columns, name="SPECTRA")

        hdulist = fits.HDUList([primary, table])
        hdulist.writeto(fname, overwrite=True, output_verify="silentfix+ignore")
        logger.info("Saved %d spectra to: %s", ntrace, fname)


def _read_new_format(
    fname,
    header,
    table,
    raw,
    continuum_normalization,
    barycentric_correction,
    radial_velocity_correction,
) -> Spectra:
    """Read new format (E_FMTVER >= 2) spectra."""
    spec_arr = table["SPEC"]
    sig_arr = table["SIG"]
    m_arr = table["M"]
    fiber_arr = table["FIBER"]

    height_arr = table["EXTR_H"] if "EXTR_H" in table.dtype.names else None
    wave_arr = table["WAVE"] if "WAVE" in table.dtype.names else None
    cont_arr = table["CONT"] if "CONT" in table.dtype.names else None
    slitfu_arr = table["SLITFU"] if "SLITFU" in table.dtype.names else None

    # Apply wavelength corrections
    if not raw and wave_arr is not None:
        velocity_correction = 0
        if barycentric_correction:
            velocity_correction -= header.get("barycorr", 0)
        if radial_velocity_correction:
            velocity_correction += header.get("radvel", 0)
        if velocity_correction != 0:
            speed_of_light = scipy.constants.speed_of_light * 1e-3
            wave_arr = wave_arr * (1 + velocity_correction / speed_of_light)

    # Apply continuum normalization if requested
    if not raw and continuum_normalization and cont_arr is not None:
        spec_arr = spec_arr / cont_arr
        sig_arr = sig_arr / cont_arr

    params = ExtractionParams.from_header(header)

    spectra = []
    for i in range(len(spec_arr)):
        m = int(m_arr[i]) if m_arr[i] >= 0 else None
        fiber = fiber_arr[i].strip()
        try:
            fiber = int(fiber)
        except ValueError:
            pass

        spec = spec_arr[i]
        sig = sig_arr[i]
        wave = wave_arr[i] if wave_arr is not None else None
        cont = cont_arr[i] if cont_arr is not None else None
        slitfu = slitfu_arr[i] if slitfu_arr is not None else None
        height = (
            float(height_arr[i])
            if height_arr is not None and not np.isnan(height_arr[i])
            else None
        )

        # Remove NaN-padding from slitfu
        if slitfu is not None:
            slitfu = slitfu[~np.isnan(slitfu)]
            if len(slitfu) == 0:
                slitfu = None

        spectra.append(
            Spectrum(
                m=m,
                fiber=fiber,
                spec=spec,
                sig=sig,
                wave=wave,
                cont=cont,
                slitfu=slitfu,
                extraction_height=height,
            )
        )

    return Spectra(header=header, data=spectra, params=params)


def _read_legacy_format(
    fname,
    header,
    table,
    raw,
    continuum_normalization,
    barycentric_correction,
    radial_velocity_correction,
) -> Spectra:
    """Read legacy format (E_FMTVER < 2 or missing) spectra."""
    # Legacy format uses lowercase keys from the table
    _data = {col.lower(): table[col][0] for col in table.dtype.names}

    spec_arr = _data.get("spec")
    sig_arr = _data.get("sig")
    wave_arr = _data.get("wave")
    cont_arr = _data.get("cont")
    columns_arr = _data.get("columns")

    if spec_arr is None:
        raise ValueError(f"No SPEC column found in {fname}")

    ntrace, ncol = spec_arr.shape

    # Expand polynomials if needed
    if not raw:
        if wave_arr is not None:
            wave_arr = _expand_polynomial(ncol, wave_arr)

            velocity_correction = 0
            if barycentric_correction:
                velocity_correction -= header.get("barycorr", 0)
            if radial_velocity_correction:
                velocity_correction += header.get("radvel", 0)
            if velocity_correction != 0:
                speed_of_light = scipy.constants.speed_of_light * 1e-3
                wave_arr = wave_arr * (1 + velocity_correction / speed_of_light)

        if cont_arr is not None:
            cont_arr = _expand_polynomial(ncol, cont_arr)

    # Create mask from columns (legacy behavior)
    if columns_arr is not None:
        mask = np.full((ntrace, ncol), True)
        for i in range(ntrace):
            mask[i, columns_arr[i, 0] : columns_arr[i, 1]] = False
        # Apply mask as NaN
        spec_arr = np.where(mask, np.nan, spec_arr)
        sig_arr = np.where(mask, np.nan, sig_arr)

    # Apply continuum normalization
    if not raw and continuum_normalization and cont_arr is not None:
        spec_arr = spec_arr / cont_arr
        sig_arr = sig_arr / cont_arr

    # Build spectra (no m/fiber info in legacy format)
    spectra = []
    for i in range(ntrace):
        spectra.append(
            Spectrum(
                m=i,  # Sequential, no real order number
                fiber=0,  # Default fiber
                spec=spec_arr[i],
                sig=sig_arr[i],
                wave=wave_arr[i] if wave_arr is not None else None,
                cont=cont_arr[i] if cont_arr is not None else None,
            )
        )

    return Spectra(header=header, data=spectra, params=None)


def _expand_polynomial(ncol: int, poly: np.ndarray) -> np.ndarray:
    """Expand polynomial coefficients to full array if needed.

    Handles three formats:
    1. 1D array (REDUCE make_wave format) - full 2D expansion
    2. Shape (ntrace, degree+1) - 1D polynomial per order
    3. Already expanded (ntrace, ncol) - pass through
    """
    if poly.ndim == 1:
        return _calc_2dpolynomial(poly)
    elif poly.shape[1] < 20:
        return _calc_1dpolynomials(ncol, poly)
    return poly


def _calc_2dpolynomial(solution2d: np.ndarray) -> np.ndarray:
    """Expand a 2D polynomial in REDUCE make_wave format."""
    ncol = int(solution2d[1])
    ntrace = int(solution2d[2])
    order_base = int(solution2d[3])
    deg_cross, deg_column, deg_order = (
        int(solution2d[7]),
        int(solution2d[8]),
        int(solution2d[9]),
    )
    coeff_in = solution2d[10:]

    coeff = np.zeros((deg_order + 1, deg_column + 1))
    coeff[0, 0] = coeff_in[0]
    coeff[0, 1:] = coeff_in[1 : 1 + deg_column]
    coeff[1:, 0] = coeff_in[1 + deg_column : 1 + deg_column + deg_order]
    if deg_cross in [4, 6]:
        coeff[1, 1] = coeff_in[deg_column + deg_order + 1]
        coeff[1, 2] = coeff_in[deg_column + deg_order + 2]
        coeff[2, 1] = coeff_in[deg_column + deg_order + 3]
        coeff[2, 2] = coeff_in[deg_column + deg_order + 4]
    if deg_cross == 6:
        coeff[1, 3] = coeff_in[deg_column + deg_order + 5]
        coeff[3, 1] = coeff_in[deg_column + deg_order + 6]

    x = np.arange(order_base, order_base + ntrace, dtype=float)
    y = np.arange(ncol, dtype=float)

    return np.polynomial.polynomial.polygrid2d(x / 100, y / 1000, coeff) / x[:, None]


def _calc_1dpolynomials(ncol: int, poly: np.ndarray) -> np.ndarray:
    """Expand 1D polynomials (one per order)."""
    ntrace = poly.shape[0]
    x = np.arange(ncol)
    result = np.zeros((ntrace, ncol))
    for i, coef in enumerate(poly):
        result[i] = np.polyval(coef, x)
    return result


# Convenience function for backwards compatibility
def read(fname: str | Path, **kwargs) -> Spectra:
    """Read spectra from a file.

    Alias for Spectra.read().
    """
    return Spectra.read(fname, **kwargs)
