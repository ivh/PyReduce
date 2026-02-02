"""
Trace data model for PyReduce.

This module defines the Trace dataclass and I/O functions for storing
trace positions, curvature, and wavelength calibration in FITS format.

The Trace dataclass consolidates what was previously scattered across
separate files (traces.npz, curve.npz, wavecal.npz) into a single structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import astropy.io.fits as fits
import numpy as np

logger = logging.getLogger(__name__)

# Format version for backwards compatibility detection
FORMAT_VERSION = 2


@dataclass
class Trace:
    """Container for a single trace's geometry and calibration data.

    A trace represents a single spectral order (or fiber within an order)
    on the detector.

    Attributes
    ----------
    m : int | None
        Spectral order number (diffraction order). This is the physical order
        number from the grating equation, not a sequential index. In echelle
        spectrographs, higher order numbers correspond to shorter wavelengths.

        The order number is assigned in one of three ways:

        1. **From order_centers.yaml** (preferred): If the instrument provides
           an ``order_centers_{channel}.yaml`` file with known order positions,
           traces are matched to these centers during detection and assigned
           the corresponding order numbers immediately.

        2. **From wavelength calibration**: If no order_centers file exists,
           ``m`` is initially None. During wavelength calibration, the linelist
           file provides ``obase`` (the base order number). Each trace is then
           assigned ``m = obase + trace_index``.

        3. **Sequential fallback**: For legacy files or MOSAIC mode where order
           identity cannot be determined, ``m`` may remain None or be assigned
           sequentially from 0.

        The order number is critical for 2D wavelength calibration, which fits
        a polynomial in both pixel position (x) and order number (m). When
        evaluating wavelengths via ``Trace.wlen()``, the trace's ``m`` value
        is used as the second coordinate in the 2D polynomial.

    fiber : str | int
        Fiber identifier. String for named groups ('A', 'B', 'cal'),
        int for bundle indices.
    pos : np.ndarray
        y(x) trace position polynomial coefficients, shape (deg+1,).
        Coefficients in numpy.polyval order (highest power first).
    column_range : tuple[int, int]
        Valid x range [start, end) for this trace.
    height : float | None
        Extraction aperture height in pixels. None to use settings default.
    slit : np.ndarray | None
        Slit curvature coefficients, shape (deg_y+1, deg_x+1).
        Evaluates to x_offset = P(y) where P's coefficients vary with x.
        slit[i, :] are coefficients for the y^i term as a function of x.
    slitdelta : np.ndarray | None
        Per-row slit correction, shape (height_pixels,).
        Residual offsets beyond polynomial fit.
    wave : np.ndarray | None
        Wavelength polynomial coefficients. Can be:
        - 1D array, shape (deg+1,): per-trace polynomial, wavelength = polyval(x)
        - 2D array, shape (deg_x+1, deg_m+1): global 2D polynomial shared across
          all traces. Wavelength = polyval2d(x, m) where m is this trace's order.
    """

    # Identity
    m: int | None
    fiber: str | int

    # Geometry
    pos: np.ndarray
    column_range: tuple[int, int]
    height: float | None = None
    slit: np.ndarray | None = None
    slitdelta: np.ndarray | None = None

    # Calibration
    wave: np.ndarray | None = None

    def slit_at_x(self, x: float | np.ndarray) -> np.ndarray | None:
        """Evaluate slit polynomial coefficients at position x.

        Parameters
        ----------
        x : float or np.ndarray
            Column position(s) to evaluate at.

        Returns
        -------
        np.ndarray or None
            Polynomial coefficients for y_offset = c0 + c1*y + c2*y^2 + ...
            Shape (deg_y+1,) for scalar x, or (len(x), deg_y+1) for array x.
            Returns None if no slit curvature is set.
        """
        if self.slit is None:
            return None
        # slit[i, :] = coefficients for y^i term as function of x
        # Evaluate each row's polynomial at x
        return np.array([np.polyval(c, x) for c in self.slit])

    def wlen(self, x: np.ndarray) -> np.ndarray | None:
        """Evaluate wavelength polynomial at column positions.

        Parameters
        ----------
        x : np.ndarray
            Column positions to evaluate at.

        Returns
        -------
        np.ndarray or None
            Wavelength values at each x position.
            Returns None if no wavelength calibration is set.
        """
        if self.wave is None:
            return None
        if self.wave.ndim == 2:
            # 2D polynomial: wave[i,j] is coeff for x^i * m^j
            # polyval2d requires x and m arrays to have same shape
            m_arr = np.full_like(x, self.m, dtype=float)
            return np.polynomial.polynomial.polyval2d(x, m_arr, self.wave)
        else:
            # 1D polynomial: standard polyval
            return np.polyval(self.wave, x)

    def y_at_x(self, x: np.ndarray) -> np.ndarray:
        """Evaluate trace y-position at column positions.

        Parameters
        ----------
        x : np.ndarray
            Column positions to evaluate at.

        Returns
        -------
        np.ndarray
            Y positions of the trace center at each x.
        """
        return np.polyval(self.pos, x)


def save_traces(
    path: str | Path,
    traces: list[Trace],
    header: fits.Header = None,
    steps: list[str] = None,
) -> None:
    """Save traces to a FITS binary table.

    Parameters
    ----------
    path : str | Path
        Output file path.
    traces : list[Trace]
        Traces to save.
    header : fits.Header, optional
        FITS header to include. If None, a minimal header is created.
    steps : list[str], optional
        Pipeline steps that have been run (stored in E_STEPS header).
    """
    if not traces:
        raise ValueError("Cannot save empty trace list")

    if header is None:
        header = fits.Header()
    else:
        header = header.copy()

    # Add format metadata
    header["E_FMTVER"] = (FORMAT_VERSION, "PyReduce format version")
    if steps:
        header["E_STEPS"] = (",".join(steps), "Pipeline steps run")

    # Determine array sizes
    max_pos_deg = max(len(t.pos) for t in traces)

    # Determine wave dimensions - can be 1D (per-trace) or 2D (global poly)
    wave_shapes = [t.wave.shape if t.wave is not None else () for t in traces]
    wave_is_2d = any(len(s) == 2 for s in wave_shapes)
    if wave_is_2d:
        max_wave_x = max((s[0] if len(s) == 2 else 0) for s in wave_shapes)
        max_wave_m = max((s[1] if len(s) == 2 else 0) for s in wave_shapes)
        max_wave_deg = 0  # Not used for 2D
    else:
        max_wave_deg = max((s[0] if len(s) >= 1 else 0) for s in wave_shapes)
        max_wave_x = max_wave_m = 0

    max_slitdelta_len = max(
        (len(t.slitdelta) if t.slitdelta is not None else 0) for t in traces
    )

    # Determine slit dimensions (deg_y+1, deg_x+1)
    slit_shapes = [(t.slit.shape if t.slit is not None else (0, 0)) for t in traces]
    max_slit_y = max(s[0] for s in slit_shapes)
    max_slit_x = max(s[1] for s in slit_shapes)

    ntrace = len(traces)

    # Build arrays
    m_arr = np.array([t.m if t.m is not None else -1 for t in traces], dtype=np.int16)
    fiber_arr = np.array([str(t.fiber) for t in traces], dtype="U16")
    col_range_arr = np.array([t.column_range for t in traces], dtype=np.int32)
    height_arr = np.array(
        [t.height if t.height is not None else np.nan for t in traces], dtype=np.float32
    )

    pos_arr = np.zeros((ntrace, max_pos_deg), dtype=np.float64)
    for i, t in enumerate(traces):
        pos_arr[i, : len(t.pos)] = t.pos

    wave_arr = None
    if wave_is_2d and max_wave_x > 0 and max_wave_m > 0:
        # 2D wavelength polynomial
        wave_arr = np.full((ntrace, max_wave_x, max_wave_m), np.nan, dtype=np.float64)
        for i, t in enumerate(traces):
            if t.wave is not None and t.wave.ndim == 2:
                wx, wm = t.wave.shape
                wave_arr[i, :wx, :wm] = t.wave
    elif max_wave_deg > 0:
        # 1D wavelength polynomial per trace
        wave_arr = np.full((ntrace, max_wave_deg), np.nan, dtype=np.float64)
        for i, t in enumerate(traces):
            if t.wave is not None and t.wave.ndim == 1:
                wave_arr[i, : len(t.wave)] = t.wave

    slit_arr = None
    if max_slit_y > 0 and max_slit_x > 0:
        slit_arr = np.full((ntrace, max_slit_y, max_slit_x), np.nan, dtype=np.float64)
        for i, t in enumerate(traces):
            if t.slit is not None:
                sy, sx = t.slit.shape
                slit_arr[i, :sy, :sx] = t.slit

    slitdelta_arr = None
    if max_slitdelta_len > 0:
        slitdelta_arr = np.full((ntrace, max_slitdelta_len), np.nan, dtype=np.float32)
        for i, t in enumerate(traces):
            if t.slitdelta is not None:
                slitdelta_arr[i, : len(t.slitdelta)] = t.slitdelta

    # Build FITS columns
    columns = [
        fits.Column(name="M", format="I", array=m_arr),
        fits.Column(name="FIBER", format="16A", array=fiber_arr),
        fits.Column(name="POS", format=f"{max_pos_deg}D", array=pos_arr),
        fits.Column(name="COL_RANGE", format="2J", array=col_range_arr),
        fits.Column(name="HEIGHT", format="E", array=height_arr),
    ]

    if slit_arr is not None:
        slit_flat = slit_arr.reshape(ntrace, -1)
        columns.append(
            fits.Column(
                name="SLIT",
                format=f"{slit_flat.shape[1]}D",
                array=slit_flat,
                dim=f"({max_slit_x},{max_slit_y})",
            )
        )
        header["SLIT_Y"] = (max_slit_y, "Slit polynomial y-degree + 1")
        header["SLIT_X"] = (max_slit_x, "Slit polynomial x-degree + 1")

    if slitdelta_arr is not None:
        columns.append(
            fits.Column(
                name="SLITDELTA", format=f"{max_slitdelta_len}E", array=slitdelta_arr
            )
        )

    if wave_arr is not None:
        if wave_is_2d:
            wave_flat = wave_arr.reshape(ntrace, -1)
            columns.append(
                fits.Column(
                    name="WAVE",
                    format=f"{wave_flat.shape[1]}D",
                    array=wave_flat,
                    dim=f"({max_wave_m},{max_wave_x})",
                )
            )
            header["WAVE_X"] = (max_wave_x, "Wave polynomial x-degree + 1")
            header["WAVE_M"] = (max_wave_m, "Wave polynomial m-degree + 1")
        else:
            columns.append(
                fits.Column(name="WAVE", format=f"{max_wave_deg}D", array=wave_arr)
            )

    # Create HDU list
    primary = fits.PrimaryHDU(header=header)
    table = fits.BinTableHDU.from_columns(columns, name="TRACES")

    hdulist = fits.HDUList([primary, table])
    hdulist.writeto(path, overwrite=True, output_verify="silentfix+ignore")
    logger.info("Saved %d traces to: %s", ntrace, path)


def load_traces(path: str | Path) -> tuple[list[Trace], fits.Header]:
    """Load traces from a FITS file.

    Also supports loading legacy NPZ format for backwards compatibility.

    Parameters
    ----------
    path : str | Path
        Input file path (.fits or .npz).

    Returns
    -------
    traces : list[Trace]
        Loaded traces.
    header : fits.Header
        FITS header (empty for NPZ files).
    """
    path = Path(path)

    if path.suffix == ".npz":
        return _load_traces_npz(path)

    with fits.open(path, memmap=False) as hdu:
        header = hdu[0].header
        fmtver = header.get("E_FMTVER", 1)

        if fmtver < 2:
            logger.warning("Loading traces from old format (version %d)", fmtver)

        data = hdu["TRACES"].data

        m_arr = data["M"]
        fiber_arr = data["FIBER"]
        pos_arr = data["POS"]
        col_range_arr = data["COL_RANGE"]
        height_arr = data["HEIGHT"]

        slit_arr = data["SLIT"] if "SLIT" in data.dtype.names else None
        slitdelta_arr = data["SLITDELTA"] if "SLITDELTA" in data.dtype.names else None
        wave_arr = data["WAVE"] if "WAVE" in data.dtype.names else None

        # Reshape slit if present
        if slit_arr is not None:
            slit_y = header.get("SLIT_Y", 0)
            slit_x = header.get("SLIT_X", 0)
            if slit_y > 0 and slit_x > 0:
                slit_arr = slit_arr.reshape(-1, slit_y, slit_x)

        # Reshape wave if 2D polynomial
        wave_is_2d = False
        if wave_arr is not None:
            wave_x = header.get("WAVE_X", 0)
            wave_m = header.get("WAVE_M", 0)
            if wave_x > 0 and wave_m > 0:
                wave_arr = wave_arr.reshape(-1, wave_x, wave_m)
                wave_is_2d = True

        traces = []
        for i in range(len(m_arr)):
            m = int(m_arr[i]) if m_arr[i] >= 0 else None
            fiber = fiber_arr[i].strip()
            # Try to convert fiber to int if it looks like one
            try:
                fiber = int(fiber)
            except ValueError:
                pass

            # Remove trailing NaN/zeros from pos
            pos = pos_arr[i]
            pos = pos[~np.isnan(pos)] if np.any(np.isnan(pos)) else pos

            column_range = (int(col_range_arr[i, 0]), int(col_range_arr[i, 1]))
            height = float(height_arr[i]) if not np.isnan(height_arr[i]) else None

            slit = None
            if slit_arr is not None:
                slit = slit_arr[i]
                if np.all(np.isnan(slit)):
                    slit = None
                else:
                    # Remove all-NaN rows/cols
                    mask_y = ~np.all(np.isnan(slit), axis=1)
                    mask_x = ~np.all(np.isnan(slit), axis=0)
                    slit = slit[mask_y][:, mask_x]

            slitdelta = None
            if slitdelta_arr is not None:
                slitdelta = slitdelta_arr[i]
                if np.all(np.isnan(slitdelta)):
                    slitdelta = None
                else:
                    slitdelta = slitdelta[~np.isnan(slitdelta)]

            wave = None
            if wave_arr is not None:
                wave = wave_arr[i]
                if np.all(np.isnan(wave)):
                    wave = None
                elif wave_is_2d:
                    # 2D polynomial - remove all-NaN rows/cols
                    mask_x = ~np.all(np.isnan(wave), axis=1)
                    mask_m = ~np.all(np.isnan(wave), axis=0)
                    wave = wave[mask_x][:, mask_m]
                else:
                    # 1D polynomial - remove trailing NaN
                    wave = wave[~np.isnan(wave)]

            traces.append(
                Trace(
                    m=m,
                    fiber=fiber,
                    pos=pos,
                    column_range=column_range,
                    height=height,
                    slit=slit,
                    slitdelta=slitdelta,
                    wave=wave,
                )
            )

        logger.info("Loaded %d traces from: %s", len(traces), path)
        return traces, header


def _load_traces_npz(path: Path) -> tuple[list[Trace], fits.Header]:
    """Load traces from legacy NPZ format.

    This handles the old format where traces, column_range, and heights
    were stored as separate arrays without order/fiber identity.

    Parameters
    ----------
    path : Path
        Input NPZ file path.

    Returns
    -------
    traces : list[Trace]
        Loaded traces (m and fiber assigned sequentially).
    header : fits.Header
        Empty header.
    """
    data = np.load(path, allow_pickle=True)

    # Handle old 'orders' key name
    if "orders" in data and "traces" not in data:
        trace_coeffs = data["orders"]
    else:
        trace_coeffs = data["traces"]

    column_range = data["column_range"]

    # Heights may or may not be present
    heights = data.get("heights", None)
    if heights is not None and heights.ndim == 0:
        heights = None

    traces = []
    for i in range(len(trace_coeffs)):
        height = (
            float(heights[i])
            if heights is not None and not np.isnan(heights[i])
            else None
        )
        traces.append(
            Trace(
                m=i,  # Sequential order number (no identity preserved)
                fiber=0,  # Default fiber
                pos=trace_coeffs[i],
                column_range=(int(column_range[i, 0]), int(column_range[i, 1])),
                height=height,
            )
        )

    logger.info("Loaded %d traces from legacy NPZ: %s", len(traces), path)
    return traces, fits.Header()
