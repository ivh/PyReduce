"""
Slit curvature data model for charslit extraction.

This module defines the SlitCurvature dataclass and I/O functions
for storing curvature coefficients and slitdeltas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SlitCurvature:
    """Container for slit curvature parameters.

    Attributes
    ----------
    coeffs : np.ndarray
        Polynomial coefficients of shape (ntrace, ncol, degree+1).
        coeffs[trace, col, :] gives the polynomial for that trace/column.
        Coefficient order: [c0, c1, c2, ...] where y_offset = c0 + c1*y + c2*y^2 + ...
    slitdeltas : np.ndarray | None
        Per-row residual offsets of shape (ntrace, nrow), or None if not computed.
        These capture deviations not modeled by the polynomial.
    degree : int
        Polynomial degree (1-5).
    """

    coeffs: np.ndarray
    slitdeltas: np.ndarray | None
    degree: int

    def get_coeffs_for_trace(self, trace_idx: int, pad_to: int = 6) -> np.ndarray:
        """Get coefficients for a single trace, optionally padded.

        Parameters
        ----------
        trace_idx : int
            Index of the trace.
        pad_to : int
            Pad coefficients to this many terms (default 6 for charslit).

        Returns
        -------
        np.ndarray
            Coefficients of shape (ncol, pad_to).
        """
        ncol = self.coeffs.shape[1]
        result = np.zeros((ncol, pad_to), dtype=np.float64)
        n_coeffs = min(self.degree + 1, pad_to)
        result[:, :n_coeffs] = self.coeffs[trace_idx, :, :n_coeffs]
        return result

    def get_slitdeltas_for_trace(self, trace_idx: int, nrow: int) -> np.ndarray:
        """Get slitdeltas for a single trace.

        Parameters
        ----------
        trace_idx : int
            Index of the trace.
        nrow : int
            Number of rows expected.

        Returns
        -------
        np.ndarray
            Slitdeltas of shape (nrow,), zeros if not available.
        """
        if self.slitdeltas is None:
            return np.zeros(nrow, dtype=np.float64)
        return self.slitdeltas[trace_idx]

    def to_p1_p2(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract legacy p1, p2 arrays for backward compatibility.

        Returns
        -------
        p1 : np.ndarray
            Linear curvature coefficients of shape (ntrace, ncol).
        p2 : np.ndarray
            Quadratic curvature coefficients of shape (ntrace, ncol).
        """
        ntrace, ncol, _ = self.coeffs.shape
        p1 = self.coeffs[:, :, 1] if self.degree >= 1 else np.zeros((ntrace, ncol))
        p2 = self.coeffs[:, :, 2] if self.degree >= 2 else np.zeros((ntrace, ncol))
        return p1, p2


def save_curvature(path: str | Path, curvature: SlitCurvature) -> None:
    """Save curvature data to an npz file.

    Parameters
    ----------
    path : str | Path
        Output file path.
    curvature : SlitCurvature
        Curvature data to save.
    """
    np.savez(
        path,
        version=2,
        degree=curvature.degree,
        coeffs=curvature.coeffs,
        slitdeltas=curvature.slitdeltas,
    )
    logger.info("Saved curvature to: %s", path)


def load_curvature(path: str | Path) -> SlitCurvature:
    """Load curvature data from an npz file.

    Supports both old format (p1, p2 arrays) and new format (coeffs, slitdeltas).

    Parameters
    ----------
    path : str | Path
        Input file path.

    Returns
    -------
    SlitCurvature
        Loaded curvature data.
    """
    data = np.load(path, allow_pickle=True)

    version = int(data.get("version", 1))

    if version == 1 or "p1" in data:
        # Old format: p1, p2 arrays of shape (ntrace, ncol)
        p1 = data["p1"]
        p2 = data["p2"]

        if p1 is None or p2 is None:
            return None

        ntrace, ncol = p1.shape
        degree = 2 if np.any(p2 != 0) else 1

        # Convert to new format: coeffs[trace, col, coef]
        coeffs = np.zeros((ntrace, ncol, degree + 1), dtype=np.float64)
        # c0 = 0 (no constant offset in old format)
        coeffs[:, :, 1] = p1  # linear term
        if degree == 2:
            coeffs[:, :, 2] = p2  # quadratic term

        logger.info("Loaded curvature from legacy format (version 1)")
        return SlitCurvature(coeffs=coeffs, slitdeltas=None, degree=degree)

    # New format
    coeffs = data["coeffs"]
    slitdeltas = data.get("slitdeltas")
    if slitdeltas is not None:
        slitdeltas = np.asarray(slitdeltas)
        if slitdeltas.ndim == 0:
            slitdeltas = None
    degree = int(data["degree"])

    logger.info("Loaded curvature (version %d, degree %d)", version, degree)
    return SlitCurvature(coeffs=coeffs, slitdeltas=slitdeltas, degree=degree)


def curvature_from_p1_p2(
    p1: np.ndarray, p2: np.ndarray, degree: int = 2
) -> SlitCurvature:
    """Create a SlitCurvature from legacy p1, p2 arrays.

    Parameters
    ----------
    p1 : np.ndarray
        Linear curvature coefficients of shape (ntrace, ncol).
    p2 : np.ndarray
        Quadratic curvature coefficients of shape (ntrace, ncol).
    degree : int
        Polynomial degree to use (default 2).

    Returns
    -------
    SlitCurvature
        Curvature data in new format.
    """
    ntrace, ncol = p1.shape
    coeffs = np.zeros((ntrace, ncol, degree + 1), dtype=np.float64)
    coeffs[:, :, 1] = p1
    if degree >= 2:
        coeffs[:, :, 2] = p2
    return SlitCurvature(coeffs=coeffs, slitdeltas=None, degree=degree)
