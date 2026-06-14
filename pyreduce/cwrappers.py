"""
Wrapper for the slit-decomposition C function.

This module provides access to the slitdec extraction algorithm in the
CFFI-compiled C library (clib/slitdec.c, copied from charslit) and sanitizes
the input parameters.

Mask convention here is charslit's: 0 = bad pixel, 1 = good pixel. No
conversion is done; callers pass and receive masks in that convention.
"""

import ctypes
import logging

import numpy as np

logger = logging.getLogger(__name__)

from .clib._slitdec import ffi as ffi_slitdec
from .clib._slitdec import lib as slitdeclib

c_double = ctypes.c_double
c_int = ctypes.c_int
c_mask = ctypes.c_ubyte


def slitdec(
    im,
    pix_unc,
    mask,
    ycen,
    slitcurve,
    slitdeltas,
    osample=6,
    lambda_sP=0.0,
    lambda_sL=1.0,
    maxiter=20,
    kappa=10.0,
    preset_slitfunc=None,
):
    """Slit decomposition with slit characterization (charslit algorithm).

    Plain-C port of charslit's ``slitdec``, compiled via CFFI. The signature and
    returned dict match the nanobind ``charslit.slitdec`` so this is a drop-in
    replacement for the external package.

    Mask convention here is charslit's (0 = bad, 1 = good); no conversion is done.

    When ``preset_slitfunc`` is given (length ``ny`` used directly, or ``nrows``
    interpolated up), the slit-function solve is skipped and only the spectrum is
    fit against it (single-pass extraction); the preset is normalized to sum
    ``osample`` inside the C code.

    Returns
    -------
    dict with keys spectrum, slitfunction, model, uncertainty, info, mask,
    return_code.
    """
    requirements = ["C", "A", "W", "O"]

    im = np.require(im, dtype=c_double, requirements=requirements)
    pix_unc = np.require(pix_unc, dtype=c_double, requirements=requirements)

    nrows, ncols = im.shape
    if pix_unc.shape != (nrows, ncols):
        raise ValueError("pix_unc must have same shape as im")
    if mask.shape != (nrows, ncols):
        raise ValueError("mask must have same shape as im")
    if ycen.shape[0] != ncols:
        raise ValueError("ycen must have length ncols")

    slitcurve = np.asarray(slitcurve, dtype=c_double)
    n_coeffs = slitcurve.shape[1]
    if slitcurve.shape[0] != ncols or not (1 <= n_coeffs <= 6):
        raise ValueError("slitcurve must have shape (ncols, n) where 1 <= n <= 6")

    osample = int(osample)
    ny = osample * (nrows + 1) + 1

    # slitdeltas may be given per-row (nrows) or per-subpixel (ny). Per-row is
    # linearly interpolated up to ny, matching the charslit wrapper.
    slitdeltas = np.asarray(slitdeltas, dtype=c_double).ravel()
    if slitdeltas.size == nrows:
        pos = np.arange(ny) * (nrows - 1.0) / (ny - 1.0)
        slitdeltas_ny = np.interp(pos, np.arange(nrows), slitdeltas)
    elif slitdeltas.size == ny:
        slitdeltas_ny = slitdeltas.copy()
    else:
        raise ValueError(
            "slitdeltas must have length nrows or ny = osample * (nrows + 1) + 1"
        )
    slitdeltas_ny = np.require(slitdeltas_ny, dtype=c_double, requirements=requirements)

    # Pad slitcurve to 6 coefficients per column (C code uses stride 6)
    slitcurve_padded = np.zeros((ncols, 6), dtype=c_double)
    slitcurve_padded[:, :n_coeffs] = slitcurve
    slitcurve_padded = np.require(
        slitcurve_padded, dtype=c_double, requirements=requirements
    )

    # The algorithm modifies mask and ycen in place, so pass copies.
    mask_copy = np.require(mask, dtype=c_mask, requirements=requirements).copy()
    ycen_copy = np.require(ycen, dtype=c_double, requirements=requirements).copy()

    # Seed sL. With a preset slit function we copy it in and tell the C code to
    # skip the sL solve (single-pass extraction); the preset may be length ny
    # (used directly) or nrows (interpolated up, mirroring slitdeltas). The C
    # code normalizes it to sum osample. Otherwise seed the flat 1/osample guess.
    use_preset = 0
    if preset_slitfunc is not None:
        use_preset = 1
        ps = np.asarray(preset_slitfunc, dtype=c_double).ravel()
        if ps.size == ny:
            sL = ps.copy()
        elif ps.size == nrows:
            pos = np.arange(ny) * (nrows - 1.0) / (ny - 1.0)
            sL = np.interp(pos, np.arange(nrows), ps)
        else:
            raise ValueError(
                "preset_slitfunc must have length nrows or ny = osample * (nrows + 1) + 1"
            )
        sL = np.require(sL, dtype=c_double, requirements=requirements)
    else:
        sL = np.full(ny, 1.0 / osample, dtype=c_double)

    # Output arrays, with the same starting guesses as the charslit wrapper.
    sP = np.ones(ncols, dtype=c_double)
    model = np.zeros((nrows, ncols), dtype=c_double)
    unc = np.zeros(ncols, dtype=c_double)
    info = np.zeros(5, dtype=c_double)

    return_code = slitdeclib.slitdec(
        ffi_slitdec.cast("int", ncols),
        ffi_slitdec.cast("int", nrows),
        ffi_slitdec.cast("double *", im.ctypes.data),
        ffi_slitdec.cast("double *", pix_unc.ctypes.data),
        ffi_slitdec.cast("unsigned char *", mask_copy.ctypes.data),
        ffi_slitdec.cast("double *", ycen_copy.ctypes.data),
        ffi_slitdec.cast("double *", slitcurve_padded.ctypes.data),
        ffi_slitdec.cast("double *", slitdeltas_ny.ctypes.data),
        ffi_slitdec.cast("int", osample),
        ffi_slitdec.cast("double", float(lambda_sP)),
        ffi_slitdec.cast("double", float(lambda_sL)),
        ffi_slitdec.cast("int", int(maxiter)),
        ffi_slitdec.cast("double", float(kappa)),
        ffi_slitdec.cast("int", use_preset),
        ffi_slitdec.cast("double *", sP.ctypes.data),
        ffi_slitdec.cast("double *", sL.ctypes.data),
        ffi_slitdec.cast("double *", model.ctypes.data),
        ffi_slitdec.cast("double *", unc.ctypes.data),
        ffi_slitdec.cast("double *", info.ctypes.data),
    )

    return {
        "spectrum": sP,
        "slitfunction": sL,
        "model": model,
        "uncertainty": unc,
        "info": info,
        "mask": mask_copy,
        "return_code": int(return_code),
    }
