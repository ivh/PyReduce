"""
Wrapper for REDUCE C functions

This module provides access to the extraction algorithms in the
C libraries and sanitizes the input parameters.

Mask convention:
    - FITS files and Python: numpy convention (1/True = bad/masked pixel)
    - C code uses REDUCE convention: 1 = good pixel, 0 = bad pixel
    All conversions happen at the boundary in this module.
"""

import ctypes
import logging

import numpy as np

logger = logging.getLogger(__name__)

from .clib._slitfunc_2d import ffi
from .clib._slitfunc_2d import lib as slitfunc_2dlib
from .clib._slitfunc_bd import lib as slitfunclib

c_double = ctypes.c_double
c_int = ctypes.c_int
c_mask = ctypes.c_ubyte


def _mask_to_c(mask: np.ndarray, dtype=None) -> np.ndarray:
    """Convert numpy mask (True=bad) to C mask (1=good, 0=bad)."""
    result = np.where(mask, 0, 1)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def _mask_from_c(mask: np.ndarray) -> np.ndarray:
    """Convert C mask (1=good, 0=bad) to numpy mask (True=bad)."""
    return mask == 0


def slitfunc(img, ycen, lambda_sp=0, lambda_sf=0.1, osample=1):
    """Decompose image into spectrum and slitfunction

    This is for horizontal straight orders only, for curved orders use slitfunc_curved instead

    Parameters
    ----------
    img : array[n, m]
        image to decompose, should just contain a small part of the overall image
    ycen : array[n]
        traces the center of the order along the image, relative to the center of the image?
    lambda_sp : float, optional
        smoothing parameter of the spectrum (the default is 0, which no smoothing)
    lambda_sf : float, optional
        smoothing parameter of the slitfunction (the default is 0.1, which )
    osample : int, optional
        Subpixel ovsersampling factor (the default is 1, which no oversampling)

    Returns
    -------
    sp, sl, model, unc
        spectrum, slitfunction, model, spectrum uncertainties
    """

    # Convert input to expected datatypes
    lambda_sf = float(lambda_sf)
    lambda_sp = float(lambda_sp)
    osample = int(osample)
    img = np.asanyarray(img, dtype=c_double)
    ycen = np.asarray(ycen, dtype=c_double)

    assert img.ndim == 2, "Image must be 2 dimensional"
    assert ycen.ndim == 1, "Ycen must be 1 dimensional"

    assert img.shape[1] == ycen.size, (
        f"Image and Ycen shapes are incompatible, got {img.shape} and {ycen.shape}"
    )

    assert osample > 0, f"Oversample rate must be positive, but got {osample}"
    assert lambda_sf >= 0, (
        f"Slitfunction smoothing must be positive, but got {lambda_sf}"
    )
    assert lambda_sp >= 0, f"Spectrum smoothing must be positive, but got {lambda_sp}"

    # Get some derived values
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1
    ycen = ycen - ycen.astype(c_int)

    # Prepare all arrays
    # Inital guess for slit function and spectrum
    sp = np.ma.sum(img, axis=0)
    requirements = ["C", "A", "W", "O"]
    sp = np.require(sp, dtype=c_double, requirements=requirements)

    sl = np.zeros(ny, dtype=c_double)

    mask = _mask_to_c(np.ma.getmaskarray(img), dtype=c_int)
    mask = np.require(mask, dtype=c_int, requirements=requirements)

    img = np.ma.getdata(img)
    img = np.require(img, dtype=c_double, requirements=requirements)

    pix_unc = np.zeros_like(img)
    pix_unc = np.require(pix_unc, dtype=c_double, requirements=requirements)

    ycen = np.require(ycen, dtype=c_double, requirements=requirements)
    model = np.zeros((nrows, ncols), dtype=c_double)
    unc = np.zeros(ncols, dtype=c_double)

    # Call the C function
    slitfunclib.slit_func_vert(
        ffi.cast("int", ncols),
        ffi.cast("int", nrows),
        ffi.cast("double *", img.ctypes.data),
        ffi.cast("double *", pix_unc.ctypes.data),
        ffi.cast("int *", mask.ctypes.data),
        ffi.cast("double *", ycen.ctypes.data),
        ffi.cast("int", osample),
        ffi.cast("double", lambda_sp),
        ffi.cast("double", lambda_sf),
        ffi.cast("double *", sp.ctypes.data),
        ffi.cast("double *", sl.ctypes.data),
        ffi.cast("double *", model.ctypes.data),
        ffi.cast("double *", unc.ctypes.data),
    )
    mask = _mask_from_c(mask)

    return sp, sl, model, unc, mask


def slitfunc_curved(
    img,
    ycen,
    p1,
    p2,
    lambda_sp,
    lambda_sf,
    osample,
    yrange,
    maxiter=20,
    gain=1,
    reject_threshold=6,
    preset_slitfunc=None,
):
    """Decompose an image into a spectrum and a slitfunction, image may be curved

    Parameters
    ----------
    img : array[n, m]
        input image
    ycen : array[n]
        traces the center of the order
    p1 : array[n]
        1st order curvature of the order along the image, set to 0 if order straight
    p2 : array[n]
        2nd order curvature of the order along the image, set to 0 if order straight
    osample : int
        Subpixel ovsersampling factor (the default is 1, no oversampling)
    lambda_sp : float
        smoothing factor spectrum (the default is 0, no smoothing)
    lambda_sl : float
        smoothing factor slitfunction (the default is 0.1, small smoothing)
    yrange : array[2]
        number of pixels below and above the central line that have been cut out
    maxiter : int, optional
        maximumim number of iterations, by default 20
    gain : float, optional
        gain of the image, by default 1
    reject_threshold : float, optional
        outlier rejection threshold in sigma, by default 6. Set to 0 to disable.
    preset_slitfunc : array[ny], optional
        If provided, use this slit function instead of solving for it.
        Size must be osample * (nrows + 1) + 1.

    Returns
    -------
    sp, sl, model, unc
        spectrum, slitfunction, model, spectrum uncertainties
    """

    # Convert datatypes to expected values
    lambda_sf = float(lambda_sf)
    lambda_sp = float(lambda_sp)
    osample = int(osample)
    maxiter = int(maxiter)
    img = np.asanyarray(img, dtype=c_double)
    ycen = np.asarray(ycen, dtype=c_double)
    yrange = np.asarray(yrange, dtype=int)

    assert img.ndim == 2, "Image must be 2 dimensional"
    assert ycen.ndim == 1, "Ycen must be 1 dimensional"
    assert maxiter > 0, "Maximum iterations must be positive"

    if np.isscalar(p1):
        p1 = np.full(img.shape[1], p1, dtype=c_double)
    else:
        p1 = np.asarray(p1, dtype=c_double)
    if np.isscalar(p2):
        p2 = np.full(img.shape[1], p2, dtype=c_double)
    else:
        p2 = np.asarray(p2, dtype=c_double)

    assert img.shape[1] == ycen.size, (
        f"Image and Ycen shapes are incompatible, got {img.shape} and {ycen.shape}"
    )
    assert img.shape[1] == p1.size, (
        f"Image and p1 shapes are incompatible, got {img.shape} and {p1.shape}"
    )
    assert img.shape[1] == p2.size, (
        f"Image and p2 shapes are incompatible, got {img.shape} and {p2.shape}"
    )

    assert osample > 0, f"Oversample rate must be positive, but got {osample}"
    assert lambda_sf >= 0, (
        f"Slitfunction smoothing must be positive, but got {lambda_sf}"
    )
    assert lambda_sp >= 0, f"Spectrum smoothing must be positive, but got {lambda_sp}"

    # assert np.ma.all(np.isfinite(img)), "All values in the image must be finite"
    assert np.all(np.isfinite(ycen)), "All values in ycen must be finite"
    assert np.all(np.isfinite(p1)), "All values in p1 must be finite"
    assert np.all(np.isfinite(p2)), "All values in p2 must be finite"

    assert yrange.ndim == 1, "Yrange must be 1 dimensional"
    assert yrange.size == 2, "Yrange must have 2 elements"
    assert yrange[0] + yrange[1] + 1 == img.shape[0], (
        "Yrange must cover the whole image"
    )
    assert yrange[0] >= 0, "Yrange must be positive"
    assert yrange[1] >= 0, "Yrange must be positive"

    # Retrieve some derived values
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    ycen_offset = ycen.astype(c_int)
    ycen_int = ycen - ycen_offset
    y_lower_lim = int(yrange[0])

    mask = np.ma.getmaskarray(img)
    img = np.ma.getdata(img)
    mask2 = ~np.isfinite(img)
    img[mask2] = 0
    mask |= ~np.isfinite(img)

    # Initial spectrum guess: median with outlier rejection, scaled to sum-equivalent
    img_masked = np.ma.array(img, mask=mask)
    if reject_threshold > 0:
        col_median = np.ma.median(img_masked, axis=0)
        col_std = np.ma.std(img_masked, axis=0)
        outliers = np.abs(img - col_median) > reject_threshold * col_std
        img_masked = np.ma.array(img, mask=mask | outliers)
    sp = np.ma.median(img_masked, axis=0).filled(0) * nrows

    mask = _mask_to_c(mask, dtype=c_mask)
    # Determine the shot noise
    # by converting electrons to photonsm via the gain
    pix_unc = np.nan_to_num(np.abs(img), copy=False)
    pix_unc *= gain
    np.sqrt(pix_unc, out=pix_unc)
    pix_unc[pix_unc < 1] = 1

    psf_curve = np.zeros((ncols, 3), dtype=c_double)
    psf_curve[:, 1] = p1
    psf_curve[:, 2] = p2

    # Initialize arrays and ensure the correct datatype for C
    requirements = ["C", "A", "W", "O"]
    sp = np.require(sp, dtype=c_double, requirements=requirements)
    mask = np.require(mask, dtype=c_mask, requirements=requirements)
    img = np.require(img, dtype=c_double, requirements=requirements)
    pix_unc = np.require(pix_unc, dtype=c_double, requirements=requirements)
    ycen_int = np.require(ycen_int, dtype=c_double, requirements=requirements)
    ycen_offset = np.require(ycen_offset, dtype=c_int, requirements=requirements)

    # This memory could be reused between swaths
    model = np.zeros((nrows, ncols), dtype=c_double)
    unc = np.zeros(ncols, dtype=c_double)

    # Handle preset slit function
    use_preset = preset_slitfunc is not None
    if use_preset:
        sl = np.require(
            preset_slitfunc, dtype=c_double, requirements=requirements
        ).copy()
        if sl.size != ny:
            raise ValueError(
                f"preset_slitfunc size {sl.size} doesn't match expected {ny}"
            )
    else:
        sl = np.zeros(ny, dtype=c_double)

    # Info contains the folowing: sucess, cost, status, iteration, delta_x
    info = np.zeros(5, dtype=c_double)

    col = np.sum(mask, axis=0) == 0
    if np.any(col):
        mask[mask.shape[0] // 2, col] = 1
    # assert not np.any(np.sum(mask, axis=0) == 0), "At least one mask column is all 0."

    # Call the C function
    slitfunc_2dlib.slit_func_curved(
        ffi.cast("int", ncols),
        ffi.cast("int", nrows),
        ffi.cast("int", ny),
        ffi.cast("double *", img.ctypes.data),
        ffi.cast("double *", pix_unc.ctypes.data),
        ffi.cast("unsigned char *", mask.ctypes.data),
        ffi.cast("double *", ycen_int.ctypes.data),
        ffi.cast("int *", ycen_offset.ctypes.data),
        ffi.cast("int", y_lower_lim),
        ffi.cast("int", osample),
        ffi.cast("double", lambda_sp),
        ffi.cast("double", lambda_sf),
        ffi.cast("int", maxiter),
        ffi.cast("double", reject_threshold),
        ffi.cast("int", 1 if use_preset else 0),
        ffi.cast("double *", psf_curve.ctypes.data),
        ffi.cast("double *", sp.ctypes.data),
        ffi.cast("double *", sl.ctypes.data),
        ffi.cast("double *", model.ctypes.data),
        ffi.cast("double *", unc.ctypes.data),
        ffi.cast("double *", info.ctypes.data),
    )

    if np.any(np.isnan(sp)):
        logger.error("NaNs in the spectrum")

    # The decomposition failed
    if info[0] == 0:
        status = info[2]
        if status == 0:
            msg = "I dont't know what happened"
            logger.error(msg)
        elif status == -1:
            # Don't warn about convergence when using preset slitfunc
            # since we expect only a single pass
            if not use_preset:
                msg = f"Did not finish convergence after maxiter ({maxiter}) iterations"
                logger.warning(msg)
        elif status == -2:
            msg = "Curvature is larger than the swath. Check the curvature!"
            logger.error(msg)
        else:
            msg = f"Check the C code, for status = {status}"
            logger.error(msg)
        # raise RuntimeError(msg)

    mask = _mask_from_c(mask)

    return sp, sl, model, unc, mask, info


# x, y, w
xi_ref = [("x", c_int), ("y", c_int), ("w", c_double)]
# x, iy, w
zeta_ref = [("x", c_int), ("iy", c_int), ("w", c_double)]


def xi_zeta_tensors(
    ncols: int,
    nrows: int,
    ycen: np.ndarray,
    yrange,  # (int, int)
    osample: int,
    p1: np.ndarray,
    p2: np.ndarray,
):
    ncols = int(ncols)
    nrows = int(nrows)
    osample = int(osample)
    ny = osample * (nrows + 1) + 1

    ycen_offset = ycen.astype(c_int)
    ycen_int = ycen - ycen_offset
    y_lower_lim = int(yrange[0])

    psf_curve = np.zeros((ncols, 3), dtype=c_double)
    psf_curve[:, 1] = p1
    psf_curve[:, 2] = p2

    requirements = ["C", "A", "W", "O"]
    ycen_int = np.require(ycen_int, dtype=c_double, requirements=requirements)
    ycen_offset = np.require(ycen_offset, dtype=c_int, requirements=requirements)

    xi = np.empty((ncols, ny, 4), dtype=xi_ref)
    zeta = np.empty((ncols, nrows, 3 * (osample + 1)), dtype=zeta_ref)
    m_zeta = np.empty((ncols, nrows), dtype=c_int)

    slitfunc_2dlib.xi_zeta_tensors(
        ffi.cast("int", ncols),
        ffi.cast("int", nrows),
        ffi.cast("int", ny),
        ffi.cast("double *", ycen_int.ctypes.data),
        ffi.cast("int *", ycen_offset.ctypes.data),
        ffi.cast("int", y_lower_lim),
        ffi.cast("int", osample),
        ffi.cast("double *", psf_curve.ctypes.data),
        ffi.cast("xi_ref *", xi.ctypes.data),
        ffi.cast("zeta_ref *", zeta.ctypes.data),
        ffi.cast("int *", m_zeta.ctypes.data),
    )

    return xi, zeta, m_zeta


def create_spectral_model(
    ncols: int,
    nrows: int,
    osample: int,
    xi: "xi_ref",
    spec: np.ndarray,
    slitfunc: np.ndarray,
):
    ncols = int(ncols)
    nrows = int(nrows)

    requirements = ["C", "A", "W", "O"]
    spec = np.require(spec, dtype=c_double, requirements=requirements)
    slitfunc = np.require(slitfunc, dtype=c_double, requirements=requirements)
    xi = np.require(xi, dtype=xi_ref, requirements=requirements)

    img = np.empty((nrows + 1, ncols), dtype=c_double)

    slitfunc_2dlib.create_spectral_model(
        ffi.cast("int", ncols),
        ffi.cast("int", nrows),
        ffi.cast("int", osample),
        ffi.cast("xi_ref *", xi.ctypes.data),
        ffi.cast("double *", spec.ctypes.data),
        ffi.cast("double *", slitfunc.ctypes.data),
        ffi.cast("double *", img.ctypes.data),
    )
    return img


def extract_with_slitfunc(
    img: np.ndarray,
    ycen: np.ndarray,
    slitfunc: np.ndarray,
    slitfunc_meta: dict,
    yrange: tuple[int, int],
    osample: int,
    p1: np.ndarray | float = 0,
    p2: np.ndarray | float = 0,
    lambda_sp: float = 0,
    gain: float = 1,
    maxiter: int = 1,
    reject_threshold: float = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract spectrum using a preset slit function (single-pass extraction).

    This function validates and adapts the slit function to match the current
    extraction parameters, then calls the C extraction code with the preset
    slit function (skipping the sL solve step).

    Parameters
    ----------
    img : array[nrows, ncols]
        Image to extract from
    ycen : array[ncols]
        Order center positions (fractional pixel)
    slitfunc : array[ny_src]
        Preset slit function from a previous extraction (e.g., from norm_flat)
    slitfunc_meta : dict
        Metadata about the slit function source, must contain:
        - "osample": oversampling factor used when slitfunc was computed
        - "extraction_height": extraction height used (or yrange tuple)
    yrange : tuple(int, int)
        Target extraction height: pixels (below, above) the trace center
    osample : int
        Target oversampling factor for slit function
    p1 : array[ncols] or float
        Linear curvature coefficient
    p2 : array[ncols] or float
        Quadratic curvature coefficient
    lambda_sp : float
        Spectrum smoothing parameter (0 = no smoothing)
    gain : float
        Detector gain for uncertainty estimation
    maxiter : int
        Maximum iterations (default 1 for single-pass)
    reject_threshold : float
        Outlier rejection threshold (default 0 = disabled)

    Returns
    -------
    sp, sl, model, unc, mask, info
        Same as slitfunc_curved()

    Raises
    ------
    ValueError
        If extraction height is larger than source (can't extrapolate)
    """
    nrows, _ = img.shape

    # Get source parameters
    src_osample = slitfunc_meta.get("osample", osample)
    src_height = slitfunc_meta.get("extraction_height")
    if src_height is None:
        src_yrange = slitfunc_meta.get("yrange", yrange)
    else:
        # Convert extraction_height to yrange if needed
        if np.isscalar(src_height):
            half = src_height / 2
            src_yrange = (int(np.floor(half)), int(np.ceil(half)))
        else:
            src_yrange = tuple(src_height)

    src_nrows = src_yrange[0] + src_yrange[1] + 1
    src_ny = src_osample * (src_nrows + 1) + 1

    # Validate source slitfunc size
    if slitfunc.size != src_ny:
        raise ValueError(
            f"Slit function size {slitfunc.size} doesn't match expected {src_ny} "
            f"for source parameters (osample={src_osample}, yrange={src_yrange})"
        )

    # Check if extraction height is compatible
    target_height = yrange[0] + yrange[1] + 1
    src_height_px = src_yrange[0] + src_yrange[1] + 1
    if target_height > src_height_px:
        raise ValueError(
            f"Target extraction height ({target_height} pixels) is larger than "
            f"source ({src_height_px} pixels). Cannot extrapolate slit function."
        )

    # Adapt slit function if parameters differ
    adapted_sl = _adapt_slitfunc(slitfunc, src_osample, src_yrange, osample, yrange)

    # Call the C extraction with preset slitfunc
    return slitfunc_curved(
        img,
        ycen,
        p1,
        p2,
        lambda_sp=lambda_sp,
        lambda_sf=0.1,  # not used when preset
        osample=osample,
        yrange=yrange,
        maxiter=maxiter,
        gain=gain,
        reject_threshold=reject_threshold,
        preset_slitfunc=adapted_sl,
    )


def _adapt_slitfunc(
    slitfunc: np.ndarray,
    src_osample: int,
    src_yrange: tuple[int, int],
    tgt_osample: int,
    tgt_yrange: tuple[int, int],
) -> np.ndarray:
    """Adapt slit function to different osample or extraction height.

    Parameters
    ----------
    slitfunc : array
        Source slit function
    src_osample : int
        Source oversampling factor
    src_yrange : tuple(int, int)
        Source extraction height (below, above)
    tgt_osample : int
        Target oversampling factor
    tgt_yrange : tuple(int, int)
        Target extraction height (below, above)

    Returns
    -------
    adapted : array
        Adapted slit function for target parameters
    """
    from scipy.interpolate import interp1d

    src_nrows = src_yrange[0] + src_yrange[1] + 1
    tgt_nrows = tgt_yrange[0] + tgt_yrange[1] + 1
    src_ny = src_osample * (src_nrows + 1) + 1
    tgt_ny = tgt_osample * (tgt_nrows + 1) + 1

    # If parameters match, return copy
    if src_osample == tgt_osample and src_yrange == tgt_yrange:
        return slitfunc.copy()

    # Create coordinate systems relative to trace center (y=0)
    # Source: spans from -src_yrange[0]-1 to src_yrange[1]+1
    src_y = np.linspace(-src_yrange[0] - 1, src_yrange[1] + 1, src_ny)
    # Target: spans from -tgt_yrange[0]-1 to tgt_yrange[1]+1
    tgt_y = np.linspace(-tgt_yrange[0] - 1, tgt_yrange[1] + 1, tgt_ny)

    # Check if resampling osample
    if src_osample != tgt_osample:
        logger.warning(
            "Resampling slit function from osample=%d to osample=%d",
            src_osample,
            tgt_osample,
        )

    # Check if truncating
    if tgt_nrows < src_nrows:
        logger.info(
            "Truncating slit function from %d to %d pixels", src_nrows, tgt_nrows
        )

    # Interpolate
    interp = interp1d(src_y, slitfunc, kind="cubic", bounds_error=False, fill_value=0.0)
    adapted = interp(tgt_y)

    # Renormalize to sum to osample
    total = adapted.sum()
    if total > 0:
        adapted *= tgt_osample / total
    else:
        logger.warning("Slit function sums to zero after adaptation")

    return adapted
