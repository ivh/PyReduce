"""
Wrapper for REDUCE C functions
"""
import ctypes
import io
import logging
import os
import sys
import tempfile
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

try:
    from .clib._slitfunc_bd import lib as slitfunclib
    from .clib._slitfunc_2d import lib as slitfunc_2dlib

    from .clib._slitfunc_bd import ffi
except ImportError:
    raise ImportError("Use build_extract.py to compile the C libraries")


c_double = np.ctypeslib.ctypes.c_double
c_int = np.ctypeslib.ctypes.c_int

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

@contextmanager
def stdout_redirector(stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)


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
    img = np.asanyarray(img)
    ycen = np.asanyarray(ycen)

    # Ensure the datatype and shape of all arrays before sending them to C code
    if not np.issubdtype(img.dtype, np.number):
        raise TypeError(
            "Input image must be a numeric type, but got %s" % str(img.dtype)
        )

    if not np.issubdtype(ycen.dtype, np.number):
        raise TypeError("Ycen must be a numeric type, but got %s" % str(ycen.dtype))

    if img.shape[1] != ycen.size:
        raise ValueError(
            "Image and Ycen shapes are incompatible, got %s and %s"
            % (img.shape, ycen.shape)
        )

    if osample <= 0:
        raise ValueError("Oversample rate must be positive, but got %i" % osample)
    if lambda_sf < 0:
        raise ValueError(
            "Slitfunction smoothing must be positive, but got %f" % lambda_sf
        )
    if lambda_sp < 0:
        raise ValueError("Spectrum smoothing must be positive, but got %f" % lambda_sp)

    # Get some derived values
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1
    ycen = ycen - ycen.astype(int)

    # Prepare all arrays
    # Inital guess for slit function and spectrum
    sp = np.ma.sum(img, axis=0)
    sp = np.require(sp, dtype=c_double, requirements=["C", "A", "W", "O"])

    sl = np.zeros(ny, dtype=c_double)

    mask = ~np.ma.getmaskarray(img)
    mask = np.require(mask, dtype=c_int, requirements=["C", "A", "W", "O"])

    img = np.ma.getdata(img)
    img = np.require(img, dtype=c_double, requirements=["C", "A", "W", "O"])

    pix_unc = np.zeros_like(img)
    pix_unc = np.require(pix_unc, dtype=c_double, requirements=["C", "A", "W", "O"])

    ycen = np.require(ycen, dtype=c_double, requirements=["C", "A", "W", "O"])
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
    mask = ~mask.astype(bool)

    return sp, sl, model, unc, mask


def slitfunc_curved(img, ycen, tilt, shear, lambda_sp=0, lambda_sf=0.1, osample=1):
    """Decompose an image into a spectrum and a slitfunction, image may be curved

    Parameters
    ----------
    img : array[n, m]
        input image
    ycen : array[n]
        traces the center of the order
    shear : array[n]
        tilt of the order along the image ???, set to 0 if order straight
    osample : int, optional
        Subpixel ovsersampling factor (the default is 1, which no oversampling)
    lambda_sp : float, optional
        smoothing factor spectrum (the default is 0, which no smoothing)
    lambda_sl : float, optional
        smoothing factor slitfunction (the default is 0.1, which small)

    Returns
    -------
    sp, sl, model, unc
        spectrum, slitfunction, model, spectrum uncertainties
    """

    # Convert datatypes to expected values
    lambda_sf = float(lambda_sf)
    lambda_sp = float(lambda_sp)
    osample = int(osample)
    img = np.asanyarray(img)
    ycen = np.asanyarray(ycen)

    if np.isscalar(tilt) and np.issubdtype(np.asanyarray(tilt).dtype, np.number):
        tilt = np.full(img.shape[1], tilt, dtype=c_double)
    else:
        tilt = np.asanyarray(tilt)
    if np.isscalar(shear) and np.issubdtype(np.asanyarray(shear).dtype, np.number):
        shear = np.full(img.shape[1], shear, dtype=c_double)
    else:
        shear = np.asanyarray(shear)

    # Check type and dimensions of arrays before sending them to the C code
    if not np.issubdtype(img.dtype, np.number):
        raise TypeError(
            "Input image must be a numeric type, but got %s" % str(img.dtype)
        )
    if not np.issubdtype(ycen.dtype, np.number):
        raise TypeError("Ycen must be a numeric type, but got %s" % str(ycen.dtype))
    if not np.issubdtype(tilt.dtype, np.number):
        raise TypeError("Tilt must be a numeric type, but got %s" % str(tilt.dtype))
    if not np.issubdtype(shear.dtype, np.number):
        raise TypeError("Shear must be a numeric type, but got %s" % str(shear.dtype))

    if img.shape[1] != ycen.size:
        raise ValueError(
            "Image and Ycen shapes are incompatible, got %s and %s"
            % (img.shape, ycen.shape)
        )
    if img.shape[1] != tilt.size:
        raise ValueError(
            "Image and Tilt shapes are incompatible, got %s and %s"
            % (img.shape, tilt.shape)
        )
    if img.shape[1] != shear.size:
        raise ValueError(
            "Image and Shear shapes are incompatible, got %s and %s"
            % (img.shape, shear.shape)
        )

    if osample <= 0:
        raise ValueError("Oversample rate must be positive, but got %i" % osample)
    if lambda_sf < 0:
        raise ValueError(
            "Slitfunction smoothing must be positive, but got %f" % lambda_sf
        )
    if lambda_sp < 0:
        raise ValueError("Spectrum smoothing must be positive, but got %f" % lambda_sp)

    # Retrieve some derived values
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    ycen_offset = ycen.astype(c_int)
    ycen = ycen - ycen_offset

    y_lower_lim = nrows // 2

    sl = np.zeros(ny, dtype=c_double)

    # Inital guess for spectrum
    sp = np.ma.sum(img, axis=0)
    sp = np.ma.filled(sp, 0)
    sp = np.require(sp, dtype=c_double, requirements=["C", "A", "W", "O"])

    # Initialize arrays and ensure the correct datatype for C
    mask = ~np.ma.getmaskarray(img)
    mask = np.require(mask, dtype=c_int, requirements=["C", "A", "W", "O"])

    img = np.ma.getdata(img)
    img = np.require(img, dtype=c_double, requirements=["C", "A", "W", "O"])

    pix_unc = np.zeros_like(img)
    pix_unc = np.require(pix_unc, dtype=c_double, requirements=["C", "A", "W", "O"])

    ycen = np.require(ycen, dtype=c_double, requirements=["C", "A", "W", "O"])

    ycen_offset = np.require(
        ycen_offset, dtype=c_int, requirements=["C", "A", "W", "O"]
    )

    tilt = np.ma.filled(tilt, 0)
    tilt = np.require(tilt, dtype=c_double, requirements=["C", "A", "W", "O"])
    # TODO DEBUG dont do this
    shear = np.zeros(ncols)
    # shear = np.ma.filled(shear, 0)
    shear = np.require(shear, dtype=c_double, requirements=["C", "A", "W", "O"])

    model = np.zeros((nrows, ncols), dtype=c_double)
    unc = np.zeros(ncols, dtype=c_double)

    # Call the C function
    # f = io.BytesIO()
    # with stdout_redirector(f):
    slitfunc_2dlib.slit_func_curved(
        ffi.cast("int", ncols),
        ffi.cast("int", nrows),
        ffi.cast("double *", img.ctypes.data),
        ffi.cast("double *", pix_unc.ctypes.data),
        ffi.cast("int *", mask.ctypes.data),
        ffi.cast("double *", ycen.ctypes.data),
        ffi.cast("int *", ycen_offset.ctypes.data),
        ffi.cast("double *", tilt.ctypes.data),
        ffi.cast("double *", shear.ctypes.data),
        ffi.cast("int", y_lower_lim),
        ffi.cast("int", osample),
        ffi.cast("double", lambda_sp),
        ffi.cast("double", lambda_sf),
        ffi.cast("double *", sp.ctypes.data),
        ffi.cast("double *", sl.ctypes.data),
        ffi.cast("double *", model.ctypes.data),
        ffi.cast("double *", unc.ctypes.data),
    )
    # output = f.getvalue().decode("utf-8")
    # if output != "":
    #     img.tofile("debug_img.dat")
    #     pix_unc.tofile("debug_pix_unc.dat")
    #     mask.tofile("debug_mask.dat")
    #     ycen.tofile("debug_ycen.dat")
    #     ycen_offset.tofile("debug_ycen_offset.dat")
    #     tilt.tofile("debug_tilt.dat")
    #     shear.tofile("debug_shear.dat")
    #     print(output)

    mask = ~mask.astype(bool)

    return sp, sl, model, unc, mask
