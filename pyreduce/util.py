# -*- coding: utf-8 -*-
"""
Collection of various useful and/or reoccuring functions across PyReduce
"""

import logging
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import scipy.interpolate
from astropy import coordinates as coord
from astropy import time
from astropy import units as u
from astropy.io import fits
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import lstsq, solve, solve_banded
from scipy.ndimage.filters import median_filter
from scipy.optimize import curve_fit, least_squares
from scipy.special import binom

from . import __version__
from .clipnflip import clipnflip
from .instruments.instrument_info import modeinfo

logger = logging.getLogger(__name__)


def resample(array, new_size):
    x = np.arange(new_size)
    xp = np.linspace(0, new_size, len(array))
    return np.interp(x, xp, array)


def remove_bias(img, ihead, bias, bhead, nfiles=1):
    if bias is not None and bhead is not None:
        b_exptime = bhead["EXPTIME"]
        i_exptime = ihead["EXPTIME"]
        if b_exptime == 0 or i_exptime == 0:
            b_exptime = 1
            i_exptime = nfiles
        img = img - bias * i_exptime / b_exptime
    return img


def in_ipynb():
    try:
        cfg = get_ipython().config
        if cfg["IPKernelApp"]["parent_appname"] == "ipython-notebook":
            return True
        else:
            return False
    except NameError:
        return False


def log_version():
    """For Debug purposes"""
    logger.debug("----------------------")
    logger.debug("PyReduce version: %s", __version__)


def start_logging(log_file="log.log"):
    """Start logging to log file and command line

    Parameters
    ----------
    log_file : str, optional
        name of the logging file (default: "log.log")
    """

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)-15s - %(levelname)s - %(name)-8s - %(message)s",
    )
    logging.captureWarnings(True)
    log_version()


def vac2air(wl_vac):
    """
    Convert vacuum wavelengths to wavelengths in air
    Author: Nikolai Piskunov
    """
    wl_air = wl_vac
    ii = np.where(wl_vac > 2e3)

    sigma2 = (1e4 / wl_vac[ii]) ** 2  # Compute wavenumbers squared
    fact = (
        1e0
        + 8.34254e-5
        + 2.406147e-2 / (130e0 - sigma2)
        + 1.5998e-4 / (38.9e0 - sigma2)
    )
    wl_air[ii] = wl_vac[ii] / fact  # Convert to air wavelength
    return wl_air


def air2vac(wl_air):
    """
    Convert wavelengths in air to vacuum wavelength
    Author: Nikolai Piskunov
    """
    wl_vac = np.copy(wl_air)
    ii = np.where(wl_air > 1999.352)

    sigma2 = (1e4 / wl_air[ii]) ** 2  # Compute wavenumbers squared
    fact = (
        1e0
        + 8.336624212083e-5
        + 2.408926869968e-2 / (1.301065924522e2 - sigma2)
        + 1.599740894897e-4 / (3.892568793293e1 - sigma2)
    )
    wl_vac[ii] = wl_air[ii] * fact  # Convert to vacuum wavelength
    return wl_vac


def swap_extension(fname, ext, path=None):
    """exchange the extension of the given file with a new one"""
    if path is None:
        path = os.path.dirname(fname)
    nameout = os.path.basename(fname)
    if nameout[-3:] == ".gz":
        nameout = nameout[:-3]
    nameout = nameout.rsplit(".", 1)[0]
    nameout = os.path.join(path, nameout + ext)
    return nameout


def find_first_index(arr, value):
    """find the first element equal to value in the array arr"""
    try:
        return next(i for i, v in enumerate(arr) if v == value)
    except StopIteration:
        raise Exception("Value %s not found" % value)


def interpolate_masked(masked):
    """Interpolate masked values, from non masked values

    Parameters
    ----------
    masked : masked_array
        masked array to interpolate on

    Returns
    -------
    interpolated : array
        interpolated non masked array
    """

    mask = np.ma.getmaskarray(masked)
    idx = np.nonzero(~mask)[0]
    interpol = np.interp(np.arange(len(masked)), idx, masked[idx])
    return interpol


def cutout_image(img, ymin, ymax, xmin, xmax):
    """Cut a section of an image out

    Parameters
    ----------
    img : array
        image
    ymin : array[ncol](int)
        lower y value
    ymax : array[ncol](int)
        upper y value
    xmin : int
        lower x value
    xmax : int
        upper x value

    Returns
    -------
    cutout : array[height, ncol]
        selection of the image
    """

    cutout = np.zeros((ymax[0] - ymin[0] + 1, xmax - xmin), dtype=img.dtype)
    for i, x in enumerate(range(xmin, xmax)):
        cutout[:, i] = img[ymin[x] : ymax[x] + 1, x]
    return cutout


def make_index(ymin, ymax, xmin, xmax, zero=0):
    """Create an index (numpy style) that will select part of an image with changing position but fixed height

    The user is responsible for making sure the height is constant, otherwise it will still work, but the subsection will not have the desired format

    Parameters
    ----------
    ymin : array[ncol](int)
        lower y border
    ymax : array[ncol](int)
        upper y border
    xmin : int
        leftmost column
    xmax : int
        rightmost colum
    zero : bool, optional
        if True count y array from 0 instead of xmin (default: False)

    Returns
    -------
    index : tuple(array[height, width], array[height, width])
        numpy index for the selection of a subsection of an image
    """

    # TODO
    # Define the indices for the pixels between two y arrays, e.g. pixels in an order
    # in x: the rows between ymin and ymax
    # in y: the column, but n times to match the x index
    ymin = np.asarray(ymin, dtype=int)
    ymax = np.asarray(ymax, dtype=int)
    xmin = int(xmin)
    xmax = int(xmax)

    if zero:
        zero = xmin

    index_x = np.array(
        [np.arange(ymin[col], ymax[col] + 1) for col in range(xmin - zero, xmax - zero)]
    )
    index_y = np.array(
        [
            np.full(ymax[col] - ymin[col] + 1, col)
            for col in range(xmin - zero, xmax - zero)
        ]
    )
    index = index_x.T, index_y.T + zero

    return index


def gridsearch(func, grid, args=(), kwargs={}):
    matrix = np.zeros(grid.shape[:-1])

    for idx in np.ndindex(grid.shape[:-1]):
        value = grid[idx]
        print(f"Value: {value}")
        try:
            result = func(value, *args, **kwargs)
            print(f"Success: {result}")
        except Exception as e:
            result = np.nan
            print(f"Failed: {e}")
        finally:
            matrix[idx] = result

    return matrix


def gaussfit(x, y):
    """
    Fit a simple gaussian to data

    gauss(x, a, mu, sigma) = a * exp(-z**2/2)
    with z = (x - mu) / sigma

    Parameters
    ----------
    x : array(float)
        x values
    y : array(float)
        y values
    Returns
    -------
    gauss(x), parameters
        fitted values for x, fit paramters (a, mu, sigma)
    """

    gauss = lambda x, A0, A1, A2: A0 * np.exp(-(((x - A1) / A2) ** 2) / 2)
    popt, _ = curve_fit(gauss, x, y, p0=[max(y), 0, 1])
    return gauss(x, *popt), popt


def gaussfit2(x, y):
    """Fit a gaussian(normal) curve to data x, y

    gauss = A * exp(-(x-mu)**2/(2*sig**2)) + offset

    Parameters
    ----------
    x : array[n]
        x values
    y : array[n]
        y values

    Returns
    -------
    popt : array[4]
        coefficients of the gaussian: A, mu, sigma**2, offset
    """

    gauss = gaussval2

    x = np.ma.compressed(x)
    y = np.ma.compressed(y)

    if len(x) == 0 or len(y) == 0:
        raise ValueError("All values masked")

    if len(x) != len(y):
        raise ValueError("The masks of x and y are different")

    # Find the peak in the center of the image
    weights = np.ones(len(y), dtype=y.dtype)
    midpoint = len(y) // 2
    weights[:midpoint] = np.linspace(0, 1, midpoint, dtype=weights.dtype)
    weights[midpoint:] = np.linspace(1, 0, len(y) - midpoint, dtype=weights.dtype)

    i = np.argmax(y * weights)
    p0 = [y[i], x[i], 1]
    with np.warnings.catch_warnings():
        np.warnings.simplefilter("ignore")
        res = least_squares(
            lambda c: gauss(x, *c, np.ma.min(y)) - y,
            p0,
            loss="soft_l1",
            bounds=(
                [min(np.ma.mean(y), y[i]), np.ma.min(x), 0],
                [np.ma.max(y) * 1.5, np.ma.max(x), len(x) / 2],
            ),
        )
        popt = list(res.x) + [np.min(y)]
    return popt


def gaussfit3(x, y):
    """A very simple (and relatively fast) gaussian fit
    gauss = A * exp(-(x-mu)**2/(2*sig**2)) + offset

    Parameters
    ----------
    x : array of shape (n,)
        x data
    y : array of shape (n,)
        y data

    Returns
    -------
    popt : list of shape (4,)
        Parameters A, mu, sigma**2, offset
    """
    mask = np.ma.getmaskarray(x) | np.ma.getmaskarray(y)
    x, y = x[~mask], y[~mask]

    gauss = gaussval2
    i = np.argmax(y[len(y) // 4 : len(y) * 3 // 4]) + len(y) // 4
    p0 = [y[i], x[i], 1, np.min(y)]

    with np.warnings.catch_warnings():
        np.warnings.simplefilter("ignore")
        popt, _ = curve_fit(gauss, x, y, p0=p0)

    return popt


def gaussfit4(x, y):
    """A very simple (and relatively fast) gaussian fit
    gauss = A * exp(-(x-mu)**2/(2*sig**2)) + offset

    Assumes x is sorted

    Parameters
    ----------
    x : array of shape (n,)
        x data
    y : array of shape (n,)
        y data

    Returns
    -------
    popt : list of shape (4,)
        Parameters A, mu, sigma**2, offset
    """
    gauss = gaussval2
    x = np.ma.compressed(x)
    y = np.ma.compressed(y)
    i = np.argmax(y)
    p0 = [y[i], x[i], 1, np.min(y)]

    with np.warnings.catch_warnings():
        np.warnings.simplefilter("ignore")
        popt, _ = curve_fit(gauss, x, y, p0=p0)

    return popt


def gaussfit_linear(x, y):
    """Transform the gaussian fit into a linear least squares problem, and solve that instead of the non-linear curve fit
    For efficiency reasons. (roughly 10 times faster than the curve fit)

    Parameters
    ----------
    x : array of shape (n,)
        x data
    y : array of shape (n,)
        y data

    Returns
    -------
    coef : tuple
        a, mu, sig, 0
    """
    x = x[y > 0]
    y = y[y > 0]

    offset = np.min(y)
    y = y - offset + 1e-12

    weights = y

    d = np.log(y)
    G = np.ones((x.size, 3), dtype=np.float)
    G[:, 0] = x ** 2
    G[:, 1] = x

    beta, _, _, _ = np.linalg.lstsq(
        (G.T * weights ** 2).T, d * weights ** 2, rcond=None
    )

    a = np.exp(beta[2] - beta[1] ** 2 / (4 * beta[0]))
    sig = -1 / (2 * beta[0])
    mu = -beta[1] / (2 * beta[0])

    return a, mu, sig, offset


def gaussval2(x, a, mu, sig, const):
    return a * np.exp(-((x - mu) ** 2) / (2 * sig)) + const


def gaussbroad(x, y, hwhm):
    """
    Apply gaussian broadening to x, y data with half width half maximum hwhm

    Parameters
    ----------
    x : array(float)
        x values
    y : array(float)
        y values
    hwhm : float > 0
        half width half maximum
    Returns
    -------
    array(float)
        broadened y values
    """

    # alternatively use:
    # from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
    # but that doesn't have an x coordinate

    nw = len(x)
    dw = (x[-1] - x[0]) / (len(x) - 1)

    if hwhm > 5 * (x[-1] - x[0]):
        return np.full(len(x), sum(y) / len(x))

    nhalf = int(3.3972872 * hwhm / dw)
    ng = 2 * nhalf + 1  # points in gaussian (odd!)
    # wavelength scale of gaussian
    wg = dw * (np.arange(0, ng, 1, dtype=float) - (ng - 1) / 2)
    xg = (0.83255461 / hwhm) * wg  # convenient absisca
    gpro = (0.46974832 * dw / hwhm) * np.exp(-xg * xg)  # unit area gaussian w/ FWHM
    gpro = gpro / np.sum(gpro)

    # Pad spectrum ends to minimize impact of Fourier ringing.
    npad = nhalf + 2  # pad pixels on each end
    spad = np.concatenate((np.full(npad, y[0]), y, np.full(npad, y[-1])))

    # Convolve and trim.
    sout = np.convolve(spad, gpro)  # convolve with gaussian
    sout = sout[npad : npad + nw]  # trim to original data / length
    return sout  # return broadened spectrum.


def polyfit1d(x, y, degree=1, regularization=0):
    idx = np.arange(degree + 1)
    coeff = np.zeros(degree + 1)

    A = np.array([np.power(x, i) for i in idx], dtype=float).T
    b = y.ravel()

    L = np.array([regularization * i ** 2 for i in idx])
    I = np.linalg.inv(A.T @ A + np.diag(L))
    coeff = I @ A.T @ b

    coeff = coeff[::-1]

    return coeff


def _get_coeff_idx(coeff):
    idx = np.indices(coeff.shape)
    idx = idx.T.swapaxes(0, 1).reshape((-1, 2))
    # degree = coeff.shape
    # idx = [[i, j] for i, j in product(range(degree[0]), range(degree[1]))]
    # idx = np.asarray(idx)
    return idx


def _scale(x, y):
    # Normalize x and y to avoid huge numbers
    # Mean 0, Variation 1
    offset_x, offset_y = np.mean(x), np.mean(y)
    norm_x, norm_y = np.std(x), np.std(y)
    if norm_x == 0:
        norm_x = 1
    if norm_y == 0:
        norm_y = 1
    x = (x - offset_x) / norm_x
    y = (y - offset_y) / norm_y
    return x, y, (norm_x, norm_y), (offset_x, offset_y)


def _unscale(x, y, norm, offset):
    x = x * norm[0] + offset[0]
    y = y * norm[1] + offset[1]
    return x, y


def polyvander2d(x, y, degree):
    # A = np.array([x ** i * y ** j for i, j in idx], dtype=float).T
    A = np.polynomial.polynomial.polyvander2d(x, y, degree)
    return A


def polyscale2d(coeff, scale_x, scale_y, copy=True):
    if copy:
        coeff = np.copy(coeff)
    idx = _get_coeff_idx(coeff)
    for k, (i, j) in enumerate(idx):
        coeff[i, j] /= scale_x ** i * scale_y ** j
    return coeff


def polyshift2d(coeff, offset_x, offset_y, copy=True):
    if copy:
        coeff = np.copy(coeff)
    idx = _get_coeff_idx(coeff)
    # Copy coeff because it changes during the loop
    coeff2 = np.copy(coeff)
    for k, m in idx:
        not_the_same = ~((idx[:, 0] == k) & (idx[:, 1] == m))
        above = (idx[:, 0] >= k) & (idx[:, 1] >= m) & not_the_same
        for i, j in idx[above]:
            b = binom(i, k) * binom(j, m)
            sign = (-1) ** ((i - k) + (j - m))
            offset = offset_x ** (i - k) * offset_y ** (j - m)
            coeff[k, m] += sign * b * coeff2[i, j] * offset
    return coeff


def plot2d(x, y, z, coeff, title=None):
    # regular grid covering the domain of the data
    if x.size > 500:
        choice = np.random.choice(x.size, size=500, replace=False)
    else:
        choice = slice(None, None, None)
    x, y, z = x[choice], y[choice], z[choice]
    X, Y = np.meshgrid(
        np.linspace(np.min(x), np.max(x), 20), np.linspace(np.min(y), np.max(y), 20)
    )
    Z = np.polynomial.polynomial.polyval2d(X, Y, coeff)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(x, y, z, c="r", s=50)
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Z")
    if title is not None:
        plt.title(title)
    # ax.axis("equal")
    # ax.axis("tight")
    plt.show()


def polyfit2d(
    x, y, z, degree=1, max_degree=None, scale=True, plot=False, plot_title=None
):
    """A simple 2D plynomial fit to data x, y, z
    The polynomial can be evaluated with numpy.polynomial.polynomial.polyval2d

    Parameters
    ----------
    x : array[n]
        x coordinates
    y : array[n]
        y coordinates
    z : array[n]
        data values
    degree : int, optional
        degree of the polynomial fit (default: 1)
    max_degree : {int, None}, optional
        if given the maximum combined degree of the coefficients is limited to this value
    scale : bool, optional
        Wether to scale the input arrays x and y to mean 0 and variance 1, to avoid numerical overflows.
        Especially useful at higher degrees. (default: True)
    plot : bool, optional
        wether to plot the fitted surface and data (slow) (default: False)

    Returns
    -------
    coeff : array[degree+1, degree+1]
        the polynomial coefficients in numpy 2d format, i.e. coeff[i, j] for x**i * y**j
    """
    # Flatten input
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()

    # Removed masked values
    mask = ~(np.ma.getmask(z) | np.ma.getmask(x) | np.ma.getmask(y))
    x, y, z = x[mask].ravel(), y[mask].ravel(), z[mask].ravel()

    if scale:
        x, y, norm, offset = _scale(x, y)

    # Create combinations of degree of x and y
    # usually: [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), ....]
    if np.isscalar(degree):
        degree = (int(degree), int(degree))
    assert len(degree) == 2, "Only 2D polynomials can be fitted"
    degree = [int(degree[0]), int(degree[1])]
    # idx = [[i, j] for i, j in product(range(degree[0] + 1), range(degree[1] + 1))]
    coeff = np.zeros((degree[0] + 1, degree[1] + 1))
    idx = _get_coeff_idx(coeff)

    # Calculate elements 1, x, y, x*y, x**2, y**2, ...
    A = polyvander2d(x, y, degree)

    # We only want the combinations with maximum order COMBINED power
    if max_degree is not None:
        mask = idx[:, 0] + idx[:, 1] <= int(max_degree)
        idx = idx[mask]
        A = A[:, mask]

    # Do least squares fit
    C, *_ = lstsq(A, z)

    # Reorder coefficients into numpy compatible 2d array
    for k, (i, j) in enumerate(idx):
        coeff[i, j] = C[k]

    # # Backup copy of coeff
    if scale:
        coeff = polyscale2d(coeff, *norm, copy=False)
        coeff = polyshift2d(coeff, *offset, copy=False)

    if plot:  # pragma: no cover
        if scale:
            x, y = _unscale(x, y, norm, offset)
        plot2d(x, y, z, coeff, title=plot_title)

    return coeff


def polyfit2d_2(x, y, z, degree=1, x0=None, loss="arctan", method="trf", plot=False):

    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    if np.isscalar(degree):
        degree_x = degree_y = degree + 1
    else:
        degree_x = degree[0] + 1
        degree_y = degree[1] + 1

    polyval2d = np.polynomial.polynomial.polyval2d

    def func(c):
        c = c.reshape(degree_x, degree_y)
        value = polyval2d(x, y, c)
        return value - z

    if x0 is None:
        x0 = np.zeros(degree_x * degree_y)
    else:
        x0 = x0.ravel()

    res = least_squares(func, x0, loss=loss, method=method)
    coef = res.x
    coef = coef.reshape(degree_x, degree_y)

    if plot:  # pragma: no cover
        # regular grid covering the domain of the data
        if x.size > 500:
            choice = np.random.choice(x.size, size=500, replace=False)
        else:
            choice = slice(None, None, None)
        x, y, z = x[choice], y[choice], z[choice]
        X, Y = np.meshgrid(
            np.linspace(np.min(x), np.max(x), 20), np.linspace(np.min(y), np.max(y), 20)
        )
        Z = np.polynomial.polynomial.polyval2d(X, Y, coef)
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        ax.scatter(x, y, z, c="r", s=50)
        plt.xlabel("X")
        plt.ylabel("Y")
        ax.set_zlabel("Z")
        ax.axis("equal")
        ax.axis("tight")
        plt.show()
    return coef


def bezier_interp(x_old, y_old, x_new):
    """
    Bezier interpolation, based on the scipy methods

    This mostly sanitizes the input by removing masked values and duplicate entries
    Note that in case of duplicate entries (in x_old) the results are not well defined as only one of the entries is used and the other is discarded

    Parameters
    ----------
    x_old : array[n]
        old x values
    y_old : array[n]
        old y values
    x_new : array[m]
        new x values

    Returns
    -------
    y_new : array[m]
        new y values
    """

    # Handle masked arrays
    if np.ma.is_masked(x_old):
        x_old = np.ma.compressed(x_old)
        y_old = np.ma.compressed(y_old)

    # avoid duplicate entries in x
    assert x_old.size == y_old.size
    x_old, index = np.unique(x_old, return_index=True)
    y_old = y_old[index]

    knots, coef, order = scipy.interpolate.splrep(x_old, y_old, s=0)
    y_new = scipy.interpolate.BSpline(knots, coef, order)(x_new)
    return y_new


def safe_interpolation(x_old, y_old, x_new=None, fill_value=0):
    """
    'Safe' interpolation method that should avoid
    the common pitfalls of spline interpolation

    masked arrays are compressed, i.e. only non masked entries are used
    remove NaN input in x_old and y_old
    only unique x values are used, corresponding y values are 'random'
    if all else fails, revert to linear interpolation

    Parameters
    ----------
    x_old : array of size (n,)
        x values of the data
    y_old : array of size (n,)
        y values of the data
    x_new : array of size (m, ) or None, optional
        x values of the interpolated values
        if None will return the interpolator object
        (default: None)

    Returns
    -------
    y_new: array of size (m, ) or interpolator
        if x_new was given, return the interpolated values
        otherwise return the interpolator object
    """

    # Handle masked arrays
    if np.ma.is_masked(x_old):
        x_old = np.ma.compressed(x_old)
        y_old = np.ma.compressed(y_old)

    mask = np.isfinite(x_old) & np.isfinite(y_old)
    x_old = x_old[mask]
    y_old = y_old[mask]

    # avoid duplicate entries in x
    # also sorts data, which allows us to use assume_sorted below
    x_old, index = np.unique(x_old, return_index=True)
    y_old = y_old[index]

    try:
        interpolator = scipy.interpolate.interp1d(
            x_old,
            y_old,
            kind="cubic",
            fill_value=fill_value,
            bounds_error=False,
            assume_sorted=True,
        )
    except ValueError:
        logging.warning(
            "Could not instantiate cubic spline interpolation, using linear instead"
        )
        interpolator = scipy.interpolate.interp1d(
            x_old,
            y_old,
            kind="linear",
            fill_value=fill_value,
            bounds_error=False,
            assume_sorted=True,
        )

    if x_new is not None:
        return interpolator(x_new)
    else:
        return interpolator


def bottom(f, order=1, iterations=40, eps=0.001, poly=False, weight=1, **kwargs):
    """
    bottom tries to fit a smooth curve to the lower envelope
    of 1D data array f. Filter size "filter"
    together with the total number of iterations determine
    the smoothness and the quality of the fit. The total
    number of iterations can be controlled by limiting the
    maximum number of iterations (iter) and/or by setting
    the convergence criterion for the fit (eps)
    04-Nov-2000 N.Piskunov wrote.
    09-Nov-2011 NP added weights and 2nd derivative constraint as LAM2

    Parameters
    ----------
    f : Callable
        Function to fit
    filter : int
        Smoothing parameter of the optimal filter (or polynomial degree of poly is True)
    iter : int
        maximum number of iterations [def: 40]
    eps : float
        convergence level [def: 0.001]
    mn : float
        minimum function values to be considered [def: min(f)]
    mx : float
        maximum function values to be considered [def: max(f)]
    lam2 : float
        constraint on 2nd derivative
    weight : array(float)
        vector of weights.
    """

    mn = kwargs.get("min", np.min(f))
    mx = kwargs.get("max", np.max(f))
    lambda2 = kwargs.get("lambda2", -1)

    if poly:
        j = np.where((f >= mn) & (f <= mx))
        xx = np.linspace(-1, 1, num=len(f))
        fmin = np.min(f[j]) - 1
        fmax = np.max(f[j]) + 1
        ff = (f[j] - fmin) / (fmax - fmin)
        ff_old = np.copy(ff)
    else:
        fff = middle(
            f, order, iterations=iterations, eps=eps, weight=weight, lambda2=lambda2
        )
        fmin = min(f) - 1
        fmax = max(f) + 1
        fff = (fff - fmin) / (fmax - fmin)
        ff = (f - fmin) / (fmax - fmin) / fff
        ff_old = np.copy(ff)

    for _ in range(iterations):
        if poly:

            if order > 0:  # this is a bug in rsi poly routine
                t = median_filter(np.polyval(np.polyfit(xx, ff, order), xx), 3)
                t = np.clip(t - ff, 0, None) ** 2
                tmp = np.polyval(np.polyfit(xx, t, order), xx)
                dev = np.sqrt(np.nan_to_num(tmp))
            else:
                t = np.tile(np.polyfit(xx, ff, order), len(f))
                t = np.polyfit(xx, np.clip(t - ff, 0, None) ** 2, order)
                t = np.tile(t, len(f))
                dev = np.nan_to_num(t)
                dev = np.sqrt(t)
        else:
            t = median_filter(opt_filter(ff, order, weight=weight, lambda2=lambda2), 3)
            dev = np.sqrt(
                opt_filter(
                    np.clip(weight * (t - ff), 0, None),
                    order,
                    weight=weight,
                    lambda2=lambda2,
                )
            )
        ff = np.clip(
            np.clip(t - dev, ff, None), None, t
        )  # the order matters, t dominates
        dev2 = np.max(weight * np.abs(ff - ff_old))
        ff_old = ff
        if dev2 <= eps:
            break

    if poly:
        if order > 0:  # this is a bug in rsi poly routine
            t = median_filter(np.polyval(np.polyfit(xx, ff, order), xx), 3)
        else:
            t = np.tile(np.polyfit(xx, ff, order), len(f))
        return t * (fmax - fmin) + fmin
    else:
        return t * fff * (fmax - fmin) + fmin


def middle(
    f,
    param,
    x=None,
    iterations=40,
    eps=0.001,
    poly=False,
    weight=1,
    lambda2=-1,
    mn=None,
    mx=None,
):
    """
    middle tries to fit a smooth curve that is located
    along the "middle" of 1D data array f. Filter size "filter"
    together with the total number of iterations determine
    the smoothness and the quality of the fit. The total
    number of iterations can be controlled by limiting the
    maximum number of iterations (iter) and/or by setting
    the convergence criterion for the fit (eps)
    04-Nov-2000 N.Piskunov wrote.
    09-Nov-2011 NP added weights and 2nd derivative constraint as LAM2

    Parameters
    ----------
    f : Callable
        Function to fit
    filter : int
        Smoothing parameter of the optimal filter (or polynomial degree of poly is True)
    iter : int
        maximum number of iterations [def: 40]
    eps : float
        convergence level [def: 0.001]
    mn : float
        minimum function values to be considered [def: min(f)]
    mx : float
        maximum function values to be considered [def: max(f)]
    lam2 : float
        constraint on 2nd derivative
    weight : array(float)
        vector of weights.
    """
    mn = mn if mn is not None else np.min(f)
    mx = mx if mx is not None else np.max(f)

    f = np.asarray(f)

    if x is None:
        xx = np.linspace(-1, 1, num=f.size)
    else:
        xx = np.asarray(x)

    if poly:
        j = (f >= mn) & (f <= mx)
        n = np.count_nonzero(j)
        if n <= round(param):
            return f

        fmin = np.min(f[j]) - 1
        fmax = np.max(f[j]) + 1
        ff = (f[j] - fmin) / (fmax - fmin)
        ff_old = ff
    else:
        fmin = np.min(f) - 1
        fmax = np.max(f) + 1
        ff = (f - fmin) / (fmax - fmin)
        ff_old = ff
        n = len(f)

    for _ in range(iterations):
        if poly:
            param = round(param)
            if param > 0:
                t = median_filter(np.polyval(np.polyfit(xx, ff, param), xx), 3)
                tmp = np.polyval(np.polyfit(xx, (t - ff) ** 2, param), xx)
            else:
                t = np.tile(np.polyfit(xx, ff, param), len(f))
                tmp = np.tile(np.polyfit(xx, (t - ff) ** 2, param), len(f))
        else:
            t = median_filter(opt_filter(ff, param, weight=weight, lambda2=lambda2), 3)
            tmp = opt_filter(
                weight * (t - ff) ** 2, param, weight=weight, lambda2=lambda2
            )

        dev = np.sqrt(np.clip(tmp, 0, None))
        ff = np.clip(t - dev, ff, t + dev)
        dev2 = np.max(weight * np.abs(ff - ff_old))
        ff_old = ff

        # print(dev2)
        if dev2 <= eps:
            break

    if poly:
        xx = np.linspace(-1, 1, len(f))
        if param > 0:
            t = median_filter(np.polyval(np.polyfit(xx, ff, param), xx), 3)
        else:
            t = np.tile(np.polyfit(xx, ff, param), len(f))

    return t * (fmax - fmin) + fmin


def top(
    f,
    order=1,
    iterations=40,
    eps=0.001,
    poly=False,
    weight=1,
    lambda2=-1,
    mn=None,
    mx=None,
):
    """
    top tries to fit a smooth curve to the upper envelope
    of 1D data array f. Filter size "filter"
    together with the total number of iterations determine
    the smoothness and the quality of the fit. The total
    number of iterations can be controlled by limiting the
    maximum number of iterations (iter) and/or by setting
    the convergence criterion for the fit (eps)
    04-Nov-2000 N.Piskunov wrote.
    09-Nov-2011 NP added weights and 2nd derivative constraint as LAM2

    Parameters
    ----------
    f : Callable
        Function to fit
    filter : int
        Smoothing parameter of the optimal filter (or polynomial degree of poly is True)
    iter : int
        maximum number of iterations [def: 40]
    eps : float
        convergence level [def: 0.001]
    mn : float
        minimum function values to be considered [def: min(f)]
    mx : float
        maximum function values to be considered [def: max(f)]
    lam2 : float
        constraint on 2nd derivative
    weight : array(float)
        vector of weights.
    """
    mn = mn if mn is not None else np.min(f)
    mx = mx if mx is not None else np.max(f)

    f = np.asarray(f)
    xx = np.linspace(-1, 1, num=f.size)

    if poly:
        j = (f >= mn) & (f <= mx)
        if np.count_nonzero(j) <= round(order):
            raise ValueError("Not enough points")
        fmin = np.min(f[j]) - 1
        fmax = np.max(f[j]) + 1
        ff = (f - fmin) / (fmax - fmin)
        ff_old = ff
    else:
        fff = middle(
            f, order, iterations=iterations, eps=eps, weight=weight, lambda2=lambda2
        )
        fmin = np.min(f) - 1
        fmax = np.max(f) + 1
        fff = (fff - fmin) / (fmax - fmin)
        ff = (f - fmin) / (fmax - fmin) / fff
        ff_old = ff

    for _ in range(iterations):
        order = round(order)
        if poly:
            t = median_filter(np.polyval(np.polyfit(xx, ff, order), xx), 3)
            tmp = np.polyval(np.polyfit(xx, np.clip(ff - t, 0, None) ** 2, order), xx)
            dev = np.sqrt(np.clip(tmp, 0, None))
        else:
            t = median_filter(opt_filter(ff, order, weight=weight, lambda2=lambda2), 3)
            tmp = opt_filter(
                np.clip(weight * (ff - t), 0, None),
                order,
                weight=weight,
                lambda2=lambda2,
            )
            dev = np.sqrt(np.clip(tmp, 0, None))

        ff = np.clip(t - eps, ff, t + dev * 3)
        dev2 = np.max(weight * np.abs(ff - ff_old))
        ff_old = ff
        if dev2 <= eps:
            break

    if poly:
        t = median_filter(np.polyval(np.polyfit(xx, ff, order), xx), 3)
        return t * (fmax - fmin) + fmin
    else:
        return t * fff * (fmax - fmin) + fmin


def opt_filter(y, par, par1=None, weight=None, lambda2=-1, maxiter=100):
    """
    Optimal filtering of 1D and 2D arrays.
    Uses tridiag in 1D case and sprsin and linbcg in 2D case.
    Written by N.Piskunov 8-May-2000

    Parameters
    ----------
    f : array
        1d or 2d array
    xwidth : int
        filter width (for 2d array width in x direction (1st index)
    ywidth : int
        (for 2d array only) filter width in y direction (2nd index) if ywidth is missing for 2d array, it set equal to xwidth
    weight : array(float)
        an array of the same size(s) as f containing values between 0 and 1
    lambda1: float
        regularization parameter
    maxiter : int
        maximum number of iteration for filtering of 2d array
    """

    y = np.asarray(y)

    if y.ndim not in [1, 2]:
        raise ValueError("Input y must have 1 or 2 dimensions")

    if par < 1:
        par = 1

    # 1D case
    if y.ndim == 1 or (y.ndim == 2 and (y.shape[0] == 1 or y.shape[1] == 1)):
        y = y.ravel()
        n = y.size

        if weight is None:
            weight = np.ones(n)
        elif np.isscalar(weight):
            weight = np.full(n, weight)
        else:
            weight = weight[:n]

        if lambda2 > 0:
            # Apply regularization lambda
            aij = np.zeros((5, n))
            # 2nd lower subdiagonal
            aij[0, 2:] = lambda2
            # Lower subdiagonal
            aij[1, 1] = -par - 2 * lambda2
            aij[1, 2:-1] = -par - 4 * lambda2
            aij[1, -1] = -par - 2 * lambda2
            # Main diagonal
            aij[2, 0] = weight[0] + par + lambda2
            aij[2, 1] = weight[1] + 2 * par + 5 * lambda2
            aij[2, 2:-2] = weight[2:-2] + 2 * par + 6 * lambda2
            aij[2, -2] = weight[-2] + 2 * par + 5 * lambda2
            aij[2, -1] = weight[-1] + par + lambda2
            # Upper subdiagonal
            aij[3, 0] = -par - 2 * lambda2
            aij[3, 1:-2] = -par - 4 * lambda2
            aij[3, -2] = -par - 2 * lambda2
            # 2nd lower subdiagonal
            aij[4, 0:-2] = lambda2
            # RHS
            b = weight * y

            f = solve_banded((2, 2), aij, b)
        else:
            a = np.full(n, -abs(par))
            b = np.copy(weight) + abs(par)
            b[1:-1] += abs(par)
            aba = np.array([a, b, a])

            f = solve_banded((1, 1), aba, weight * y)

        return f
    else:
        # 2D case
        if par1 is None:
            par1 = par
        if par == 0 and par1 == 0:
            raise ValueError("xwidth and ywidth can't both be 0")
        n = y.size
        nx, ny = y.shape

        lam_x = abs(par)
        lam_y = abs(par1)

        n = nx * ny
        ndiag = 2 * nx + 1
        aij = np.zeros((n, ndiag))
        aij[nx, 0] = weight[0, 0] + lam_x + lam_y
        aij[nx, 1:nx] = weight[0, 1:nx] + 2 * lam_x + lam_y
        aij[nx, nx : n - nx] = weight[1 : ny - 1] + 2 * (lam_x + lam_y)
        aij[nx, n - nx : n - 1] = weight[ny - 1, 0 : nx - 1] + 2 * lam_x + lam_y
        aij[nx, n - 1] = weight[ny - 1, nx - 1] + lam_x + lam_y

        aij[nx - 1, 1:n] = -lam_x
        aij[nx + 1, 0 : n - 1] = -lam_x

        ind = np.arrange(ny - 1) * nx + nx + nx * n
        aij[ind - 1] = aij[ind - 1] - lam_x
        aij[ind] = aij[ind] - lam_x

        ind = np.arrange(ny - 1) * nx + nx
        aij[nx + 1, ind - 1] = 0
        aij[nx - 1, ind] = 0

        aij[0, nx:n] = -lam_y
        aij[ndiag - 1, 0 : n - nx] = -lam_y

        rhs = f * weight

        model = solve_banded((nx, nx), aij, rhs)
        model = np.reshape(model, (ny, nx))
        return model


def helcorr(obs_long, obs_lat, obs_alt, ra2000, dec2000, jd, system="barycentric"):
    """
    calculates heliocentric Julian date, barycentric and heliocentric radial
    velocity corrections, using astropy functions

    Parameters
    ---------
    obs_long : float
        Longitude of observatory (degrees, western direction is positive)
    obs_lat : float
        Latitude of observatory (degrees)
    obs_alt : float
        Altitude of observatory (meters)
    ra2000 : float
        Right ascension of object for epoch 2000.0 (hours)
    dec2000 : float
        Declination of object for epoch 2000.0 (degrees)
    jd : float
        Julian date for the middle of exposure in MJD
    system : {"barycentric", "heliocentric"}, optional
        reference system of the result, barycentric: around earth-sun gravity center,
        heliocentric: around sun, usually barycentric is preferred (default: "barycentric)

    Returns
    -------
    correction : float
        radial velocity correction due to barycentre offset
    hjd : float
        Heliocentric Julian date for middle of exposure
    """

    # jd = 2400000.5 + jd
    jd = time.Time(jd, format="mjd")

    ra = coord.Longitude(ra2000, unit=u.hour)
    dec = coord.Latitude(dec2000, unit=u.degree)

    observatory = coord.EarthLocation.from_geodetic(obs_long, obs_lat, height=obs_alt)
    sky_location = coord.SkyCoord(ra, dec, obstime=jd, location=observatory)
    times = time.Time(jd, location=observatory)

    if system == "barycentric":
        correction = sky_location.radial_velocity_correction().to(u.km / u.s).value
        ltt = times.light_travel_time(sky_location)
    elif system == "heliocentric":
        correction = (
            sky_location.radial_velocity_correction("heliocentric").to(u.km / u.s).value
        )
        ltt = times.light_travel_time(sky_location, "heliocentric")
    else:
        raise AttributeError(
            "Could not parse system, values are: ('barycentric', 'heliocentric')"
        )

    times = (times.utc + ltt).value - 2400000

    return -correction, times
