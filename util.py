import os
import argparse

import numpy as np
from itertools import product
from scipy.ndimage.filters import median_filter
from scipy.linalg import solve_banded, solve
from scipy.optimize import curve_fit
import scipy.interpolate
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# from modeinfo_uves import modeinfo_uves as modeinfo
from clipnflip import clipnflip
from instruments.instrument_info import modeinfo

import logging


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="General REDUCE script")
    parser.add_argument("-b", "--bias", action="store_true", help="Create master bias")
    parser.add_argument("-f", "--flat", action="store_true", help="Create master flat")
    parser.add_argument("-o", "--orders", action="store_true", help="Trace orders")
    parser.add_argument("-n", "--norm_flat", action="store_true", help="Normalize flat")
    parser.add_argument(
        "-w", "--wavecal", action="store_true", help="Prepare wavelength calibration"
    )
    parser.add_argument(
        "-s", "--science", action="store_true", help="Extract science spectrum"
    )

    parser.add_argument("instrument", type=str, help="instrument used")
    parser.add_argument("target", type=str, help="target star")

    args = parser.parse_args()
    instrument = args.instrument.upper()
    target = args.target.upper()

    steps_to_take = {
        "bias": args.bias,
        "flat": args.flat,
        "orders": args.orders,
        "norm_flat": args.norm_flat,
        "wavecal": args.wavecal,
        "science": args.science,
    }
    steps_to_take = [k for k, v in steps_to_take.items() if v]

    # if no steps are specified use all
    if len(steps_to_take) == 0:
        steps_to_take = ["bias", "flat", "orders", "norm_flat", "wavecal", "science"]

    return {"instrument": instrument, "target": target, "steps": steps_to_take}


def start_logging(log_file="log.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Command Line output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(ch_formatter)

    # Log file settings
    file = logging.FileHandler(log_file)
    file.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file.setFormatter(file_formatter)

    logger.addHandler(ch)
    logger.addHandler(file)

    logging.captureWarnings(True)

    logging.debug("----------------------")


def load_fits(fname, instrument, mode, extension, **kwargs):
    """
    load fits file, REDUCE style
    
    primary and extension header are combined
    modeinfo is applied to header
    data is clipnflipped
    mask is applied
    """
    hdu = fits.open(fname)
    header = hdu[extension].header
    header.extend(hdu[0].header, strip=False)
    header = modeinfo(header, instrument, mode)

    if kwargs.get("header_only", False):
        return header

    data = clipnflip(hdu[extension].data, header)

    if kwargs.get("dtype") is not None:
        data = data.astype(kwargs["dtype"])

    data = np.ma.masked_array(data, mask=kwargs.get("mask"))

    return data, header


def save_fits(fname, header, **kwargs):
    """
    Save fits with binary table in first extension
    Keywords describe data columns
    """
    primary = fits.PrimaryHDU(header=header)

    columns = []
    for key, value in kwargs.items():
        arr = value.flatten()[None, :].astype(np.float32)  # TODO good enough?
        dtype = "E"
        form = "%i%s" % (value.size, dtype)
        dim = str(value.shape[::-1])
        columns += [fits.Column(name=key.upper(), array=arr, format=form, dim=dim)]

    table = fits.BinTableHDU.from_columns(columns)

    hdulist = fits.HDUList(hdus=[primary, table])
    hdulist.writeto(fname, overwrite=True)


def swap_extension(fname, ext, path=None):
    """ exchange the extension of the given file with a new one """
    if path is None:
        path = os.path.dirname(fname)
    nameout = os.path.basename(fname)
    if nameout[-3:] == ".gz":
        nameout = nameout[:-3]
    nameout = nameout.rsplit(".", 1)[0]
    nameout = os.path.join(path, nameout + ext)
    return nameout


def find_first_index(arr, value):
    """ find the first element equal to value in the array arr """
    try:
        return next(i for i, v in enumerate(arr) if v == value)
    except StopIteration:
        raise Exception("Value %s not found" % value)


def interpolate_masked(masked):
    mask = np.ma.getmaskarray(masked)
    idx = np.nonzero(~mask)[0]
    interpol = np.interp(np.arange(len(masked)), idx, masked[idx])
    return interpol


def make_index(ymin, ymax, xmin, xmax, zero=0):
    # TODO
    # Define the indices for the pixels between two y arrays, e.g. pixels in an order
    # in x: the rows between ymin and ymax
    # in y: the column, but n times to match the x index
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
    # Tranpose makes it so that the image orientation stays the same
    index = (index_x.T, index_y.T + zero)
    return index


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

    gauss = lambda x, A0, A1, A2: A0 * np.exp(-((x - A1) / A2) ** 2 / 2)
    popt, _ = curve_fit(gauss, x, y, p0=[max(y), 1, 1])
    return gauss(x, *popt), popt


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


def polyfit2d(x, y, z, degree=1, plot=False):
    # Create combinations of degree of x and y
    # usually: [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), ....]
    if np.isscalar(degree):
        idx = [[i, j] for i, j in product(range(degree + 1), repeat=2)]
        coeff = np.zeros((degree + 1, degree + 1))
    else:
        idx = [[i, j] for i, j in product(range(degree[0] + 1), range(degree[1] + 1))]
        coeff = np.zeros((degree[0] + 1, degree[1] + 1))
        degree = max(degree)

    # We only want the combinations with maximum order COMBINED power
    idx = np.array(idx)
    idx = idx[idx[:, 0] + idx[:, 1] <= degree]

    # Calculate elements 1, x, y, x*y, x**2, y**2, ...
    A = np.array([np.power(x, i) * np.power(y, j) for i, j in idx]).T
    z = z.flatten()

    # Do least squares fit
    C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    # Reorder coefficients into numpy compatible 2d array
    for k, (i, j) in enumerate(idx):
        coeff[i, j] = C[k]

    if plot:
        # regular grid covering the domain of the data
        choice = np.random.choice(x.size, size=500, replace=False)
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
        ax.axis("equal")
        ax.axis("tight")
        plt.show()
    return coeff


def bezier_interp(x_old, y_old, x_new):
    # Handle masked arrays
    if np.ma.is_masked(x_old):
        x_old = np.ma.compressed(x_old)
        y_old = np.ma.compressed(y_old)
    knots, coef, order = scipy.interpolate.splrep(x_old, y_old)
    y_new = scipy.interpolate.BSpline(knots, coef, order)(x_new)
    return y_new

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

    syntax: bottom,f,{filter/order}[,iter=iter[,eps=eps
    where f      is the function to fit,
          filter is the smoothing parameter for the optimal filter.
                 if poly is set, it is interpreted as the order
                 of the smoothing polynomial,
          iter   is the maximum number of iterations [def: 40]
          eps    is convergence level [def: 0.001]
          mn     minimum function values to be considered [def: min(f)]
          mx     maximum function values to be considered [def: max(f)]
          lam2   constraint on 2nd derivative
          weight    vector of weights.
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


def middle(f, order=1, iterations=40, eps=0.001, poly=False, weight=1, **kwargs):
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

    syntax: middle,f,{filter/order}[,iter=iter[,eps=eps
    where f      is the function to fit,
          filter is the smoothing parameter for the optimal filter.
                 if poly is set, it is interpreted as the order
                 of the smoothing polynomial,
          iter   is the maximum number of iterations [def: 40]
          eps    is convergence level [def: 0.001]
          mn     minimum function values to be considered [def: min(f)]
          mx     maximum function values to be considered [def: max(f)]
          lam2   constraint on 2nd derivative
          wgt    vector of weights.
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
        ff_old = ff
    else:
        fmin = np.min(f) - 1
        fmax = np.max(f) + 1
        ff = (f - fmin) / (fmax - fmin)
        ff_old = ff
        n = len(f)

    for _ in range(iterations):
        if poly:
            if order > 0:  # this is a bug in rsi poly routine
                t = median_filter(np.polyval(np.polyfit(xx, ff, order), xx), 3)
                tmp = np.polyval(np.polyfit(xx, (t - ff) ** 2, order), xx)
                dev = np.sqrt(np.nan_to_num(tmp))
            else:
                t = np.tile(np.polyfit(xx, ff, order), len(f))
                t = np.tile(np.polyfit(xx, (t - ff) ** 2, order), len(f))
                t = np.nan_to_num(t)
                dev = np.sqrt(t)
        else:
            t = median_filter(opt_filter(ff, order, weight=weight, lambda2=lambda2), 3)
            dev = np.sqrt(
                opt_filter(
                    weight * (t - ff) ** 2, order, weight=weight, lambda2=lambda2
                )
            )
        ff = np.clip(np.clip(t - dev, ff, None), None, t + dev)
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


def top(f, order=1, iterations=40, eps=0.001, poly=False, weight=1, **kwargs):
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

    syntax: top,f,{filter/order}[,iter=iter[,eps=eps
    where f      is the function to fit,
          filter is the smoothing parameter for the optimal filter.
                 if poly is set, it is interpreted as the order
                 of the smoothing polynomial,
          iter   is the maximum number of iterations [def: 40]
          eps    is convergence level [def: 0.001]
          mn     minimum function values to be considered [def: min(f)]
          mx     maximum function values to be considered [def: max(f)]
          lam2   constraint on 2nd derivative
          wgt    vector of weights.
    """
    mn = kwargs.get("min", np.min(f))
    mx = kwargs.get("max", np.max(f))
    lambda2 = kwargs.get("lambda2", -1)

    if poly:
        j = np.where((f >= mn) & (f <= mx))
        xx = np.linspace(-1, 1, num=len(f))
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
        order = int(order)
        if poly:
            t = median_filter(np.polyval(np.polyfit(xx, ff, order), xx), 3)
            tmp = np.polyval(np.polyfit(xx, np.clip(ff - t, 0, None) ** 2, order), xx)
            tmp[np.isnan(tmp) | (tmp < 0)] = 0
            dev = np.sqrt(tmp)
        else:
            t = median_filter(opt_filter(ff, order, weight=weight, lambda2=lambda2), 3)
            dev = np.sqrt(
                opt_filter(
                    np.clip(weight * (ff - t), 0, None),
                    order,
                    weight=weight,
                    lambda2=lambda2,
                )
            )
        ff = np.clip(np.clip(t - eps, ff, None), None, t + dev * 3)
        dev2 = np.max(weight * np.abs(ff - ff_old))
        ff_old = ff
        if dev2 <= eps:
            break

    if poly:
        t = median_filter(np.polyval(np.polyfit(xx, ff, order), xx), 3)
        return t * (fmax - fmin) + fmin
    else:
        return t * fff * (fmax - fmin) + fmin


def opt_filter(y, par, par1=None, weight=None, lambda2=-1):
    """
    Optimal filtering of 1D and 2D arrays.
    Uses tridiag in 1D case and sprsin and linbcg in 2D case.
    Written by N.Piskunov 8-May-2000

    optimal filtering routine:
    syntax: r=opt_filter(f,xwidth[,ywidth[,weight=weight[,/double[,maxit=maxiter]]]])
    where:  f      - 1d or 2d array of type i,f or d
            xwidth - filter width (for 2d array width in x direction (1st index)
            ywidth - (for 2d array only) filter width in y direction (2nd index)
                     if ywidth is missing for 2d array, it set equal to xwidth
            weight - an array of the same size(s) as f containing values between 0 and 1
            double - perform calculations in double precision
            maxiter- maximum number of iteration for filtering of 2d array
            weight - weight for the function (values between 0 and 1)
       opt_filter solves the optimization problem for r:
            total(weight*(f - r)**2) + width*total((r(i) - r(i-1))**2) = min
    """

    if par < 1:
        par = 1

    if y.shape[0] == len(y) or y.shape[1] == len(y):
        if par < 0:
            return y
        n = len(y)
        if lambda2 > 0:
            aij = np.zeros((n, 5))
            # 2nd lower subdiagonal
            aij[0, 2 : n - 1 + 1] = lambda2
            # Lower subdiagonal
            aij[1, 1] = -par - 2 * lambda2
            aij[1, 2 : n - 2 + 1] = -par - 4 * lambda2
            aij[1, n - 1] = -par - 2 * lambda2
            # Main diagonal
            aij[2, 0] = weight[0] + par + lambda2
            aij[2, 1] = weight[1] + 2e0 * par + 5e0 * lambda2
            aij[2, 2 : n - 3 + 1] = weight[2 : n - 3 + 1] + 2e0 * par + 6e0 * lambda2
            aij[2, n - 2] = weight[n - 2] + 2e0 * par + 5e0 * lambda2
            aij[2, n - 1] = weight[n - 1] + par + lambda2
            # Upper subdiagonal
            aij[3, 0] = -par - 2e0 * lambda2
            aij[3, 1 : n - 3 + 1] = -par - 4e0 * lambda2
            aij[3, n - 2] = -par - 2e0 * lambda2
            # 2nd lower subdiagonal
            aij[4, 0 : n - 3 + 1] = lambda2
            # RHS
            b = weight * y

            b = solve_banded([1, 1], aij, b)
            # i = call_external(band_solv_name, 'bandsol', aij, b, long(n), 5)
            f = b
        else:
            a = np.full(n, -abs(par))
            weight = np.full(n, weight)
            b = np.array(
                [
                    weight[0] + abs(par),
                    *(weight[1:-1] + np.full(n - 2, 2 * abs(par))),
                    weight[-1] + abs(par),
                ]
            )
            aba = np.array([a, b, a])

            f = solve_banded((1, 1), aba, weight * np.ma.getdata(y))

        return f
    else:
        if par1 is None:
            par1 = par
        if par == 0 and par1 == 0:
            return y
        n = len(y)
        nc, nr = y.shape

        adiag = abs(par)
        bdiag = abs(par1)

        # Main diagonal first:
        aa = np.array(
            (
                1. + adiag + bdiag,
                np.full(nc - 2, 1. + 2. * adiag + bdiag),
                1. + adiag + bdiag,
                np.full(n - 2 * nc, 1. + 2. * adiag + 2. * bdiag),
                1. + adiag + bdiag,
                np.full(nc - 2, 1. + 2. * adiag + bdiag),
                1. + adiag + bdiag,
                np.full(n - 1, -adiag),
                np.full(n - 1, -adiag),
                np.full(n - nc, -bdiag),
                np.full(n - nc, -bdiag),
            )
        )  # lower sub-diagonal for y

        col = np.arange(nr - 2) * nc + nc  # special cases:
        aaa = np.full(nr - 2, 1. + adiag + 2. * bdiag)
        aa[col] = aaa  # last columns
        aa[col + nc - 1] = aaa  # first column
        col = n + np.arange(nr - 1) * nc + nc - 1
        aa[col] = 0.
        aa[col + n - 1] = 0.

        col = np.array(
            (
                np.arange(n),
                np.arange(n - 1) + 1,
                np.arange(n - 1),
                np.arange(n - nc) + nc,
                np.arange(n - nc),
            )
        )  # lower sub-diagonal for y

        row = np.array(
            (
                np.arange(n),
                np.arange(n - 1),
                np.arange(n - 1) + 1,
                np.arange(n - nc),
                np.arange(n - nc) + nc,
            )
        )  # lower sub-diagonal for y

        # aaa = sprsin(col, row, aa, n, thresh=-2. * (adiag > bdiag))
        col = bdiag
        row = adiag
        # aa = np.reshape(y, n)  # start with an initial guess at the solution.

        aa = solve(aaa, y)  # solve the linear system ax=b.
        aa.shape = nc, nr  # restore the shape of the result.
        return aaa
