import os

import numpy as np
from scipy.ndimage.filters import median_filter
from scipy.linalg import solve_banded, solve
from astropy.io import fits

# from modeinfo_uves import modeinfo_uves as modeinfo
from clipnflip import clipnflip
from modeinfo import modeinfo


def load_fits(fname, instrument, extension, **kwargs):
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
    instrument, mode = instrument.split("_")
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
        arr = value.flatten()[None, :]
        dtype = "D" if value.dtype == np.float64 else "E"
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


def make_index(ymin, ymax, xmin, xmax):
    # TODO
    # Define the indices for the pixels between two y arrays, e.g. pixels in an order
    # in x: the rows between ymin and ymax
    # in y: the column, but n times to match the x index

    index_x = np.array([np.arange(ymin[col], ymax[col]) for col in range(xmin, xmax)])
    index_y = np.array(
        [np.full(ymax[col] - ymin[col], col) for col in range(xmin, xmax)]
    )
    # Tranpose makes it so that the image orientation stays the same
    index = (index_x.T, index_y.T)
    return index


# TODO whats the actual difference between top, middle, and bottom?
# The clipping of t?


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
                dev = np.sqrt(np.polyval(np.polyfit(xx, t, order), xx))
                dev = np.nan_to_num(dev)
            else:
                t = np.tile(np.polyfit(xx, ff, order), len(f))
                t = np.polyfit(xx, np.clip(t - ff, 0, None) ** 2, order)
                t = np.tile(t, len(f))
                dev = np.sqrt(t)
                dev = np.nan_to_num(dev)
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
                dev = np.sqrt(np.polyval(np.polyfit(xx, (t - ff) ** 2, order), xx))
            else:
                t = np.tile(np.polyfit(xx, ff, order), len(f))
                dev = np.sqrt(np.tile(np.polyfit(xx, (t - ff) ** 2, order), len(f)))
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
            dev = np.sqrt(
                np.polyval(np.polyfit(xx, np.clip(ff - t, 0, None) ** 2, order), xx)
            )
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
