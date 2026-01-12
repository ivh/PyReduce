"""
Find clusters of pixels with signal and fit polynomial traces.

Note on terminology:
- "trace": A single polynomial fit to a cluster of pixels (e.g., one fiber)
- "spectral order": A group of traces at similar wavelengths (e.g., all fibers in one echelle order)

The main function `trace` detects and fits individual traces.
Use `merge_traces` and `group_and_refit` to organize traces into spectral orders.
"""

import logging
from functools import cmp_to_key
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from numpy.polynomial.polynomial import Polynomial
from scipy.ndimage import binary_closing, binary_opening, label
from scipy.ndimage.filters import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from . import util

logger = logging.getLogger(__name__)


def whittaker_smooth(y, lam, axis=0):
    """Whittaker smoother (optimal filter).

    Solves: min sum((y - z)^2) + lam * sum((z[i] - z[i-1])^2)

    Parameters
    ----------
    y : array
        Input data (1D or 2D)
    lam : float
        Smoothing parameter (higher = smoother)
    axis : int
        Axis along which to smooth (for 2D arrays)

    Returns
    -------
    z : array
        Smoothed data
    """
    if y.ndim == 1:
        n = len(y)
        # Construct tridiagonal matrix: W + lam * D'D
        # where D is first-difference matrix
        diag_main = np.ones(n) + 2 * lam
        diag_main[0] = 1 + lam
        diag_main[-1] = 1 + lam
        diag_off = -lam * np.ones(n - 1)
        A = diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csc")
        return spsolve(A, y)
    else:
        # Apply along specified axis
        return np.apply_along_axis(lambda row: whittaker_smooth(row, lam), axis, y)


def fit(x, y, deg, regularization=0):
    # order = polyfit1d(y, x, deg, regularization)
    if deg == "best":
        order = best_fit(x, y)
    else:
        order = Polynomial.fit(y, x, deg=deg, domain=[]).coef[::-1]
    return order


def best_fit(x, y):
    aic = np.inf
    for k in range(5):
        coeff_new = fit(x, y, k)
        chisq = np.sum((np.polyval(coeff_new, y) - x) ** 2)
        aic_new = 2 * k + chisq
        if aic_new > aic:
            break
        else:
            coeff = coeff_new
            aic = aic_new
    return coeff


def determine_overlap_rating(xi, yi, xj, yj, mean_cluster_thickness, nrow, ncol, deg=2):
    # i and j are the indices of the 2 clusters
    i_left, i_right = yi.min(), yi.max()
    j_left, j_right = yj.min(), yj.max()

    # The number of pixels in the smaller cluster
    # this limits the accuracy of the fit
    n_min = min(i_right - i_left, j_right - j_left)

    # Fit a polynomial to each cluster
    order_i = fit(xi, yi, deg)
    order_j = fit(xj, yj, deg)

    # Get polynomial points inside cluster limits for each cluster and polynomial
    y_ii = np.polyval(order_i, np.arange(i_left, i_right))
    y_ij = np.polyval(order_i, np.arange(j_left, j_right))
    y_jj = np.polyval(order_j, np.arange(j_left, j_right))
    y_ji = np.polyval(order_j, np.arange(i_left, i_right))

    # difference of polynomials within each cluster limit
    diff_i = np.abs(y_ii - y_ji)
    diff_j = np.abs(y_ij - y_jj)

    ind_i = np.where((diff_i < mean_cluster_thickness) & (y_ji >= 0) & (y_ji < nrow))
    ind_j = np.where((diff_j < mean_cluster_thickness) & (y_ij >= 0) & (y_ij < nrow))

    # TODO: There should probably be some kind of normaliztion, that scales with the size of the cluster?
    # or possibly only use the closest pixels to determine overlap, since the polynomial is badly constrained outside of the bounds.
    overlap = min(n_min, len(ind_i[0])) + min(n_min, len(ind_j[0]))
    # overlap = overlap / ((i_right - i_left) + (j_right - j_left))
    overlap /= 2 * n_min
    if i_right < j_left:
        overlap *= 1 - (i_right - j_left) / ncol
    elif j_right < i_left:
        overlap *= 1 - (j_right - i_left) / ncol

    overlap_region = [-1, -1]
    if len(ind_i[0]) > 0:
        overlap_region[0] = np.min(ind_i[0]) + i_left
    if len(ind_j[0]) > 0:
        overlap_region[1] = np.max(ind_j[0]) + j_left

    return overlap, overlap_region


def create_merge_array(x, y, mean_cluster_thickness, nrow, ncol, deg, threshold):
    n_clusters = list(x.keys())
    nmax = len(n_clusters) ** 2
    merge = np.zeros((nmax, 5))
    for k, (i, j) in enumerate(combinations(n_clusters, 2)):
        overlap, region = determine_overlap_rating(
            x[i], y[i], x[j], y[j], mean_cluster_thickness, nrow, ncol, deg=deg
        )
        merge[k] = [i, j, overlap, *region]
    merge = merge[merge[:, 2] > threshold]
    merge = merge[np.argsort(merge[:, 2])[::-1]]
    return merge


def update_merge_array(
    merge, x, y, j, mean_cluster_thickness, nrow, ncol, deg, threshold
):
    j = int(j)
    n_clusters = np.array(list(x.keys()))
    update = []
    for i in n_clusters[n_clusters != j]:
        overlap, region = determine_overlap_rating(
            x[i], y[i], x[j], y[j], mean_cluster_thickness, nrow, ncol, deg=deg
        )
        if overlap <= threshold:
            # no , or little overlap
            continue
        update += [[i, j, overlap, *region]]
    if len(update) == 0:
        return merge
    update = np.array(update)
    merge = np.concatenate((merge, update))
    merge = merge[np.argsort(merge[:, 2])[::-1]]
    return merge


def calculate_mean_cluster_thickness(x, y):
    mean_cluster_thickness = 10  # Default thickness if no clusters found
    cluster_thicknesses = []

    for cluster in x.keys():
        if cluster == 0:
            continue  # Skip the background cluster if present

        # Get all y-coordinates and corresponding x-coordinates for this cluster
        y_coords = y[cluster]
        x_coords = x[cluster]

        # Find unique columns and precompute the x-coordinates for each column
        unique_columns = np.unique(y_coords)
        column_thicknesses = []

        for col in unique_columns:
            # Select x-coordinates that correspond to the current column
            col_indices = y_coords == col
            if np.any(col_indices):
                x_in_col = x_coords[col_indices]
                thickness = x_in_col.max() - x_in_col.min()
                column_thicknesses.append(thickness)

        # Average thickness per cluster, if any columns were processed
        if column_thicknesses:
            cluster_thicknesses.append(np.mean(column_thicknesses))

    # Compute the final mean thickness adjusted by the number of clusters
    if cluster_thicknesses:
        mean_cluster_thickness = (
            1.5 * np.mean(cluster_thicknesses) / len(cluster_thicknesses)
        )

    return mean_cluster_thickness


# origianl version
# def calculate_mean_cluster_thickness(x, y):
#     # Calculate mean cluster thickness
#     # TODO optimize
#     n_clusters = list(x.keys())
#     mean_cluster_thickness = 10
#     for cluster in n_clusters:
#         # individual columns of this cluster
#         columns = np.unique(y[cluster])
#         delta = 0
#         for col in columns:
#             # thickness of the cluster in each column
#             tmp = x[cluster][y[cluster] == col]
#             delta += np.max(tmp) - np.min(tmp)
#         mean_cluster_thickness += delta / len(columns)

#     mean_cluster_thickness *= 1.5 / len(n_clusters)
#     return mean_cluster_thickness


def delete(i, x, y, merge):
    del x[i], y[i]
    merge = merge[(merge[:, 0] != i) & (merge[:, 1] != i)]
    return x, y, merge


def combine(i, j, x, y, merge, mct, nrow, ncol, deg, threshold):
    # Merge pixels
    y[j] = np.concatenate((y[j], y[i]))
    x[j] = np.concatenate((x[j], x[i]))
    # Delete obsolete data
    x, y, merge = delete(i, x, y, merge)
    merge = merge[(merge[:, 0] != j) & (merge[:, 1] != j)]
    # Update merge array
    merge = update_merge_array(merge, x, y, j, mct, nrow, ncol, deg, threshold)
    return x, y, merge


def merge_clusters(
    img,
    x,
    y,
    n_clusters,
    manual=True,
    deg=2,
    auto_merge_threshold=0.9,
    merge_min_threshold=0.1,
    plot_title=None,
):
    """Merge clusters that belong together

    Parameters
    ----------
    img : array[nrow, ncol]
        the image the order trace is based on
    x : dict(int, array(int))
        x coordinates of cluster points
    y : dict(int, array(int))
        y coordinates of cluster points
    n_clusters : array(int)
        cluster numbers
    manual : bool, optional
        if True ask before merging clusters (default: True)
    deg : int, optional
        polynomial degree for fitting (default: 2)
    auto_merge_threshold : float, optional
        overlap threshold for automatic merging (default: 0.9)
    merge_min_threshold : float, optional
        minimum overlap to consider merging (default: 0.1)
    plot_title : str, optional
        title for plots

    Returns
    -------
    x : dict(int: array)
        x coordinates of clusters, key=cluster id
    y : dict(int: array)
        y coordinates of clusters, key=cluster id
    n_clusters : array(int)
        cluster labels
    """

    nrow, ncol = img.shape
    mct = calculate_mean_cluster_thickness(x, y)

    merge = create_merge_array(x, y, mct, nrow, ncol, deg, merge_min_threshold)

    if manual:
        plt.ion()
        plt.figure()  # dedicated figure for manual merge mode

    k = 0
    while k < len(merge):
        i, j, overlap, _, _ = merge[k]
        i, j = int(i), int(j)

        if overlap >= auto_merge_threshold and auto_merge_threshold != 1:
            answer = "y"
        elif manual:
            title = f"Probability: {overlap}"
            if plot_title is not None:
                title = f"{plot_title}\n{title}"
            plot_trace_pair(i, j, x, y, img, deg, title=title)
            while True:
                if manual:
                    answer = input("Merge? [y/n]")
                if answer in "ynrg":
                    break
        else:
            answer = "n"

        if answer == "y":
            # just merge automatically
            logger.info("Merging orders %i and %i", i, j)
            x, y, merge = combine(
                i, j, x, y, merge, mct, nrow, ncol, deg, merge_min_threshold
            )
        elif answer == "n":
            k += 1
        elif answer == "r":
            x, y, merge = delete(i, x, y, merge)
        elif answer == "g":
            x, y, merge = delete(j, x, y, merge)

    if manual:
        plt.close()
        plt.ioff()

    n_clusters = list(x.keys())
    return x, y, n_clusters


def fit_polynomials_to_clusters(x, y, clusters, degree, regularization=0):
    """Fits a polynomial of degree opower to points x, y in cluster clusters

    Parameters
    ----------
    x : dict(int: array)
        x coordinates seperated by cluster
    y : dict(int: array)
        y coordinates seperated by cluster
    clusters : list(int)
        cluster labels, equivalent to x.keys() or y.keys()
    degree : int
        degree of polynomial fit
    Returns
    -------
    traces : dict(int, array[degree+1])
        coefficients of polynomial fit for each cluster
    """

    traces = {c: fit(x[c], y[c], degree, regularization) for c in clusters}
    return traces


def plot_traces(im, x, y, clusters, traces, column_range, title=None):
    """Plot traces and image"""
    plt.figure()
    cluster_img = np.zeros(im.shape, dtype=im.dtype)
    for c in clusters:
        cluster_img[x[c], y[c]] = c + 1
    cluster_img = np.ma.masked_array(cluster_img, mask=cluster_img == 0)

    plt.subplot(121)
    # Handle non-finite values for plotting
    plot_im = np.where(np.isfinite(im), im, np.nan)
    valid = np.isfinite(plot_im)
    if np.any(valid):
        bot, top = np.percentile(plot_im[valid], (1, 99))
        if bot >= top:
            bot, top = None, None
    else:
        bot, top = None, None
    plt.imshow(plot_im, origin="lower", vmin=bot, vmax=top)
    plt.title("Input Image + Trace polynomials")
    plt.xlabel("x [pixel]")
    plt.ylabel("y [pixel]")
    plt.ylim([0, im.shape[0]])

    if traces is not None:
        for i, tr in enumerate(traces):
            x = np.arange(*column_range[i], 1)
            y = np.polyval(tr, x)
            plt.plot(x, y)

    plt.subplot(122)
    plt.imshow(cluster_img, cmap=plt.get_cmap("tab20"), origin="upper")
    plt.title("Detected Clusters + Trace Polynomials")
    plt.xlabel("x [pixel]")
    plt.ylabel("y [pixel]")

    if traces is not None:
        for i, tr in enumerate(traces):
            x = np.arange(*column_range[i], 1)
            y = np.polyval(tr, x)
            plt.plot(x, y)

    plt.ylim([0, im.shape[0]])
    if title is not None:
        plt.suptitle(title)
    util.show_or_save("trace_fitted")


def plot_trace_pair(i, j, x, y, img, deg, title=""):
    """Plot two trace candidates for merge decision"""
    _, ncol = img.shape

    trace_i = fit(x[i], y[i], deg)
    trace_j = fit(x[j], y[j], deg)

    xp = np.arange(ncol)
    yi = np.polyval(trace_i, xp)
    yj = np.polyval(trace_j, xp)

    xmin = min(np.min(x[i]), np.min(x[j])) - 50
    xmax = max(np.max(x[i]), np.max(x[j])) + 50
    ymin = min(np.min(y[i]), np.min(y[j])) - 50
    ymax = max(np.max(y[i]), np.max(y[j])) + 50

    yymin = min(max(0, ymin), img.shape[0] - 2)
    yymax = min(ymax, img.shape[0] - 1)
    xxmin = min(max(0, xmin), img.shape[1] - 2)
    xxmax = min(xmax, img.shape[1] - 1)

    vmin, vmax = np.percentile(img[yymin:yymax, xxmin:xxmax], (5, 95))

    plt.clf()
    plt.title(title)
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.plot(xp, yi, "r")
    plt.plot(xp, yj, "g")
    plt.plot(y[i], x[i], "r.")
    plt.plot(y[j], x[j], "g.")
    plt.xlim([ymin, ymax])
    plt.ylim([xmin, xmax])
    util.show_or_save(f"trace_merge_{i}_{j}")


def trace(
    im,
    min_cluster=None,
    min_width=None,
    filter_x=0,
    filter_y=None,
    filter_type="boxcar",
    noise=0,
    noise_relative=0,
    degree=4,
    border_width=None,
    degree_before_merge=2,
    regularization=0,
    closing_shape=(5, 5),
    opening_shape=(2, 2),
    plot=False,
    plot_title=None,
    manual=True,
    auto_merge_threshold=0.9,
    merge_min_threshold=0.1,
    sigma=0,
    debug_dir=None,
):
    """Identify and trace orders

    Parameters
    ----------
    im : array[nrow, ncol]
        order definition image
    min_cluster : int, optional
        minimum cluster size in pixels (default: 500)
    filter_x : int, optional
        Smoothing width along x-axis/dispersion direction (default: 0, no smoothing).
        Useful for noisy data or thin fiber traces.
    filter_y : int, optional
        Smoothing width along y-axis/cross-dispersion direction (default: auto).
        Used to estimate local background. For thin closely-spaced traces, use small values.
    filter_type : str, optional
        Type of smoothing filter: "boxcar" (default), "gaussian", or "whittaker".
        Boxcar is a uniform moving average. Whittaker preserves edges better.
    noise : float, optional
        Absolute noise threshold added to background (default: 0).
    noise_relative : float, optional
        Relative noise threshold as fraction of background (default: 0).
        If both noise and noise_relative are 0, defaults to 0.001 (0.1%).
    opower : int, optional
        polynomial degree of the order fit (default: 4)
    border_width : int or list of 4 int, optional
        Pixels to ignore at image edges for order tracing.
        If int, same value applied to all edges.
        If list: [top, bottom, left, right] for per-side control.
    plot : bool, optional
        wether to plot the final order fits (default: False)
    manual : bool, optional
        wether to manually select clusters to merge (strongly recommended) (default: True)
    debug_dir : str, optional
        if set, write intermediate images (filtered, background, mask) to this directory

    Returns
    -------
    traces : array[nord, opower+1]
        trace polynomial coefficients (in numpy order, i.e. largest exponent first)
    """

    # Convert to signed integer, to avoid underflow problems
    im = np.asanyarray(im)
    im = im.astype(int)

    if filter_y is None:
        col = im[:, im.shape[0] // 2]
        col = median_filter(col, 5)
        threshold = np.percentile(col, 90)
        npeaks = find_peaks(col, height=threshold)[0].size
        filter_y = im.shape[0] // (npeaks * 2)
        logger.info("Median filter size (y), estimated: %i", filter_y)
    elif filter_y <= 0:
        raise ValueError(f"Expected filter_y > 0, but got {filter_y}")

    if border_width is None:
        # find width of orders, based on central column
        col = im[:, im.shape[0] // 2]
        col = median_filter(col, 5)
        idx = np.argmax(col)
        width = peak_widths(col, [idx])[0][0]
        border_width = int(np.ceil(width))
        logger.info("Image border width, estimated: %i", border_width)

    # Normalize border_width to [top, bottom, left, right]
    if isinstance(border_width, (list, tuple)):
        if len(border_width) != 4:
            raise ValueError(
                f"border_width list must have 4 elements [top, bottom, left, right], "
                f"got {len(border_width)}"
            )
        bw_top, bw_bottom, bw_left, bw_right = [int(b) for b in border_width]
        if any(b < 0 for b in (bw_top, bw_bottom, bw_left, bw_right)):
            raise ValueError(
                f"All border_width values must be >= 0, got {border_width}"
            )
    elif isinstance(border_width, (int, float, np.integer, np.floating)):
        bw = int(border_width)
        if bw < 0:
            raise ValueError(f"Expected border_width >= 0, but got {bw}")
        bw_top = bw_bottom = bw_left = bw_right = bw
    else:
        raise TypeError(
            f"border_width must be int or list of 4 int, got {type(border_width)}"
        )

    if min_cluster is None:
        min_cluster = im.shape[1] // 4
        logger.info("Minimum cluster size, estimated: %i", min_cluster)
    elif not np.isscalar(min_cluster):
        raise TypeError(f"Expected scalar minimum cluster size, but got {min_cluster}")

    if min_width is None:
        min_width = 0.25
    if min_width == 0:
        pass
    elif isinstance(min_width, (float, np.floating)):
        min_width = int(min_width * im.shape[0])
        logger.info("Minimum trace width: %i", min_width)

    # Validate filter_type
    valid_filters = ("boxcar", "gaussian", "whittaker")
    if filter_type not in valid_filters:
        raise ValueError(
            f"filter_type must be one of {valid_filters}, got {filter_type}"
        )

    # Prepare image for thresholding
    # Convert masked values to NaN, interpolate, then back to regular ndarray
    if np.ma.is_masked(im):
        im_clean = np.ma.filled(im.astype(float), fill_value=np.nan)
        kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=2.5)
        im_clean = np.asarray(interpolate_replace_nans(im_clean, kernel))
        im_clean = np.nan_to_num(im_clean, nan=0.0)
    else:
        im_clean = np.asarray(im, dtype=float)

    # Select filter function based on filter_type
    if filter_type == "boxcar":

        def smooth(data, size, axis):
            return uniform_filter1d(data, int(size), axis=axis, mode="nearest")
    elif filter_type == "gaussian":

        def smooth(data, size, axis):
            return gaussian_filter1d(data, size, axis=axis)
    else:  # whittaker

        def smooth(data, size, axis):
            return whittaker_smooth(data, size, axis=axis)

    # Optionally smooth along x (dispersion) to reduce noise
    # Applied to both signal and background so we detect y-structure only
    if filter_x > 0:
        im_clean = smooth(im_clean, filter_x, axis=1)

    # Estimate local background by smoothing along y (cross-dispersion)
    background = smooth(im_clean, filter_y, axis=0)

    # Default to 0.1% relative threshold if neither noise parameter is set
    if noise == 0 and noise_relative == 0:
        noise_relative = 0.001
        logger.info("Using default noise_relative=0.001 (0.1%% of background)")

    # Threshold: pixels above local background are signal
    # Combines absolute (noise) and relative (noise_relative) thresholds
    mask = im_clean > background * (1 + noise_relative) + noise
    mask_initial = mask.copy()
    # remove borders
    if bw_top > 0:
        mask[:bw_top, :] = False
    if bw_bottom > 0:
        mask[-bw_bottom:, :] = False
    if bw_left > 0:
        mask[:, :bw_left] = False
    if bw_right > 0:
        mask[:, -bw_right:] = False
    # remove masked areas with no clusters
    mask = np.ma.filled(mask, fill_value=False)
    # close gaps inbetween clusters
    struct = np.full(closing_shape, 1)
    mask = binary_closing(mask, struct, border_value=1)
    # remove small lonely clusters
    struct = np.full(opening_shape, 1)
    # struct = generate_binary_structure(2, 1)
    mask = binary_opening(mask, struct)

    # Write debug output if requested
    if debug_dir is not None:
        import os

        from astropy.io import fits

        os.makedirs(debug_dir, exist_ok=True)
        fits.writeto(
            os.path.join(debug_dir, "trace_filtered.fits"),
            im_clean.astype(np.float32),
            overwrite=True,
        )
        fits.writeto(
            os.path.join(debug_dir, "trace_background.fits"),
            background.astype(np.float32),
            overwrite=True,
        )
        fits.writeto(
            os.path.join(debug_dir, "trace_mask_initial.fits"),
            mask_initial.astype(np.uint8),
            overwrite=True,
        )
        fits.writeto(
            os.path.join(debug_dir, "trace_mask_final.fits"),
            mask.astype(np.uint8),
            overwrite=True,
        )
        logger.info("Wrote debug images to %s", debug_dir)

    # label clusters
    clusters, _ = label(mask)

    # remove small clusters
    sizes = np.bincount(clusters.ravel())
    mask_sizes = sizes > min_cluster
    mask_sizes[0] = True  # This is the background, which we don't need to remove
    clusters[~mask_sizes[clusters]] = 0

    # # Reorganize x, y, clusters into a more convenient "pythonic" format
    # # x, y become dictionaries, with an entry for each cluster
    # # n is just a list of all cluster labels (ignore cluster == 0)
    n = np.unique(clusters)
    n = n[n != 0]
    x = {i: np.where(clusters == c)[0] for i, c in enumerate(n)}
    y = {i: np.where(clusters == c)[1] for i, c in enumerate(n)}

    def best_fit_degree(x, y):
        L1 = np.sum((np.polyval(np.polyfit(y, x, 1), y) - x) ** 2)
        L2 = np.sum((np.polyval(np.polyfit(y, x, 2), y) - x) ** 2)

        # aic1 = 2 + 2 * np.log(L1) + 4 / (x.size - 2)
        # aic2 = 4 + 2 * np.log(L2) + 12 / (x.size - 3)

        if L1 < L2:
            return 1
        else:
            return 2

    if sigma > 0:
        cluster_degrees = {i: best_fit_degree(x[i], y[i]) for i in x.keys()}
        bias = {i: np.polyfit(y[i], x[i], deg=cluster_degrees[i])[-1] for i in x.keys()}
        n = list(x.keys())
        yt = np.concatenate([y[i] for i in n])
        xt = np.concatenate([x[i] - bias[i] for i in n])
        coef = np.polyfit(yt, xt, deg=degree_before_merge)

        res = np.polyval(coef, yt)
        cutoff = sigma * (res - xt).std()

        # DEBUG plot
        # uy = np.unique(yt)
        # mask = np.abs(res - xt) > cutoff
        # plt.plot(yt, xt, ".")
        # plt.plot(yt[mask], xt[mask], "r.")
        # plt.plot(uy, np.polyval(coef, uy))
        # plt.show()
        #

        m = {
            i: np.abs(np.polyval(coef, y[i]) - (x[i] - bias[i])) < cutoff
            for i in x.keys()
        }

        k = max(x.keys()) + 1
        for i in range(1, k):
            new_img = np.zeros(im.shape, dtype=int)
            new_img[x[i][~m[i]], y[i][~m[i]]] = 1
            clusters, _ = label(new_img)

            x[i] = x[i][m[i]]
            y[i] = y[i][m[i]]
            if len(x[i]) == 0:
                del x[i], y[i]

            nnew = np.max(clusters)
            if nnew != 0:
                xidx, yidx = np.indices(im.shape)
                for j in range(1, nnew + 1):
                    xn = xidx[clusters == j]
                    yn = yidx[clusters == j]
                    if xn.size >= min_cluster:
                        x[k] = xn
                        y[k] = yn
                        k += 1
                # plt.imshow(clusters, origin="lower")
                # plt.show()

    if plot:  # pragma: no cover
        plt.figure()
        title = "Identified clusters"
        if plot_title is not None:
            title = f"{plot_title}\n{title}"
        plt.title(title)
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        clusters = np.ma.zeros(im.shape, dtype=int)
        for i in x.keys():
            clusters[x[i], y[i]] = i + 1
        clusters[clusters == 0] = np.ma.masked

        plt.imshow(clusters, origin="lower", cmap="prism")
        util.show_or_save("trace_clusters")

    # Merge clusters, if there are even any possible mergers left
    x, y, n = merge_clusters(
        im,
        x,
        y,
        n,
        manual=manual,
        deg=degree_before_merge,
        auto_merge_threshold=auto_merge_threshold,
        merge_min_threshold=merge_min_threshold,
        plot_title=plot_title,
    )

    if min_width > 0:
        sizes = {k: v.max() - v.min() for k, v in y.items()}
        mask_sizes = {k: v > min_width for k, v in sizes.items()}
        for k, v in mask_sizes.items():
            if not v:
                del x[k]
                del y[k]
        n = x.keys()

    traces = fit_polynomials_to_clusters(x, y, n, degree)

    # sort traces from bottom to top, using relative position

    def compare(i, j):
        _, xi, i_left, i_right = i
        _, xj, j_left, j_right = j

        if i_right < j_left or j_right < i_left:
            return xi.mean() - xj.mean()

        left = max(i_left, j_left)
        right = min(i_right, j_right)

        return xi[left:right].mean() - xj[left:right].mean()

    xp = np.arange(im.shape[1])
    keys = [(c, np.polyval(traces[c], xp), y[c].min(), y[c].max()) for c in x.keys()]
    keys = sorted(keys, key=cmp_to_key(compare))
    key = [k[0] for k in keys]

    n = np.arange(len(n), dtype=int)
    x = {c: x[key[c]] for c in n}
    y = {c: y[key[c]] for c in n}
    traces = np.array([traces[key[c]] for c in n])

    column_range = np.array([[np.min(y[i]), np.max(y[i]) + 1] for i in n])

    if plot:  # pragma: no cover
        plot_traces(im, x, y, n, traces, column_range, title=plot_title)

    return traces, column_range


def merge_traces(
    traces_a,
    column_range_a,
    traces_b,
    column_range_b,
    order_centers=None,
    order_numbers=None,
    ncols=None,
):
    """
    Merge two sets of traces from different illumination patterns.

    Traces are assigned to spectral orders based on their y-position at x=ncols/2
    compared to order_centers. Within each order, traces are sorted by y-position
    and assigned fiber IDs.

    Parameters
    ----------
    traces_a : array (n_traces_a, degree+1)
        Polynomial coefficients from first illumination set (even fibers)
    column_range_a : array (n_traces_a, 2)
        Column ranges for first set
    traces_b : array (n_traces_b, degree+1)
        Polynomial coefficients from second illumination set (odd fibers)
    column_range_b : array (n_traces_b, 2)
        Column ranges for second set
    order_centers : array-like, optional
        Expected y-positions of order centers at x=ncols/2
    order_numbers : array-like, optional
        Actual order numbers corresponding to each center. If None, uses 0-based indices.
    ncols : int, optional
        Number of columns in the image (for center calculation)

    Returns
    -------
    traces_by_order : dict
        {order_num: array (n_fibers, degree+1)} traces per order
    column_range_by_order : dict
        {order_num: array (n_fibers, 2)} column ranges per order
    fiber_ids_by_order : dict
        {order_num: array (n_fibers,)} fiber indices per order (0-74)
    """
    if len(traces_a) == 0 and len(traces_b) == 0:
        return {}, {}, {}

    # Combine all traces
    if len(traces_a) > 0 and len(traces_b) > 0:
        traces = np.vstack([traces_a, traces_b])
        column_range = np.vstack([column_range_a, column_range_b])
        is_even = np.concatenate(
            [np.ones(len(traces_a), dtype=bool), np.zeros(len(traces_b), dtype=bool)]
        )
    elif len(traces_a) > 0:
        traces = traces_a
        column_range = column_range_a
        is_even = np.ones(len(traces_a), dtype=bool)
    else:
        traces = traces_b
        column_range = column_range_b
        is_even = np.zeros(len(traces_b), dtype=bool)

    # Evaluate y-position at center column
    if ncols is None:
        ncols = int(np.max(column_range[:, 1]))
    x_center = ncols // 2
    y_positions = np.array([np.polyval(t, x_center) for t in traces])

    # Assign each trace to nearest order center
    if order_centers is None:
        # No order centers - put all in order 0
        order_ids = np.zeros(len(traces), dtype=int)
    else:
        order_centers = np.array(order_centers)
        center_indices = np.array(
            [np.argmin(np.abs(order_centers - y)) for y in y_positions]
        )
        if order_numbers is not None:
            order_numbers = np.array(order_numbers)
            order_ids = order_numbers[center_indices]
        else:
            order_ids = center_indices

    # Group by order, sort by y within each order, assign fiber IDs
    traces_by_order = {}
    column_range_by_order = {}
    fiber_ids_by_order = {}

    for order_idx in np.unique(order_ids):
        mask = order_ids == order_idx
        order_traces = traces[mask]
        order_cr = column_range[mask]
        order_y = y_positions[mask]
        order_is_even = is_even[mask]

        # Sort by y-position within this order
        sort_idx = np.argsort(order_y)
        order_traces = order_traces[sort_idx]
        order_cr = order_cr[sort_idx]
        order_is_even = order_is_even[sort_idx]

        # Assign fiber IDs: even fibers get 1,3,5,...  odd get 2,4,6,...
        fiber_ids = np.zeros(len(order_traces), dtype=int)
        even_count = 0
        odd_count = 0
        for i, is_e in enumerate(order_is_even):
            if is_e:
                fiber_ids[i] = even_count * 2 + 1
                even_count += 1
            else:
                fiber_ids[i] = odd_count * 2 + 2
                odd_count += 1

        traces_by_order[order_idx] = order_traces
        column_range_by_order[order_idx] = order_cr
        fiber_ids_by_order[order_idx] = fiber_ids

    return traces_by_order, column_range_by_order, fiber_ids_by_order


def group_and_refit(
    traces_by_order, column_range_by_order, fiber_ids_by_order, groups, degree=4
):
    """
    Group physical fiber traces into logical fibers and refit polynomials.

    For each spectral order and each fiber group, evaluates all member
    polynomials at each column, averages the y-positions, and fits a new
    polynomial.

    Parameters
    ----------
    traces_by_order : dict
        {order_idx: array (n_fibers, degree+1)} traces per order
    column_range_by_order : dict
        {order_idx: array (n_fibers, 2)} column ranges per order
    fiber_ids_by_order : dict
        {order_idx: array (n_fibers,)} fiber IDs per order (0-74)
    groups : dict
        Mapping of group name to fiber index range, e.g.:
        {'A': (0, 36), 'cal': (36, 38), 'B': (38, 75)}
    degree : int
        Polynomial degree for refitted traces

    Returns
    -------
    logical_traces : dict
        {group_name: array (n_orders, degree+1)} polynomials per group
    logical_column_range : array (n_orders, 2)
        Column range per order
    fiber_counts : dict
        {group_name: dict {order_idx: int}} fiber counts per order
    """
    from numpy.polynomial.polynomial import Polynomial

    order_indices = sorted(traces_by_order.keys())

    logical_traces = {name: [] for name in groups.keys()}
    logical_column_range = []
    fiber_counts = {name: {} for name in groups.keys()}

    for order_idx in order_indices:
        traces = traces_by_order[order_idx]
        column_range = column_range_by_order[order_idx]
        fiber_ids = fiber_ids_by_order[order_idx]

        # Find shared column range for this order
        col_min = np.max(column_range[:, 0])
        col_max = np.min(column_range[:, 1])
        x_eval = np.arange(col_min, col_max)
        logical_column_range.append([col_min, col_max])

        for group_name, (start, end) in groups.items():
            # Find traces belonging to this group
            mask = (fiber_ids >= start) & (fiber_ids < end)
            group_traces = traces[mask]

            if len(group_traces) == 0:
                logger.warning(
                    "No traces for group %s in order %d", group_name, order_idx
                )
                # Use NaN coefficients for missing groups
                logical_traces[group_name].append(np.full(degree + 1, np.nan))
                fiber_counts[group_name][order_idx] = 0
                continue

            # Evaluate all traces at each column and average
            y_values = np.array([np.polyval(t, x_eval) for t in group_traces])
            y_mean = np.mean(y_values, axis=0)

            # Fit new polynomial to averaged positions
            fit = Polynomial.fit(x_eval, y_mean, deg=degree, domain=[])
            coeffs = fit.coef[::-1]  # Convert to np.polyval order

            logical_traces[group_name].append(coeffs)
            fiber_counts[group_name][order_idx] = len(group_traces)

    # Convert lists to arrays
    for name in groups.keys():
        logical_traces[name] = np.array(logical_traces[name])

    logical_column_range = np.array(logical_column_range)

    return logical_traces, logical_column_range, fiber_counts


def _merge_fiber_traces(traces, column_range, merge_method, degree=4):
    """Apply merge method to a set of fiber traces.

    Parameters
    ----------
    traces : ndarray (n_fibers, degree+1)
        Polynomial coefficients for each fiber
    column_range : ndarray (n_fibers, 2)
        Column range for each fiber
    merge_method : str or list[int]
        "average", "center", or list of 1-based indices
    degree : int
        Polynomial degree for refitted traces (used with "average")

    Returns
    -------
    merged_traces : ndarray (n_output, degree+1)
    merged_cr : ndarray (n_output, 2)
    """
    n_fibers = len(traces)
    if n_fibers == 0:
        return np.empty((0, traces.shape[1])), np.empty((0, 2))

    # Find shared column range
    col_min = int(np.max(column_range[:, 0]))
    col_max = int(np.min(column_range[:, 1]))
    shared_cr = np.array([[col_min, col_max]])

    if merge_method == "center":
        # Select middle trace
        idx = n_fibers // 2
        return traces[idx : idx + 1], column_range[idx : idx + 1]

    elif merge_method == "average":
        # Average y-positions and refit
        x_eval = np.arange(col_min, col_max)
        y_values = np.array([np.polyval(t, x_eval) for t in traces])
        y_mean = np.mean(y_values, axis=0)

        fit = Polynomial.fit(x_eval, y_mean, deg=degree, domain=[])
        coeffs = fit.coef[::-1]  # Convert to np.polyval order
        return coeffs.reshape(1, -1), shared_cr

    elif isinstance(merge_method, list):
        # Select specific indices (1-based within group)
        indices = [i - 1 for i in merge_method]  # Convert to 0-based
        valid = [i for i in indices if 0 <= i < n_fibers]
        if not valid:
            logger.warning("No valid indices in merge method %s", merge_method)
            return np.empty((0, traces.shape[1])), np.empty((0, 2))
        return traces[valid], column_range[valid]

    else:
        raise ValueError(f"Unknown merge method: {merge_method}")


def _load_order_centers(filepath, instrument_dir=None):
    """Load order_centers from a YAML file.

    Parameters
    ----------
    filepath : str
        Path to YAML file (absolute or relative to instrument_dir)
    instrument_dir : str, optional
        Directory to resolve relative paths against

    Returns
    -------
    order_centers : dict[int, float]
        Order number -> y-position at detector center
    """
    from pathlib import Path

    import yaml

    path = Path(filepath)
    if not path.is_absolute() and instrument_dir:
        path = Path(instrument_dir) / path

    with open(path) as f:
        data = yaml.safe_load(f)

    # Handle both flat dict and nested structure
    if "order_centers" in data:
        data = data["order_centers"]

    return {int(k): float(v) for k, v in data.items()}


def _load_bundle_centers(filepath, instrument_dir=None):
    """Load bundle_centers from a YAML file.

    Parameters
    ----------
    filepath : str
        Path to YAML file (absolute or relative to instrument_dir)
    instrument_dir : str, optional
        Directory to resolve relative paths against

    Returns
    -------
    bundle_centers : dict[int, float]
        Bundle ID -> y-position at detector center
    """
    from pathlib import Path

    import yaml

    path = Path(filepath)
    if not path.is_absolute() and instrument_dir:
        path = Path(instrument_dir) / path

    with open(path) as f:
        data = yaml.safe_load(f)

    if "bundle_centers" in data:
        data = data["bundle_centers"]

    return {int(k): float(v) for k, v in data.items()}


def _assign_traces_to_bundles(traces, column_range, bundle_centers):
    """Assign each trace to a bundle based on y-position.

    Parameters
    ----------
    traces : ndarray (n_traces, degree+1)
        Polynomial coefficients for each trace
    column_range : ndarray (n_traces, 2)
        Column range for each trace
    bundle_centers : dict[int, float]
        Bundle ID -> y-position at detector center

    Returns
    -------
    bundle_traces : dict[int, tuple[ndarray, ndarray]]
        {bundle_id: (traces, column_range)} for traces assigned to each bundle,
        sorted by y-position within each bundle
    """
    n_traces = len(traces)
    if n_traces == 0:
        return {}

    x_center = int(np.mean(column_range[:, 0] + column_range[:, 1]) / 2)
    y_positions = np.array([np.polyval(t, x_center) for t in traces])

    bundle_ids = np.array(list(bundle_centers.keys()))
    y_centers = np.array([bundle_centers[b] for b in bundle_ids])

    bundle_traces = {b: [] for b in bundle_ids}
    for i, y in enumerate(y_positions):
        distances = np.abs(y - y_centers)
        closest_idx = np.argmin(distances)
        b = bundle_ids[closest_idx]
        bundle_traces[b].append((y, traces[i], column_range[i]))

    result = {}
    for b, items in bundle_traces.items():
        if items:
            items.sort(key=lambda x: x[0])
            tr_list = [item[1] for item in items]
            cr_list = [item[2] for item in items]
            result[b] = (np.array(tr_list), np.array(cr_list))

    return result


def _merge_bundle_traces(
    traces, column_range, merge_method, degree, expected_size, bundle_center
):
    """Apply merge method to bundle traces, handling missing fibers.

    Parameters
    ----------
    traces : ndarray (n_fibers, degree+1)
        Polynomial coefficients for fibers in this bundle
    column_range : ndarray (n_fibers, 2)
        Column range for each fiber
    merge_method : str or list[int]
        "average", "center", or list of 1-based indices
    degree : int
        Polynomial degree for refitted traces
    expected_size : int
        Expected number of fibers in bundle
    bundle_center : float
        Y-position of bundle center (for fallback when fibers missing)

    Returns
    -------
    merged_traces : ndarray (n_output, degree+1)
    merged_cr : ndarray (n_output, 2)
    """
    n_fibers = len(traces)
    if n_fibers == 0:
        return np.empty((0, degree + 1)), np.empty((0, 2))

    col_min = int(np.max(column_range[:, 0]))
    col_max = int(np.min(column_range[:, 1]))
    x_center = (col_min + col_max) // 2

    if merge_method == "center":
        if n_fibers == expected_size:
            # All present: pick middle index
            idx = n_fibers // 2
            return traces[idx : idx + 1], column_range[idx : idx + 1]
        else:
            # Missing fibers: pick trace closest to bundle_center
            y_positions = np.array([np.polyval(t, x_center) for t in traces])
            distances = np.abs(y_positions - bundle_center)
            idx = np.argmin(distances)
            return traces[idx : idx + 1], column_range[idx : idx + 1]

    elif merge_method == "average":
        # Average all present fibers
        x_eval = np.arange(col_min, col_max)
        y_values = np.array([np.polyval(t, x_eval) for t in traces])
        y_mean = np.mean(y_values, axis=0)

        fit = Polynomial.fit(x_eval, y_mean, deg=degree, domain=[])
        coeffs = fit.coef[::-1]
        shared_cr = np.array([[col_min, col_max]])
        return coeffs.reshape(1, -1), shared_cr

    elif isinstance(merge_method, list):
        # Select specific indices - this may fail with missing fibers
        indices = [i - 1 for i in merge_method]
        valid = [i for i in indices if 0 <= i < n_fibers]
        if not valid:
            logger.warning(
                "No valid indices in merge method %s for %d fibers",
                merge_method,
                n_fibers,
            )
            return np.empty((0, degree + 1)), np.empty((0, 2))
        return traces[valid], column_range[valid]

    else:
        raise ValueError(f"Unknown merge method: {merge_method}")


def _assign_traces_to_orders(traces, column_range, order_centers):
    """Assign each trace to a spectral order based on y-position.

    Parameters
    ----------
    traces : ndarray (n_traces, degree+1)
        Polynomial coefficients for each trace
    column_range : ndarray (n_traces, 2)
        Column range for each trace
    order_centers : dict[int, float]
        Order number -> y-position at detector center

    Returns
    -------
    order_traces : dict[int, tuple[ndarray, ndarray]]
        {order_m: (traces, column_range)} for traces assigned to each order
    """
    n_traces = len(traces)
    if n_traces == 0:
        return {}

    # Get detector center x-coordinate
    x_center = int(np.mean(column_range[:, 0] + column_range[:, 1]) / 2)

    # Evaluate each trace at center to get y-position
    y_positions = np.array([np.polyval(t, x_center) for t in traces])

    # Get order numbers and centers as arrays
    order_nums = np.array(list(order_centers.keys()))
    y_centers = np.array([order_centers[m] for m in order_nums])

    # Assign each trace to closest order, keeping track of y-position for sorting
    order_traces = {m: [] for m in order_nums}  # list of (y, trace, cr)
    for i, y in enumerate(y_positions):
        distances = np.abs(y - y_centers)
        closest_idx = np.argmin(distances)
        m = order_nums[closest_idx]
        order_traces[m].append((y, traces[i], column_range[i]))

    # Sort by y-position within each order and convert to arrays
    result = {}
    for m, items in order_traces.items():
        if items:
            # Sort by y-position (fiber number increases with y typically)
            items.sort(key=lambda x: x[0])
            tr_list = [item[1] for item in items]
            cr_list = [item[2] for item in items]
            result[m] = (np.array(tr_list), np.array(cr_list))

    return result


def organize_fibers(
    traces, column_range, fibers_config, degree=4, instrument_dir=None, channel_index=0
):
    """Organize traced fibers into groups according to config.

    Takes raw fiber traces and groups them according to either explicit
    named groups or repeating bundle patterns.

    For per_order=True instruments, grouping is applied within each spectral
    order, returning {group: {order_m: trace}}.

    Parameters
    ----------
    traces : ndarray (n_fibers, degree+1)
        Polynomial coefficients for each fiber trace
    column_range : ndarray (n_fibers, 2)
        Column range for each fiber
    fibers_config : FibersConfig
        Configuration specifying groups or bundles
    degree : int
        Polynomial degree for refitted traces (used with "average" merge)
    instrument_dir : str, optional
        Directory for resolving relative order_centers_file paths
    channel_index : int
        Index into per-channel lists (for multi-channel instruments)

    Returns
    -------
    group_traces : dict
        For per_order=False: {group_name: ndarray} - merged trace(s) per group
        For per_order=True: {group_name: {order_m: ndarray}} - per order per group
    group_column_range : dict
        Same structure as group_traces but with column ranges
    group_fiber_counts : dict[str, int]
        Number of physical fibers in each group (per order if per_order=True)
    """
    n_fibers = len(traces)
    group_traces = {}
    group_column_range = {}
    group_fiber_counts = {}

    # Handle per-order grouping
    if fibers_config.per_order:
        # Load order_centers from file if not inline
        order_centers = fibers_config.order_centers
        if order_centers is None and fibers_config.order_centers_file:
            # Handle per-channel list
            centers_file = fibers_config.order_centers_file
            if isinstance(centers_file, list):
                centers_file = centers_file[channel_index]
            order_centers = _load_order_centers(centers_file, instrument_dir)

        order_traces = _assign_traces_to_orders(traces, column_range, order_centers)

        # Validate fibers per order if specified
        fibers_per_order = fibers_config.fibers_per_order
        if isinstance(fibers_per_order, list):
            fibers_per_order = fibers_per_order[channel_index]
        if fibers_per_order is not None:
            for m, (tr, _cr) in order_traces.items():
                if len(tr) != fibers_per_order:
                    logger.warning(
                        "Order %d has %d fibers, expected %d",
                        m,
                        len(tr),
                        fibers_per_order,
                    )

        # Apply grouping within each order
        if fibers_config.groups is not None:
            for name, group_cfg in fibers_config.groups.items():
                group_traces[name] = {}
                group_column_range[name] = {}
                group_fiber_counts[name] = 0

                for m, (order_tr, order_cr) in sorted(order_traces.items()):
                    start, end = group_cfg.range
                    start_idx = start - 1
                    end_idx = end - 1

                    n_in_order = len(order_tr)
                    if end_idx > n_in_order:
                        logger.warning(
                            "Group %s range [%d, %d) exceeds fiber count %d in order %d",
                            name,
                            start,
                            end,
                            n_in_order,
                            m,
                        )
                        end_idx = min(end_idx, n_in_order)
                    start_idx = max(start_idx, 0)

                    if start_idx >= end_idx:
                        continue

                    fiber_tr = order_tr[start_idx:end_idx]
                    fiber_cr = order_cr[start_idx:end_idx]

                    merged_tr, merged_cr = _merge_fiber_traces(
                        fiber_tr, fiber_cr, group_cfg.merge, degree
                    )

                    group_traces[name][m] = merged_tr
                    group_column_range[name][m] = merged_cr
                    group_fiber_counts[name] = end_idx - start_idx

        elif fibers_config.bundles is not None:
            bundle_cfg = fibers_config.bundles
            bundle_size = bundle_cfg.size

            # Process each order's bundles
            for m, (order_tr, order_cr) in sorted(order_traces.items()):
                n_in_order = len(order_tr)
                if n_in_order % bundle_size != 0:
                    raise ValueError(
                        f"Order {m} has {n_in_order} fibers, "
                        f"not divisible by bundle size {bundle_size}"
                    )

                n_bundles = n_in_order // bundle_size
                for i in range(n_bundles):
                    name = f"bundle_{i + 1}"
                    if name not in group_traces:
                        group_traces[name] = {}
                        group_column_range[name] = {}
                        group_fiber_counts[name] = bundle_size

                    start_idx = i * bundle_size
                    end_idx = (i + 1) * bundle_size

                    bundle_tr = order_tr[start_idx:end_idx]
                    bundle_cr = order_cr[start_idx:end_idx]

                    merged_tr, merged_cr = _merge_fiber_traces(
                        bundle_tr, bundle_cr, bundle_cfg.merge, degree
                    )

                    group_traces[name][m] = merged_tr
                    group_column_range[name][m] = merged_cr

        return group_traces, group_column_range, group_fiber_counts

    # Non-per-order grouping (original behavior)
    if fibers_config.groups is not None:
        # Explicit named groups
        for name, group_cfg in fibers_config.groups.items():
            start, end = group_cfg.range
            # Convert 1-based to 0-based indices
            start_idx = start - 1
            end_idx = end - 1  # half-open, so end is exclusive

            if start_idx < 0 or end_idx > n_fibers:
                logger.warning(
                    "Group %s range [%d, %d) exceeds fiber count %d",
                    name,
                    start,
                    end,
                    n_fibers,
                )
                end_idx = min(end_idx, n_fibers)
                start_idx = max(start_idx, 0)

            group_tr = traces[start_idx:end_idx]
            group_cr = column_range[start_idx:end_idx]
            n_in_group = len(group_tr)

            merged_tr, merged_cr = _merge_fiber_traces(
                group_tr, group_cr, group_cfg.merge, degree
            )

            group_traces[name] = merged_tr
            group_column_range[name] = merged_cr
            group_fiber_counts[name] = n_in_group

    elif fibers_config.bundles is not None:
        # Repeating bundle pattern
        bundle_cfg = fibers_config.bundles
        bundle_size = bundle_cfg.size

        # Load bundle_centers if provided (handles missing fibers)
        bundle_centers = bundle_cfg.bundle_centers
        if bundle_centers is None and bundle_cfg.bundle_centers_file:
            centers_file = bundle_cfg.bundle_centers_file
            if isinstance(centers_file, list):
                centers_file = centers_file[channel_index]
            bundle_centers = _load_bundle_centers(centers_file, instrument_dir)

        if bundle_centers is not None:
            # Assign traces to bundles by proximity to bundle centers
            bundle_traces_dict = _assign_traces_to_bundles(
                traces, column_range, bundle_centers
            )

            for bundle_id, (bundle_tr, bundle_cr) in sorted(bundle_traces_dict.items()):
                name = f"bundle_{bundle_id}"
                n_in_bundle = len(bundle_tr)

                if n_in_bundle != bundle_size:
                    logger.info(
                        "Bundle %d has %d fibers (expected %d)",
                        bundle_id,
                        n_in_bundle,
                        bundle_size,
                    )

                merged_tr, merged_cr = _merge_bundle_traces(
                    bundle_tr,
                    bundle_cr,
                    bundle_cfg.merge,
                    degree,
                    bundle_size,
                    bundle_centers[bundle_id],
                )

                group_traces[name] = merged_tr
                group_column_range[name] = merged_cr
                group_fiber_counts[name] = n_in_bundle

            # Also handle bundles with zero traces (all fibers missing)
            for bundle_id in bundle_centers:
                name = f"bundle_{bundle_id}"
                if name not in group_traces:
                    logger.warning("Bundle %d has no traces assigned", bundle_id)
                    group_traces[name] = np.empty((0, degree + 1))
                    group_column_range[name] = np.empty((0, 2))
                    group_fiber_counts[name] = 0

        else:
            # Fixed-size division (original behavior, requires exact divisibility)
            if n_fibers % bundle_size != 0:
                raise ValueError(
                    f"Number of fibers ({n_fibers}) not divisible by bundle size ({bundle_size}). "
                    f"Use bundle_centers_file for instruments with missing fibers."
                )

            n_bundles = n_fibers // bundle_size
            if bundle_cfg.count is not None and bundle_cfg.count != n_bundles:
                raise ValueError(
                    f"Expected {bundle_cfg.count} bundles but found {n_bundles}"
                )

            # Create a group for each bundle
            for i in range(n_bundles):
                name = f"bundle_{i + 1}"  # 1-based naming
                start_idx = i * bundle_size
                end_idx = (i + 1) * bundle_size

                bundle_tr = traces[start_idx:end_idx]
                bundle_cr = column_range[start_idx:end_idx]

                merged_tr, merged_cr = _merge_fiber_traces(
                    bundle_tr, bundle_cr, bundle_cfg.merge, degree
                )

                group_traces[name] = merged_tr
                group_column_range[name] = merged_cr
                group_fiber_counts[name] = bundle_size

    return group_traces, group_column_range, group_fiber_counts


def _stack_per_order_traces(order_dict):
    """Stack per-order traces {order_m: trace} into arrays ordered by m."""
    if not order_dict:
        return np.empty((0, 1)), np.empty((0, 2))
    sorted_orders = sorted(order_dict.keys())
    traces = np.vstack([order_dict[m] for m in sorted_orders])
    return traces, sorted_orders


def select_traces_for_step(
    raw_traces,
    raw_cr,
    group_traces,
    group_cr,
    fibers_config,
    step_name,
):
    """Select which traces to use for a given reduction step.

    Looks up fibers_config.use[step_name] to determine selection.

    Parameters
    ----------
    raw_traces : ndarray (n_fibers, degree+1)
        All individual fiber traces
    raw_cr : ndarray (n_fibers, 2)
        Column ranges for individual fibers
    group_traces : dict
        Grouped/merged traces from organize_fibers()
        Non-per-order: {group_name: ndarray}
        Per-order: {group_name: {order_m: ndarray}}
    group_cr : dict
        Column ranges with same structure as group_traces
    fibers_config : FibersConfig or None
        Fiber configuration (may be None for single-fiber instruments)
    step_name : str
        Name of the reduction step (e.g., "science", "curvature")

    Returns
    -------
    selected : dict[str, tuple[ndarray, ndarray]]
        {group_name: (traces, column_range)} for each selected group
        For "all" or "groups" selection, returns {"all": (traces, cr)}
    """
    # No fiber config means use raw traces
    if fibers_config is None:
        return {"all": (raw_traces, raw_cr)}

    # No groups/bundles defined means use raw traces
    if fibers_config.groups is None and fibers_config.bundles is None:
        return {"all": (raw_traces, raw_cr)}

    # Determine selection for this step
    selection = "groups"  # default when groups/bundles defined
    if fibers_config.use is not None and step_name in fibers_config.use:
        selection = fibers_config.use[step_name]

    per_order = fibers_config.per_order

    if selection == "all":
        return {"all": (raw_traces, raw_cr)}

    elif selection == "groups":
        # Stack all group traces into single array
        all_traces = []
        all_cr = []
        for name in sorted(group_traces.keys()):
            if per_order:
                # Per-order: {group: {order: trace}} - stack orders for each group
                traces, _ = _stack_per_order_traces(group_traces[name])
                cr, _ = _stack_per_order_traces(group_cr[name])
                all_traces.append(traces)
                all_cr.append(cr)
            else:
                all_traces.append(group_traces[name])
                all_cr.append(group_cr[name])
        if all_traces:
            return {"all": (np.vstack(all_traces), np.vstack(all_cr))}
        else:
            return {"all": (raw_traces, raw_cr)}

    elif isinstance(selection, list):
        # Select specific groups by name - keep them separate
        result = {}
        for name in selection:
            if name not in group_traces:
                logger.warning("Group '%s' not found in trace data", name)
                continue

            if per_order:
                # Per-order: stack orders for this group
                traces, _ = _stack_per_order_traces(group_traces[name])
                cr, _ = _stack_per_order_traces(group_cr[name])
                result[name] = (traces, cr)
            else:
                result[name] = (group_traces[name], group_cr[name])

        if not result:
            logger.warning("No valid groups selected, using all raw traces")
            return {"all": (raw_traces, raw_cr)}
        return result

    else:
        logger.warning("Unknown selection type: %s, using raw traces", selection)
        return {"all": (raw_traces, raw_cr)}
