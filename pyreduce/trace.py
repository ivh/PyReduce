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
    orders : dict(int, array(float))
        coefficients of polynomial fits to clusters
    x : dict(int, array(int))
        x coordinates of cluster points
    y : dict(int, array(int))
        y coordinates of cluster points
    n_clusters : array(int)
        cluster numbers
    threshold : int, optional
        overlap threshold for merging clusters (the default is 100)
    manual : bool, optional
        if True ask before merging orders

    Returns
    -------
    x : dict(int: array)
        x coordinates of clusters, key=cluster id
    y : dict(int: array)
        y coordinates of clusters, key=cluster id
    n_clusters : int
        number of identified clusters
    """

    nrow, ncol = img.shape
    mct = calculate_mean_cluster_thickness(x, y)

    merge = create_merge_array(x, y, mct, nrow, ncol, deg, merge_min_threshold)

    if manual:
        plt.ion()

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
            plot_order(i, j, x, y, img, deg, title=title)
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
    orders : dict(int, array[degree+1])
        coefficients of polynomial fit for each cluster
    """

    orders = {c: fit(x[c], y[c], degree, regularization) for c in clusters}
    return orders


def plot_orders(im, x, y, clusters, orders, order_range, title=None):
    """Plot orders and image"""

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
    plt.title("Input Image + Order polynomials")
    plt.xlabel("x [pixel]")
    plt.ylabel("y [pixel]")
    plt.ylim([0, im.shape[0]])

    if orders is not None:
        for i, order in enumerate(orders):
            x = np.arange(*order_range[i], 1)
            y = np.polyval(order, x)
            plt.plot(x, y)

    plt.subplot(122)
    plt.imshow(cluster_img, cmap=plt.get_cmap("tab20"), origin="upper")
    plt.title("Detected Clusters + Order Polynomials")
    plt.xlabel("x [pixel]")
    plt.ylabel("y [pixel]")

    if orders is not None:
        for i, order in enumerate(orders):
            x = np.arange(*order_range[i], 1)
            y = np.polyval(order, x)
            plt.plot(x, y)

    plt.ylim([0, im.shape[0]])
    if title is not None:
        plt.suptitle(title)
    util.show_or_save("orders_trace")


def plot_order(i, j, x, y, img, deg, title=""):
    """Plot a single order"""
    _, ncol = img.shape

    order_i = fit(x[i], y[i], deg)
    order_j = fit(x[j], y[j], deg)

    xp = np.arange(ncol)
    yi = np.polyval(order_i, xp)
    yj = np.polyval(order_j, xp)

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
    util.show_or_save(f"orders_single_{i}_{j}")


def trace(
    im,
    min_cluster=None,
    min_width=None,
    filter_x=0,
    filter_y=None,
    filter_type="boxcar",
    noise=None,
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
        noise to filter out (default: 8)
    opower : int, optional
        polynomial degree of the order fit (default: 4)
    border_width : int, optional
        number of pixels at the bottom and top borders of the image to ignore for order tracing (default: 5)
    plot : bool, optional
        wether to plot the final order fits (default: False)
    manual : bool, optional
        wether to manually select clusters to merge (strongly recommended) (default: True)
    debug_dir : str, optional
        if set, write intermediate images (filtered, background, mask) to this directory

    Returns
    -------
    orders : array[nord, opower+1]
        order tracing coefficients (in numpy order, i.e. largest exponent first)
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
    elif border_width < 0:
        raise ValueError(f"Expected border width > 0, but got {border_width}")

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

    if noise is None:
        tmp = np.abs(background.flatten())
        noise = np.percentile(tmp, 5)
        logger.info("Background noise, estimated: %f", noise)
    elif not np.isscalar(noise):
        raise TypeError(f"Expected scalar noise level, but got {noise}")

    # Threshold: pixels above local background are signal
    mask = im_clean > background + noise
    mask_initial = mask.copy()
    # remove borders
    if border_width != 0:
        mask[:border_width, :] = mask[-border_width:, :] = False
        mask[:, :border_width] = mask[:, -border_width:] = False
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
    # # x, y become dictionaries, with an entry for each order
    # # n is just a list of all orders (ignore cluster == 0)
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
        util.show_or_save("orders_clusters")

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

    orders = fit_polynomials_to_clusters(x, y, n, degree)

    # sort orders from bottom to top, using relative position

    def compare(i, j):
        _, xi, i_left, i_right = i
        _, xj, j_left, j_right = j

        if i_right < j_left or j_right < i_left:
            return xi.mean() - xj.mean()

        left = max(i_left, j_left)
        right = min(i_right, j_right)

        return xi[left:right].mean() - xj[left:right].mean()

    xp = np.arange(im.shape[1])
    keys = [(c, np.polyval(orders[c], xp), y[c].min(), y[c].max()) for c in x.keys()]
    keys = sorted(keys, key=cmp_to_key(compare))
    key = [k[0] for k in keys]

    n = np.arange(len(n), dtype=int)
    x = {c: x[key[c]] for c in n}
    y = {c: y[key[c]] for c in n}
    orders = np.array([orders[key[c]] for c in n])

    column_range = np.array([[np.min(y[i]), np.max(y[i]) + 1] for i in n])

    if plot:  # pragma: no cover
        plot_orders(im, x, y, n, orders, column_range, title=plot_title)

    return orders, column_range


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
