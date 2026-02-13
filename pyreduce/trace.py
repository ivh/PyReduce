"""
Find clusters of pixels with signal and fit polynomial traces.

Note on terminology:
- "trace": A single polynomial fit to a cluster of pixels (e.g., one fiber)
- "spectral order": A group of traces at similar wavelengths (e.g., all fibers in one echelle order)

The main function `trace()` detects and fits individual traces, returning Trace objects.
Use `group_fibers()` to merge traces into fiber groups according to instrument config.
"""

import logging
import re
from functools import cmp_to_key
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


def _natural_sort_key(s):
    """Sort key for natural ordering (e.g., bundle_2 before bundle_10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from numpy.polynomial.polynomial import Polynomial
from scipy.ndimage import binary_closing, binary_opening, label
from scipy.ndimage.filters import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from . import util
from .trace_model import Trace as TraceData

logger = logging.getLogger(__name__)


def _find_beam_pairs_dp(y_positions, fibers_per_order):
    """Find optimal pairing of traces into beam pairs using gap analysis.

    For dual-beam instruments (fibers_per_order=2), beam pairs have smaller
    gaps than inter-order gaps. This function uses dynamic programming to
    find the pairing that maximizes the number of paired traces while only
    allowing pairs whose gap is below an automatically computed threshold.

    Parameters
    ----------
    y_positions : array
        Sorted y-positions of traces at detector center.
    fibers_per_order : int
        Number of fibers (beams) per order (typically 2).

    Returns
    -------
    list of tuples
        Each tuple contains `fibers_per_order` trace indices forming one order.
    """
    n = len(y_positions)
    if n < fibers_per_order:
        return []

    # Only implemented for fibers_per_order=2
    if fibers_per_order != 2:
        # Fallback to simple sequential grouping
        groups = []
        for i in range(0, n - n % fibers_per_order, fibers_per_order):
            groups.append(tuple(range(i, i + fibers_per_order)))
        return groups

    gaps = np.diff(y_positions)

    # Find threshold separating beam-pair gaps from inter-order gaps.
    # Use Otsu's method: find the threshold that minimizes the weighted
    # intra-class variance of the two groups (beam-pair vs inter-order).
    sorted_gaps = np.sort(gaps)
    best_threshold = np.median(sorted_gaps)
    best_variance = np.inf

    for i in range(1, len(sorted_gaps)):
        if sorted_gaps[i] == sorted_gaps[i - 1]:
            continue
        candidate = (sorted_gaps[i - 1] + sorted_gaps[i]) / 2
        lo_group = sorted_gaps[:i]
        hi_group = sorted_gaps[i:]
        w0 = len(lo_group) / len(sorted_gaps)
        w1 = len(hi_group) / len(sorted_gaps)
        weighted_var = w0 * np.var(lo_group) + w1 * np.var(hi_group)
        if weighted_var < best_variance:
            best_variance = weighted_var
            best_threshold = candidate

    threshold = best_threshold

    n_beam = int(np.sum(gaps <= threshold))
    n_inter = int(np.sum(gaps > threshold))
    logger.info(
        "Beam-pair gap threshold: %.1f px (%d beam gaps, %d inter-order gaps)",
        threshold,
        n_beam,
        n_inter,
    )

    # DP: find maximum number of paired traces using only small-gap pairs.
    # dp[i] = (num_paired, total_gap) for traces 0..i-1
    # At each position, either skip a trace or pair it with the previous one.
    dp_count = np.zeros(n + 1, dtype=int)
    dp_gap = np.zeros(n + 1)
    choice = np.zeros(n + 1, dtype=int)  # 0=skip, 1=pair

    for i in range(2, n + 1):
        # Option 1: skip trace i-1
        dp_count[i] = dp_count[i - 1]
        dp_gap[i] = dp_gap[i - 1]
        choice[i] = 0

        # Option 2: pair traces i-2 and i-1
        if gaps[i - 2] <= threshold:
            new_count = dp_count[i - 2] + 2
            new_gap = dp_gap[i - 2] + gaps[i - 2]
            if new_count > dp_count[i] or (
                new_count == dp_count[i] and new_gap < dp_gap[i]
            ):
                dp_count[i] = new_count
                dp_gap[i] = new_gap
                choice[i] = 1

    # Backtrack to find pairs
    pairs = []
    i = n
    while i >= 2:
        if choice[i] == 1:
            pairs.append((i - 2, i - 1))
            i -= 2
        else:
            i -= 1
    pairs.reverse()

    logger.info(
        "DP pairing: %d pairs from %d traces (%d unpaired)",
        len(pairs),
        n,
        n - 2 * len(pairs),
    )
    return pairs


def _assign_order_and_fiber_inplace(
    traces: list,
    order_centers: dict[int, float] | None,
    ncol: int,
    fibers_per_order: int | None = None,
) -> None:
    """Assign m (order number) and fiber_idx to Trace objects.

    Parameters
    ----------
    traces : list[Trace]
        Trace objects (modified in place)
    order_centers : dict[int, float] | None
        Order number -> y-position mapping. If None, m stays None
        (unless fibers_per_order is set for auto-pairing).
    ncol : int
        Number of columns in detector
    fibers_per_order : int or None
        If set and order_centers is None, group every N consecutive traces
        (sorted by y-position) into the same order with sequential fiber_idx.
    """
    if not traces:
        return

    x_center = ncol // 2
    y_positions = [t.y_at_x(x_center) for t in traces]

    if order_centers is not None:
        # Match traces to known order centers
        order_nums = np.array(list(order_centers.keys()))
        y_centers = np.array([order_centers[m] for m in order_nums])

        for i, y in enumerate(y_positions):
            distances = np.abs(y - y_centers)
            closest_idx = np.argmin(distances)
            traces[i].m = int(order_nums[closest_idx])

        # Group by m to assign fiber_idx within each order
        from collections import defaultdict

        traces_by_m = defaultdict(list)
        for i, t in enumerate(traces):
            traces_by_m[t.m].append((i, y_positions[i]))

        # Assign fiber_idx: sort by y within each order, then 1, 2, 3...
        for _m, trace_list in traces_by_m.items():
            trace_list.sort(key=lambda x: x[1])
            for fiber_idx, (trace_idx, _y) in enumerate(trace_list, start=1):
                traces[trace_idx].fiber_idx = fiber_idx

    elif fibers_per_order is not None and fibers_per_order > 1:
        # Auto-pair traces using gap analysis (beam-pair gaps are smaller
        # than inter-order gaps). Uses DP to find optimal pairing.
        traces.sort(key=lambda t: t.y_at_x(x_center))
        y_pos = np.array([t.y_at_x(x_center) for t in traces])
        pair_indices = _find_beam_pairs_dp(y_pos, fibers_per_order)

        # Assign m and fiber_idx from DP result
        order_num = 0
        paired_set = set()
        for group in pair_indices:
            for fiber_idx, trace_idx in enumerate(group, start=1):
                traces[trace_idx].m = order_num
                traces[trace_idx].fiber_idx = fiber_idx
                paired_set.add(trace_idx)
            order_num += 1

        # Drop unpaired traces
        n_dropped = len(traces) - len(paired_set)
        if n_dropped > 0:
            logger.warning(
                "%d traces could not be paired and will be dropped", n_dropped
            )
        traces[:] = [t for i, t in enumerate(traces) if i in paired_set]

        logger.info(
            "Auto-paired %d traces into %d orders (fibers_per_order=%d)",
            len(traces),
            order_num,
            fibers_per_order,
        )
    else:
        # No order_centers and no fibers_per_order: assign sequential m and
        # fiber_idx=1 to all traces (one fiber per order).
        traces.sort(key=lambda t: t.y_at_x(x_center))
        for i, t in enumerate(traces):
            t.m = i
            t.fiber_idx = 1

    # Sort traces by (m descending, fiber_idx)
    def sort_key(t):
        m_val = t.m if t.m is not None else float("inf")
        return (-m_val, t.fiber_idx or 0)

    traces.sort(key=sort_key)

    logger.info("Assigned order/fiber to %d traces", len(traces))
    if order_centers is not None:
        unique_m = {t.m for t in traces if t.m is not None}
        logger.info("  Order numbers (m): %s", sorted(unique_m, reverse=True))


def _compute_heights_inplace(traces: list, ncol: int) -> None:
    """Compute and set extraction heights on Trace objects based on neighbor distances.

    For each trace, measures the distance to neighbors at multiple reference
    columns (0.1, 0.2, ..., 0.9 of detector width) and uses the maximum.

    Parameters
    ----------
    traces : list[Trace]
        Trace objects (modified in place to set height attribute)
    ncol : int
        Number of columns in detector
    """
    ntrace = len(traces)
    if ntrace == 0:
        return

    if ntrace == 1:
        # Single trace: no neighbors, leave height as None
        return

    # Reference columns at 0.1, 0.2, ..., 0.9 of detector width
    ref_fractions = np.linspace(0.1, 0.9, 9)
    ref_cols = (ref_fractions * ncol).astype(int)

    for i, t in enumerate(traces):
        # Determine which reference columns are within this trace's range
        valid_cols = ref_cols[
            (ref_cols >= t.column_range[0]) & (ref_cols < t.column_range[1])
        ]
        if len(valid_cols) == 0:
            valid_cols = [(t.column_range[0] + t.column_range[1]) // 2]

        max_height = 0.0

        for x in valid_cols:
            y_i = t.y_at_x(x)

            if i == 0:
                y_next = traces[i + 1].y_at_x(x)
                height = abs(y_next - y_i)
            elif i == ntrace - 1:
                y_prev = traces[i - 1].y_at_x(x)
                height = abs(y_i - y_prev)
            else:
                y_prev = traces[i - 1].y_at_x(x)
                y_next = traces[i + 1].y_at_x(x)
                height = (y_next - y_prev) / 2

            max_height = max(max_height, height)

        t.height = max_height


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

    # Skip all merge computation when merging is disabled
    if auto_merge_threshold == 1 and not manual:
        n_clusters = list(x.keys())
        return x, y, n_clusters

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


def plot_traces(im, traces, title=None):
    """Plot traces and image.

    Parameters
    ----------
    im : ndarray
        Input image
    traces : list[Trace]
        Trace objects to plot
    title : str, optional
        Plot title
    """
    plt.figure()

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

    for t in traces:
        x = np.arange(*t.column_range, 1)
        y = t.y_at_x(x)
        plt.plot(x, y)

    plt.subplot(122)
    plt.imshow(plot_im, origin="lower", vmin=bot, vmax=top)
    plt.title("Trace Polynomials")
    plt.xlabel("x [pixel]")
    plt.ylabel("y [pixel]")

    for t in traces:
        x = np.arange(*t.column_range, 1)
        y = t.y_at_x(x)
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
    order_centers: dict[int, float] | None = None,
    fibers_per_order: int | None = None,
):
    """Identify and trace orders, returning Trace objects.

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
    degree : int, optional
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
    order_centers : dict[int, float], optional
        Mapping of order number (m) -> y-position at detector center. If provided,
        traces are assigned m values by matching to these centers. Otherwise,
        m remains None (to be assigned later from wavecal obase).
    fibers_per_order : int, optional
        Number of fiber traces per spectral order. When set and order_centers is None,
        consecutive traces (sorted by y) are grouped into orders of this size.
        Used for instruments like HARPSpol where a Wollaston prism splits each
        order into multiple beams.

    Returns
    -------
    list[Trace]
        Trace objects with:
        - m: assigned from order_centers if provided, else None
        - fiber_idx: 1 for single-fiber, or sequential within each order for multi-fiber
        - group: None (not yet grouped)
        - pos: polynomial coefficients
        - column_range: valid column range
        - height: computed from neighbor distances (None for single trace)
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

    # Plot cross-section of signal vs background at plot level 2
    if plot >= 2:  # pragma: no cover
        ncol = im_clean.shape[1]
        mid = ncol // 2
        cols = slice(mid - 25, mid + 25)
        signal_profile = np.median(im_clean[:, cols], axis=1)
        background_profile = np.median(background[:, cols], axis=1)
        threshold_profile = background_profile * (1 + noise_relative) + noise

        plt.figure()
        plt.plot(signal_profile, label="signal (median of 50 middle cols)")
        plt.plot(background_profile, label=f"smoothed ({filter_type}={filter_y})")
        plt.plot(
            threshold_profile, label=f"threshold (noise={noise}, rel={noise_relative})"
        )
        plt.xlabel("Row (cross-dispersion)")
        plt.ylabel("Counts")
        title = "Signal vs background profile"
        if plot_title is not None:
            title = f"{plot_title}\n{title}"
        plt.title(title)
        plt.legend()
        util.show_or_save("trace_signal_vs_background")
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
    n_initial = clusters.max()
    logger.info("Found %d clusters initially", n_initial)

    # remove small clusters
    sizes = np.bincount(clusters.ravel())
    mask_sizes = sizes > min_cluster
    mask_sizes[0] = True  # This is the background, which we don't need to remove
    n_too_small = np.sum(~mask_sizes) - 1  # -1 for background
    clusters[~mask_sizes[clusters]] = 0

    # # Reorganize x, y, clusters into a more convenient "pythonic" format
    # # x, y become dictionaries, with an entry for each cluster
    # # n is just a list of all cluster labels (ignore cluster == 0)
    n = np.unique(clusters)
    n = n[n != 0]
    x = {i: np.where(clusters == c)[0] for i, c in enumerate(n)}
    y = {i: np.where(clusters == c)[1] for i, c in enumerate(n)}
    if n_too_small > 0:
        logger.info(
            "Removed %d clusters smaller than min_cluster=%d, %d remain",
            n_too_small,
            min_cluster,
            len(n),
        )

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
        n_before_sigma = len(x)
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
        n_after_sigma = len(x)
        if n_after_sigma != n_before_sigma:
            logger.info(
                "Sigma clipping: %d -> %d clusters", n_before_sigma, n_after_sigma
            )

    if plot:  # pragma: no cover
        plt.figure()
        title = "Identified clusters"
        if plot_title is not None:
            title = f"{plot_title}\n{title}"
        plt.title(title)
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")

        # Sort clusters by mean y-position so we can assign alternating colors
        sorted_clusters = sorted(x.keys(), key=lambda i: np.mean(x[i]))
        # Use distinct colors that cycle, so adjacent clusters are visually distinct
        distinct_colors = [
            "#e41a1c",
            "#377eb8",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#a65628",
        ]
        from matplotlib.colors import ListedColormap

        n_colors = len(distinct_colors)

        clusters = np.ma.zeros(im.shape, dtype=int)
        for color_idx, i in enumerate(sorted_clusters):
            clusters[x[i], y[i]] = (color_idx % n_colors) + 1
        clusters[clusters == 0] = np.ma.masked

        cmap = ListedColormap(distinct_colors)
        plt.imshow(clusters, origin="lower", cmap=cmap, vmin=1, vmax=n_colors)
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
        n_before_width = len(x)
        sizes = {k: v.max() - v.min() for k, v in y.items()}
        mask_sizes = {k: v > min_width for k, v in sizes.items()}
        for k, v in mask_sizes.items():
            if not v:
                del x[k]
                del y[k]
        n = x.keys()
        n_too_narrow = n_before_width - len(x)
        if n_too_narrow > 0:
            logger.info(
                "Removed %d clusters narrower than min_width=%d, %d remain",
                n_too_narrow,
                min_width,
                len(x),
            )

    logger.info("Fitting polynomials to %d clusters", len(x))
    traces_dict = fit_polynomials_to_clusters(x, y, n, degree)

    # Sort traces from bottom to top, using relative position
    def compare(i, j):
        _, xi, i_left, i_right = i
        _, xj, j_left, j_right = j

        if i_right < j_left or j_right < i_left:
            return xi.mean() - xj.mean()

        left = max(i_left, j_left)
        right = min(i_right, j_right)

        return xi[left:right].mean() - xj[left:right].mean()

    xp = np.arange(im.shape[1])
    keys = [
        (c, np.polyval(traces_dict[c], xp), y[c].min(), y[c].max()) for c in x.keys()
    ]
    keys = sorted(keys, key=cmp_to_key(compare))

    # Create Trace objects in sorted order
    trace_objects = []
    for cluster_id, _, _, _ in keys:
        pos = traces_dict[cluster_id]
        cr = (int(y[cluster_id].min()), int(y[cluster_id].max()) + 1)
        trace_objects.append(TraceData(m=None, pos=pos, column_range=cr))

    # Compute extraction heights based on trace spacing
    _compute_heights_inplace(trace_objects, im.shape[1])

    # Assign order numbers and fiber indices
    _assign_order_and_fiber_inplace(
        trace_objects, order_centers, im.shape[1], fibers_per_order=fibers_per_order
    )

    if plot:  # pragma: no cover
        plot_traces(im, trace_objects, title=plot_title)

    return trace_objects


def select_traces_for_step(
    traces: list[TraceData],
    fibers_config,
    step_name: str,
) -> dict[str, list[TraceData]]:
    """Select which traces to use for a given reduction step.

    Looks up fibers_config.use[step_name] to determine selection mode.

    Parameters
    ----------
    traces : list[Trace]
        All trace objects
    fibers_config : FibersConfig or None
        Fiber configuration (may be None for single-fiber instruments)
    step_name : str
        Name of the reduction step (e.g., "science", "curvature")

    Returns
    -------
    selected : dict[str, list[Trace]]
        {group_name: [traces]} for each selected group
    """
    if not traces:
        return {}

    # No fiber config means use all traces (single-fiber instrument)
    if fibers_config is None:
        return {"all": traces}

    # No groups/bundles defined means use all traces
    if fibers_config.groups is None and fibers_config.bundles is None:
        return {"all": traces}

    # Determine selection for this step from config
    if fibers_config.use is not None:
        selection = fibers_config.use.get(
            step_name, fibers_config.use.get("default", "groups")
        )
    else:
        selection = "groups"

    if selection == "groups":
        # Return all traces that have explicit group assignment
        grouped = [t for t in traces if t.group is not None]
        if grouped:
            return {"all": grouped}
        # No grouped traces, return all
        return {"all": traces}

    elif selection == "per_fiber":
        # Return traces grouped by fiber_idx for per-fiber processing
        result = {}
        fiber_indices = {t.fiber_idx for t in traces if t.fiber_idx is not None}
        if not fiber_indices:
            logger.warning("No fiber_idx set on traces, using all traces")
            return {"all": traces}
        for idx in sorted(fiber_indices):
            idx_traces = [t for t in traces if t.fiber_idx == idx]
            idx_traces.sort(key=lambda t: (t.m if t.m is not None else 0))
            result[f"fiber_{idx}"] = idx_traces
        return result

    elif isinstance(selection, list):
        # Select specific groups by name - keep them separate
        result = {}
        for name in selection:
            # Match by group (compare as string, skip ungrouped traces)
            selected = [
                t for t in traces if t.group is not None and str(t.group) == name
            ]
            if not selected:
                logger.warning("Group '%s' not found in trace data", name)
                continue
            # Sort by m (order number) for consistent ordering
            selected.sort(key=lambda t: (t.m if t.m is not None else 0))
            result[name] = selected

        if not result:
            logger.warning("No valid groups selected, using all traces")
            return {"all": traces}
        return result

    else:
        logger.warning("Unknown selection type: %s, using all traces", selection)
        return {"all": traces}


def group_fibers(
    traces: list[TraceData],
    fibers_config,
    degree: int = 4,
) -> list[TraceData]:
    """Merge individual fiber traces into groups according to config.

    Takes traces with fiber_idx set (individual fibers) and returns NEW traces
    with group set (merged/grouped result). The input traces are not modified.

    Parameters
    ----------
    traces : list[Trace]
        Individual fiber traces with fiber_idx set and group=None.
        Can have m set (from order_centers) or None (to be assigned later).
    fibers_config : FibersConfig
        Configuration specifying groups or bundles.
    degree : int, optional
        Polynomial degree for refitted traces (used with "average" merge).

    Returns
    -------
    list[Trace]
        New Trace objects with:
        - group: set to group name (e.g., "A", "B", "bundle_1")
        - fiber_idx: None (merged, no individual identity)
        - m: preserved from input traces
        - pos: new polynomial from merging
        - column_range: intersection of member traces
        - height: from config or computed from member traces

        Returns empty list if no grouping config is provided.
    """
    if not traces:
        return []

    if fibers_config is None:
        return []

    if fibers_config.groups is None and fibers_config.bundles is None:
        return []

    # Group input traces by order number (m)
    from collections import defaultdict

    traces_by_m = defaultdict(list)
    for t in traces:
        traces_by_m[t.m].append(t)

    # Sort within each order by fiber_idx
    for m in traces_by_m:
        traces_by_m[m].sort(key=lambda t: t.fiber_idx if t.fiber_idx else 0)

    result = []

    if fibers_config.groups is not None:
        # Named groups with explicit ranges
        for group_name, group_cfg in fibers_config.groups.items():
            start, end = group_cfg.range
            start_idx = start - 1  # Convert to 0-based
            end_idx = end - 1  # Half-open, end is exclusive

            for m, order_traces in sorted(traces_by_m.items()):
                # Select fibers in range for this order
                if end_idx > len(order_traces):
                    logger.warning(
                        "Group %s range [%d, %d) exceeds fiber count %d in order %s",
                        group_name,
                        start,
                        end,
                        len(order_traces),
                        m,
                    )
                    end_idx = min(end_idx, len(order_traces))
                start_idx = max(start_idx, 0)

                if start_idx >= end_idx:
                    continue

                selected = order_traces[start_idx:end_idx]
                merged = _merge_trace_objects(selected, group_cfg.merge, degree)

                if merged is not None:
                    # Compute height from group config
                    height = _compute_group_height_from_traces(selected, group_cfg)

                    result.append(
                        TraceData(
                            m=m,
                            group=group_name,
                            fiber_idx=None,
                            pos=merged.pos,
                            column_range=merged.column_range,
                            height=height,
                        )
                    )

    elif fibers_config.bundles is not None:
        bundle_cfg = fibers_config.bundles
        bundle_size = bundle_cfg.size

        for m, order_traces in sorted(traces_by_m.items()):
            n_in_order = len(order_traces)
            if n_in_order == 0:
                continue

            # Check divisibility
            if n_in_order % bundle_size != 0:
                logger.warning(
                    "Order %s has %d fibers, not divisible by bundle size %d",
                    m,
                    n_in_order,
                    bundle_size,
                )

            n_bundles = (n_in_order + bundle_size - 1) // bundle_size
            for i in range(n_bundles):
                bundle_name = f"bundle_{i + 1}"
                start_idx = i * bundle_size
                end_idx = min((i + 1) * bundle_size, n_in_order)

                selected = order_traces[start_idx:end_idx]
                merged = _merge_trace_objects(selected, bundle_cfg.merge, degree)

                if merged is not None:
                    height = _compute_group_height_from_traces(selected, bundle_cfg)

                    result.append(
                        TraceData(
                            m=m,
                            group=bundle_name,
                            fiber_idx=None,
                            pos=merged.pos,
                            column_range=merged.column_range,
                            height=height,
                        )
                    )

    # Sort by (m descending, group)
    def sort_key(t):
        m = t.m if t.m is not None else float("inf")
        return (-m, str(t.group) if t.group else "")

    result.sort(key=sort_key)

    logger.info("Grouped %d fibers into %d traces", len(traces), len(result))
    return result


def _merge_trace_objects(
    traces: list[TraceData],
    merge_method: str | list[int],
    degree: int,
) -> TraceData | None:
    """Merge multiple traces into one according to merge method.

    Parameters
    ----------
    traces : list[Trace]
        Traces to merge (must have same m)
    merge_method : str or list[int]
        "average", "center", or list of 1-based indices
    degree : int
        Polynomial degree for refitting

    Returns
    -------
    Trace or None
        Merged trace (m and group not set), or None if no valid traces
    """
    if not traces:
        return None

    n = len(traces)

    # Find shared column range
    col_min = max(t.column_range[0] for t in traces)
    col_max = min(t.column_range[1] for t in traces)
    if col_min >= col_max:
        col_min = min(t.column_range[0] for t in traces)
        col_max = max(t.column_range[1] for t in traces)

    if merge_method == "center":
        idx = n // 2
        return TraceData(
            m=traces[idx].m,
            pos=traces[idx].pos,
            column_range=(col_min, col_max),
        )

    elif merge_method == "average":
        if col_min >= col_max:
            idx = n // 2
            return TraceData(
                m=traces[idx].m,
                pos=traces[idx].pos,
                column_range=traces[idx].column_range,
            )

        x_eval = np.arange(col_min, col_max)
        y_values = np.array([t.y_at_x(x_eval) for t in traces])
        y_mean = np.mean(y_values, axis=0)

        fit = Polynomial.fit(x_eval, y_mean, deg=degree, domain=[])
        coeffs = fit.coef[::-1]

        return TraceData(
            m=traces[0].m,
            pos=coeffs,
            column_range=(col_min, col_max),
        )

    elif isinstance(merge_method, list):
        indices = [i - 1 for i in merge_method]
        valid = [i for i in indices if 0 <= i < n]
        if not valid:
            logger.warning("No valid indices in merge method %s", merge_method)
            return None
        if len(valid) == 1:
            idx = valid[0]
            return TraceData(
                m=traces[idx].m,
                pos=traces[idx].pos,
                column_range=(col_min, col_max),
            )
        # Multiple indices: average them
        x_eval = np.arange(col_min, col_max)
        y_values = np.array([traces[i].y_at_x(x_eval) for i in valid])
        y_mean = np.mean(y_values, axis=0)

        fit = Polynomial.fit(x_eval, y_mean, deg=degree, domain=[])
        coeffs = fit.coef[::-1]

        return TraceData(
            m=traces[0].m,
            pos=coeffs,
            column_range=(col_min, col_max),
        )

    else:
        raise ValueError(f"Unknown merge method: {merge_method}")


def _compute_group_height_from_traces(
    traces: list[TraceData],
    config,
) -> float | None:
    """Compute extraction height for a group of traces.

    Parameters
    ----------
    traces : list[Trace]
        Traces in this group
    config : FiberGroupConfig or FiberBundleConfig
        Config with height specification

    Returns
    -------
    float or None
        Height in pixels, or None to use default
    """
    if config.height is None:
        return None

    if config.height == "derived":
        n = len(traces)
        if n < 2:
            return None

        # Get y-positions at column center
        col_min = max(t.column_range[0] for t in traces)
        col_max = min(t.column_range[1] for t in traces)
        x_center = (col_min + col_max) // 2

        y_positions = sorted([t.y_at_x(x_center) for t in traces])
        spacings = np.diff(y_positions)
        fiber_diameter = np.median(spacings) if len(spacings) > 0 else 0

        span = y_positions[-1] - y_positions[0]
        return span + fiber_diameter

    return float(config.height)


def _verify_trace_ordering(traces: list) -> None:
    """Verify that traces are ordered correctly: m decreases as y increases.

    For echelle spectrographs, higher spectral orders (larger m) have shorter
    wavelengths and appear lower on the detector (smaller y). So as we go up
    in y, m should decrease.

    Parameters
    ----------
    traces : list[Trace]
        Trace objects to verify

    Raises
    ------
    ValueError
        If traces are not ordered correctly
    """
    if len(traces) < 2:
        return

    # Get y-position at detector center for each trace
    # Use the middle of the common column range
    x_mid = np.median([np.mean(t.column_range) for t in traces])
    y_positions = [t.y_at_x(x_mid) for t in traces]
    m_values = [t.m for t in traces]

    # Sort by y to check ordering
    sorted_pairs = sorted(zip(y_positions, m_values, strict=False))

    # Check that m decreases (or stays same for multi-fiber) as y increases
    prev_m = sorted_pairs[0][1]
    for y, m in sorted_pairs[1:]:
        if m is not None and prev_m is not None and m > prev_m:
            logger.warning(
                "Trace ordering may be incorrect: m=%d at y=%.1f is higher than m=%d below it. "
                "Expected m to decrease as y increases.",
                m,
                y,
                prev_m,
            )
            break
        if m is not None:
            prev_m = m
