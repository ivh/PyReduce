"""
Find clusters of pixels with signal
And combine them into continous orders
"""

import logging
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import morphology, label
from scipy.ndimage.filters import gaussian_filter1d, median_filter
from scipy.signal import peak_widths, find_peaks

# from PyReduce.cwrappers import find_clusters


def merge_clusters(
    img, orders, x, y, n_clusters, threshold=100, manual=True, plot=False
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

    n_row, n_col = img.shape

    x_poly = np.arange(n_col, dtype=int)
    y_poly = {i: np.polyval(order, x_poly) for i, order in orders.items()}

    # Calculate mean cluster thickness
    # TODO optimize
    mean_cluster_thickness = 10
    for cluster in n_clusters:
        # individual columns of this cluster
        columns = np.unique(y[cluster])

        delta = 0
        for col in columns:
            # thickness of the cluster in each column
            tmp = x[cluster][y[cluster] == col]
            delta += np.max(tmp) - np.min(tmp)

        mean_cluster_thickness += delta / len(columns)

    mean_cluster_thickness *= 1.5 / len(n_clusters)

    # determine cluster limits
    cluster_limit = {i: (np.min(y[i]), np.max(y[i])) for i in n_clusters}

    # TODO: Optimize this, quite slow for many clusters
    # for each pair of clusters
    # merge = [[cluster1, cluster2, overlap, overlap region limits[2]]]
    merge = []
    for i, j in combinations(n_clusters, 2):
        # Get cluster limits
        i_left, i_right = cluster_limit[i]
        j_left, j_right = cluster_limit[j]

        # Get polynomial points inside cluster limits for each
        y_ii = y_poly[i][i_left:i_right]
        y_ij = y_poly[i][j_left:j_right]
        y_jj = y_poly[j][j_left:j_right]
        y_ji = y_poly[j][i_left:i_right]

        # difference of polynomials within each cluster limit
        diff_i = np.abs(y_ii - y_ji)
        diff_j = np.abs(y_ij - y_jj)

        ind_i = np.where(
            (diff_i < mean_cluster_thickness) & (y_ji >= 0) & (y_ji < n_row)
        )
        ind_j = np.where(
            (diff_j < mean_cluster_thickness) & (y_ij >= 0) & (y_ij < n_row)
        )

        overlap = len(ind_i[0]) + len(ind_j[0])
        overlap_region = [-1, -1]
        if len(ind_i[0]) > 0:
            overlap_region[0] = np.min(ind_i[0]) + i_left
        if len(ind_j[0]) > 0:
            overlap_region[1] = np.max(ind_j[0]) + j_left

        if overlap == 0:
            continue
        merge += [[i, j, overlap, *overlap_region]]

    merge = np.array(merge)

    if len(merge) == 0:
        # No merging necessary
        return x, y, n_clusters

    # Cut small overlaps
    # TODO: convert overlap into percentage
    merge = merge[merge[:, 2] > threshold]
    # TODO fix merges, so that everycluster ends up in the right group
    # e.g. if more than 3 groups merge together, they should all end up in the same one
    merge = merge[np.argsort(merge[:, 2])]
    delete = []
    if manual:
        plt.ion()
    for before, after, overlap, region0, region1 in merge:
        if region1 - region0 > 0.9 * (
            cluster_limit[before][1] - cluster_limit[before][0]
        ) or region1 - region0 > 0.9 * (
            cluster_limit[after][1] - cluster_limit[after][0]
        ):
            # merge automatically
            answer = "y"
        elif manual:
            plot_order(before, after, x, y, x_poly, y_poly, img)

            while True:
                if manual:
                    answer = input("Merge? [y/n]")
                if answer in "ynrg":
                    break
        else:
            answer = "y"

        if answer == "n":
            pass
        elif answer == "y":
            logging.info("Merging orders %i and %i", before, after)
            y[after] = np.concatenate((y[after], y[before]))
            x[after] = np.concatenate((x[after], x[before]))
            delete += [before]
        elif answer == "r":
            delete += [before]
        elif answer == "g":
            delete += [after]

    if manual:
        plt.close()
        plt.ioff()

    delete = np.unique(delete)
    for d in delete:
        del x[d]
        del y[d]
        n_clusters = n_clusters[n_clusters != d]

    return x, y, n_clusters


def fit_polynomials_to_clusters(x, y, clusters, degree):
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
    orders = {c: np.polyfit(y[c], x[c], degree) for c in clusters}
    return orders


def plot_orders(im, x, y, clusters, orders, order_range):
    """ Plot orders and image """

    cluster_img = np.zeros_like(im)
    for c in clusters:
        cluster_img[x[c], y[c]] = c
    cluster_img = np.ma.masked_array(cluster_img, mask=cluster_img == 0)

    plt.subplot(211)
    bot, top = np.percentile(im, (1, 99))
    plt.imshow(im, origin="lower", vmin=bot, vmax=top)
    plt.title("Input Image + Order polynomials")
    plt.xlabel("x [pixel]")
    plt.ylabel("y [pixel]")

    if orders is not None:
        for i, order in enumerate(orders):
            x = np.arange(*order_range[i], 1)
            y = np.polyval(order, x)
            plt.plot(x, y)

    plt.subplot(212)
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
    plt.show()


def plot_order(i, j, x, y, x_poly, y_poly, img):
    """ Plot a single order """
    plt.clf()
    plt.imshow(img)
    plt.plot(x_poly, y_poly[i], "r")
    plt.plot(x_poly, y_poly[j], "g")
    plt.plot(y[i], x[i], "r.")
    plt.plot(y[j], x[j], "g.")

    n_row, n_col = img.shape
    xmin = min(np.min(x[i]), np.min(x[j])) - 50
    xmax = max(np.max(x[i]), np.max(x[j])) + 50

    ymin = min(np.min(y[i]), np.min(y[j])) - 50
    ymax = max(np.max(y[i]), np.max(y[j])) + 50

    plt.xlim([ymin, ymax])
    plt.ylim([xmin, xmax])

    plt.show()


def mark_orders(
    im,
    min_cluster=500,
    filter_size=120,
    noise=8,
    opower=4,
    border_width=5,
    plot=False,
    manual=True,
):
    """ Identify and trace orders

    Parameters
    ----------
    im : array[nrow, ncol]
        order definition image
    min_cluster : int, optional
        minimum cluster size in pixels (default: 500)
    filter_size : int, optional
        size of the running filter (default: 120)
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

    Returns
    -------
    orders : array[nord, opower+1]
        order tracing coefficients (in numpy order, i.e. largest exponent first)
    """

    # Convert to signed integer, to avoid underflow problems
    im = np.asarray(im)
    im = im.astype(np.int16)

    if filter_size is None:
        col = im[:, im.shape[0] // 2]
        col = median_filter(col, 5)
        threshold = np.percentile(col, 90)
        npeaks = find_peaks(col, height=threshold)[0].size
        filter_size = im.shape[0] // npeaks
        logging.info("Median filter size, estimated: %i", filter_size)
    elif filter_size <= 0:
        raise ValueError(f"Expected filter size > 0, but got {filter_size}")

    if border_width is None:
        # find width of orders, based on central column
        col = im[:, im.shape[0] // 2]
        col = median_filter(col, 5)
        idx = np.argmax(col)
        width = peak_widths(col, [idx])[0][0]
        border_width = int(np.ceil(width))
        logging.info("Image border width, estimated: %i", border_width)
    elif border_width < 0:
        raise ValueError(f"Expected border width > 0, but got {border_width}")

    if min_cluster is None:
        min_cluster = im.shape[1] // 4
        logging.info("Minimum cluster size, estimated: %i", min_cluster)
    elif not np.isscalar(min_cluster):
        raise TypeError(f"Expected scalar minimum cluster size, but got {min_cluster}")

    # blur image along columns, and use the median + blurred + noise as threshold
    blurred = gaussian_filter1d(im, filter_size, axis=0)

    if noise is None:
        tmp = np.abs(blurred.flatten())
        noise = np.percentile(tmp, 5)
        logging.info("Background noise, estimated: %f", noise)
    elif not np.isscalar(noise):
        raise TypeError(f"Expected scalar noise level, but got {noise}")

    threshold = np.ma.median(blurred - im, axis=0)
    mask = im > blurred + noise + np.abs(threshold)
    # remove borders
    if border_width != 0:
        mask[:border_width, :] = mask[-border_width:, :] = False
    # remove masked areas with no clusters
    mask = np.ma.filled(mask, fill_value=False)
    # close gaps inbetween clusters
    struct = np.full((5, 5), 1)
    mask = morphology.binary_closing(mask, struct, border_value=1)
    # remove small lonely clusters
    # struct = morphology.generate_binary_structure(2, 1)
    mask = morphology.binary_opening(mask, struct)

    # label clusters
    clusters, _ = label(mask)

    # remove small clusters
    sizes = np.bincount(clusters.ravel())
    mask_sizes = sizes > min_cluster
    mask_sizes[0] = True  # This is the background, which we don't need to remove
    for i in np.arange(len(sizes))[~mask_sizes]:
        clusters[clusters == i] = 0

    if plot:
        plt.title("Identified clusters")
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        _clusters = np.ma.masked_array(clusters, mask=clusters == 0)
        plt.imshow(_clusters, origin="lower", cmap="prism")
        plt.show()

    # # Reorganize x, y, clusters into a more convenient "pythonic" format
    # # x, y become dictionaries, with an entry for each order
    # # n is just a list of all orders (ignore cluster == 0)
    n = np.unique(clusters)
    n = n[n != 0]
    x = {i: np.where(clusters == c)[0] for i, c in enumerate(n)}
    y = {i: np.where(clusters == c)[1] for i, c in enumerate(n)}
    n = np.arange(len(n))

    # if plot:
    #     plot_orders(im, x, y, n, None, None)

    # fit polynomials
    orders = fit_polynomials_to_clusters(x, y, n, 2)

    # Merge clusters, if there are even any possible mergers left
    x, y, n = merge_clusters(im, orders, x, y, n, plot=plot, manual=manual)

    orders = fit_polynomials_to_clusters(x, y, n, opower)

    # sort orders from bottom to top, using mean coordinate
    # TODO: better metric for position?
    key = np.array([[i, np.mean(x[i])] for i in n])
    key = key[np.argsort(key[:, 1]), 0]
    n = np.arange(len(n), dtype=int)
    x = {c: x[key[c]] for c in n}
    y = {c: y[key[c]] for c in n}
    orders = np.array([orders[key[c]] for c in n])

    column_range = np.array([[np.min(y[i]), np.max(y[i]) + 1] for i in n])

    if plot:
        plot_orders(im, x, y, n, orders, column_range)

    return orders, column_range
