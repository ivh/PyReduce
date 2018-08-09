"""
Find clusters of pixels with signal
And combine them into continous orders
"""

import logging
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from cwrappers import find_clusters


def merge_clusters(
    shape, orders, x, y, n_clusters, threshold=100, manual=False, plot=False
):
    """Merge clusters that belong together
    
    Parameters
    ----------
    shape : tuple(int, int)
        shape of the image, (row, column)
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

    n_row, n_col = shape

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

    # for each pair of clusters
    # merge = [[overlap, other cluster, overlap region limits[2]]]
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

    # Cut small overlaps
    # TODO: convert overlap into percentage
    merge = merge[merge[:, 2] > threshold]
    # TODO fix merges, so that everycluster ends up in the right group
    # e.g. if more than 3 groups merge together, they should all end up in the same one
    merge = merge[np.argsort(merge[:, 0])]
    delete = []
    if manual:
        plt.ion()
    for before, after, overlap, _, _ in merge:
        if plot or manual:
            plot_order(before, after, x, y, x_poly, y_poly, shape)

        while True:
            if manual:
                answer = input("Merge? [y/n]")
            else:
                answer = "y"

            if answer == "n":
                break
            elif answer == "y":
                logging.info("Merging orders %i and %i", before, after)
                y[after] = np.concatenate((y[after], y[before]))
                x[after] = np.concatenate((x[after], x[before]))
                delete += [before]
                break
            elif answer == "r":
                delete += [before]
                break
            elif answer == "g":
                delete += [after]
                break
            else:
                print("Choose one")

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

    plt.subplot(121)
    plt.imshow(im, origin="lower")
    plt.title("Input")

    plt.subplot(122)
    plt.imshow(cluster_img, cmap=plt.get_cmap("tab20"), origin="upper")
    plt.title("Clusters")

    for i, order in enumerate(orders):
        x = np.arange(*order_range[i], 1)
        y = np.polyval(order, x)
        plt.plot(x, y)

    plt.ylim([0, im.shape[0]])
    plt.show()


def plot_order(i, j, x, y, x_poly, y_poly, shape):
    """ Plot a single order """
    plt.clf()
    plt.plot(x_poly, y_poly[i], "r")
    plt.plot(x_poly, y_poly[j], "g")
    plt.plot(y[i], x[i], "r.")
    plt.plot(y[j], x[j], "g.")

    n_row, n_col = shape
    plt.xlim([0, n_col])
    plt.ylim([0, n_row])

    plt.show()


def mark_orders(
    im, min_cluster=500, filter_size=120, noise=8, opower=4, plot=False, manual=True
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
    plot : bool, optional
        wether to plot the final order fits (default: False)
    manual : bool, optional
        wether to manually select clusters to merge (strongly recommended) (default: True)

    Returns
    -------
    orders : array[nord, opower+1]
        order tracing coefficients (in numpy order, i.e. largest exponent first)
    """

    # Getting x and y coordinates of all pixels sticking above the filtered image
    x, y, clusters, n_clusters = find_clusters(im, min_cluster, filter_size, noise)
    # disregard borders of the image
    clusters[(x == 0) | (y == 0) | (x == im.shape[1] - 1) | (y == im.shape[0] - 1)] = 0
    if n_clusters == 0:
        raise Exception("No clusters found")

    # Reorganize x, y, clusters into a more convenient "pythonic" format
    # x, y become dictionaries, with an entry for each order
    # n is just a list of all orders (ignore cluster == 0)
    n = np.unique(clusters)
    n = n[n != 0]
    x = {c: x[clusters == c] for c in n}
    y = {c: y[clusters == c] for c in n}

    # fit polynomials
    orders = fit_polynomials_to_clusters(x, y, n, opower)

    # Merge clusters
    x, y, n = merge_clusters(im.shape, orders, x, y, n, plot=plot, manual=manual)

    orders = fit_polynomials_to_clusters(x, y, n, opower)

    # TODO: Discard bad clusters

    # sort orders from bottom to top, using mean coordinate
    # TODO: better metric for position?
    key = np.array([[i, np.mean(x[i])] for i in n])
    key = key[np.argsort(key[:, 1]), 0]
    n = np.arange(len(n), dtype=int)
    x = {c: x[key[c]] for c in n}
    y = {c: y[key[c]] for c in n}
    orders = np.array([orders[key[c]] for c in n])

    column_range = np.array([[np.min(y[i]), np.max(y[i])] for i in n])

    if plot:
        plot_orders(im, x, y, n, orders, column_range)

    return orders, column_range
