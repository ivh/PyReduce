import numpy as np
import matplotlib.pyplot as plt

from cwrappers import find_clusters


# TODO: implement the correct default value for the parameters
def mark_orders(im, **kwargs):
    min_cluster = kwargs.get("orders_threshold", 500)
    filter_size = kwargs.get("orders_filter", 120)
    noise = kwargs.get("orders_noise", 8)
    opower = kwargs.get("orders_opower", 4)

    # Getting x and y coordinates of all pixels sticking above the filtered image
    x, y, clusters, n_clusters = find_clusters(
        im, min_cluster, filter_size, noise
    )  # no shift_offset
    if n_clusters == 0:
        raise Exception("No clusters found")

    orders = []
    for i in range(1, n_clusters + 1):
        x_i = x[clusters == i]
        y_i = y[clusters == i]
        fit = np.polyfit(y_i, x_i, opower)
        orders += [fit]

    order_range = [i for i in range(1, n_clusters+1)]

    # TODO combine clusters
    if kwargs.get("plot", False):
        cluster_img = np.full_like(im, 0)
        cluster_img[x, y] = clusters
        cluster_img = np.ma.masked_array(cluster_img, mask=cluster_img == 0)

        plt.subplot(121)
        plt.imshow(im)
        plt.title("Input")

        plt.subplot(122)
        plt.imshow(cluster_img, cmap=plt.get_cmap("tab20"))
        plt.title("Clusters")

        for order in orders:
            x = np.arange(im.shape[1])
            y = np.polyval(order, x)
            plt.plot(x, y)

        plt.ylim([0, im.shape[0]])
        plt.show()

    return orders, order_range
