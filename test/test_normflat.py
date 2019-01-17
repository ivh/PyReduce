import pytest

from pyreduce.normalize_flat import normalize_flat

# def test_normflat(flat, orders, settings):
#     flat, fhead = flat
#     orders, column_range = orders

#     order_range = (0, len(orders) - 1)

#     flat, blaze = normalize_flat(
#         flat,
#         orders,
#         gain=fhead["e_gain"],
#         readnoise=fhead["e_readn"],
#         dark=fhead["e_drk"],
#         column_range=column_range,
#         order_range=order_range,
#         extraction_width=settings["normflat.extraction_width"],
#         degree=settings["normflat.scatter_degree"],
#         threshold=settings["normflat.threshold"],
#         lambda_sf=settings["normflat.smooth_slitfunction"],
#         lambda_sp=settings["normflat.smooth_spectrum"],
#         swath_width=settings["normflat.swath_width"],
#         plot=False,
#     )
