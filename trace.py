import numpy as np

# From my understanding of the IDL code, locate_clusters takes into
## input: im,n,nx,ny,filter,mask,noise,shift_offset
## output: x,y
# TODO: check with TM: it seems this function is implemented in C in cluster.c
def locate_clusters(im,n,x,y,nx,ny,filter,mask,noise,shift_offset):

    a = 0
    return(x,y)

# TODO: implement the correct default value for the parameters
def mark_orders(im,power,filter=20.,error,thres=400,plot,polarim,mask,manual,noise=1.,color,shift_offset,cluster_limits):

    # Computing the size of the image array
    nx = im.shape[0]    # size in x-dimension
    ny = im.shape[1]    # size in y-dimension
    n = im.size         # number of elements in the array

    # Getting x and y coordinates of all pixels sticking above the filtered image
    (x,y) = locate_clusters(im,n,x,y,nx,ny,filter,mask,noise) # no shift_offset

    # Locating and marking clusters of pixels
    # TODO: here the IDL call of cluster is complex and will be simplified with the python wrapper
    (index,nregions) = cluster(x,y,nx,ny,nregions,thres, iplot) # TODO: implement the input/output parameters

    # Obtaining the coordinates where index (output from cluster) is positive
    ind = (index > 0).nonzero()
    n = ind.size
    if ind > thres:
        x = x[ind]
        y = y[ind]
        index = index[ind]
    else:
        print('MARK_ORDERS: No clusters identified')    #TODO: decide on error nomenclature

    if nregions <= 1:
        print('MARK_ORDERS: No clusters identified')    #TODO: decide on error nomenclature

    # Finding all the clusters crossing the bottom boundary
    


    return(orders)
