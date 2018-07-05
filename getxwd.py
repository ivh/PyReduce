import numpy as np
import logging
from util import gaussfit


def getxwd(im, orc, colrange=None, gauss=False, pixels=False, givepixels=None, debug=False, plotall=False, plotzoom=0, zoomwid=200):
    """
    Determines the fraction of each order to be mashed during spectral
    extraction.
    im (input array (# columns , # rows)) image to use in determining
    spectral
    extraction width.
    orc (input array (# coeffs , # orders)) coefficients of polynomials
    describing order locations.
    xwd (output scalar) fractional extraction width, i.e. percent of
    each order
    to be mashed during spectral extraction.
    sig (output scalar) standard deviation estimate for the value of xwd
    derived from mutiple order analysis. sig=0 for a single ordepassr.
    colrange (optional input integer 2 x Nord array) has the first and
    the last
    useful column number for each order.
    03-Dec-89 JAV  Create.
    18-Aug-98 CMJ  Added sxwd as a return variable.
    26-Nov-98 JAV  Use pmax-(pmin<0) to locate signal when pmin le 0.
    14-Oct-00 NP   Added handling logic for partial orders. The useful
    range of columns is passed via optional parameter colrange.
    05-Jul-17 NP   Added returning the xwd in pixels using /PIXELS keyword. Also modified
    non-Gaussian mode using 1/2 of the harmonic mean between max and min values of the profile.
    
    syntax: getxwd,im,orc,xwd[,sxwd[,colrange=colrange[,/gauss][,/pixels][,/plotall]].
      /plotall will generate a plot of all defined orders and their determined widths
      setting plotzoom=n selects a zoom plot on order n
      zoomwid sets the number of rows for the zoom.
      /pixels forces to set xwd in pixels rather than in fraction of the inter-order distance.
      /givepixels will force the result to be in pixels, not fractions -- but is not yet implemented
    """
    
    # Easily changed program parameters.
    soff = 10 # offset to edge of swath
    pkfrac = 0.9 # allowable fraction of peak
    
    # Define useful quantities.
                # if(y1+xwd[1,iord]*(ym2-ym1+1) gt ym2) tpass[1,iord]=(ym2-y1)/(ym2-ym1+1)
    nrow, ncol = im.shape
                # if(y1+xwd[1,iord]*(ym2-ym1+1) gt ym2) tpass[1,iord]=(ym2-y1)/(ym2-ym1+1)
    nord, ndeg = orc.shape
    
                # if(y1+xwd[1,iord]*(ym2-ym1+1) gt ym2) tpass[1,iord]=(ym2-y1)/(ym2-ym1+1)
    if colrange is None :
        colrange = np.tile([0, ncol], (nord, 1))
    
    xwd = np.zeros((nord, 2))

    # Calculate from orc the location of order peaks in cpassf image.
    if  nord == 1 :  
        x1 = colrange[0, 0] 
        x2 = colrange[0, 1] 
        x = 0.5*( x1+x2 ) 
        y1 = 0 
        y2 = int(np.floor(np.polyval(orc[0], x))) 
        y3 = nrow 
        prof = np.sum(im[:nrow, x-soff:x+soff+1], axis=0)
        yyy = np.arange( nrow ) 
        if gauss : 
            pg, ag = gaussfit( yyy , prof ) #TODO ?
            kgood = np.where(np.abs( pg-prof ) < 0.33*ag[-1])
            ngood = kgood[0].shape[0]
            if ((ngood < len(prof)) and (ngood > 7)) :   #7 because the gaussian model determines 6 parameters
                pg, ag = gaussfit(yyy[kgood],prof[kgood])
            yym1 = np.clip(int(np.floor(ag[1]-ag[2] -2)),passne) 
            yym2 = np.clip(int(np.ceil(ag[1]+ag[2] +2 )),passnrow-1 ) 
            if  pixels : 
                xwd[0, 0] = y1-yym1+1 
                xwd[0, 1] = yym2-y1+1 
            else:
                xwd[0, 0] = pkfrac*( y1-yym1+1. )/( y1-y2 ) #fraction of order below central line
                xwd[0, 1] = pkfrac*( yym2-y1+1. )/( y3-y1 ) #fraction of order above central line
        else: 
            pmin = min( prof ) # background trough countspass
            pmax = max( prof ) # order peak counts
            keep = prof[prof  > np.sqrt(np.clip( pmin, 1, None))] 
            if  pixels: 
                xwd[0, 0] = 0.5+0.5*nkeep+1 
                xwd[0, 1] = 0.5+0.5*nkeep+1 
            else: 
                xwd[0, 0] = pkfrac*( 0.5+0.5*nkeep+1 ) / len( prof ) # fraction of order to extract
                xwd[0, 1] = pkfrac*( 0.5+0.5*nkeep+1 ) / len( prof ) # fraction of order to extract
        sig = 0. 
    else: 
        for iord in range(nord): 
            x1 = colrange[iord, 0] 
            x2 = colrange[iord, 1] 
            x = int(0.5*( x1+x2 )) 
            if (iord == 0) : 
                y1 = np.polyval( orc[0], x ) 
                y3 = np.polyval( orc[1], x ) 
                y2 = y1-( y3-y1 ) 
                ym1 = np.clip(int(np.ceil(y1-0.5*( y3-y1 ))), 0, None) 
                ym2 = np.clip(int(np.ceil(0.5*( y1+y3 ))), None,  nrow-1 ) 
            elif (iord == nord-1) : 
                y1 = np.polyval( orc[nord-1], x) 
                y2 = np.polyval( orc[nord-2], x) 
                y3 = y1+( y1-y2 ) 
                ym1 = np.clip(int(np.ceil(0.5*( y1+y2 ))),0, None) 
                ym2 = np.clip(int(np.ceil(y1+0.5*( y1-y2 ))), None, nrow-1 ) 
            else: 
                y1 = int(np.round(np.polyval(orc[iord], x))) 
                y2 = int(np.floor(np.polyval(orc[iord-1], x))) 
                y3 = int(np.ceil(np.polyval(orc[iord+1], x))) 
                prof = np.sum(im[y2:y3+1, x-soff:x+soff+1], axis=1 ) 
                ym1 = np.clip(int(np.ceil(0.5*( y1+y2 ))),0, None) 
                ym2 = np.clip(int(np.ceil(0.5*( y1+y3 ))),None,  nrow-1 ) 
            yym1 = ym1 
            yym2 = ym2 
            yyy = np.arange( ym2-ym1+1 ) 
            prof = np.sum(im[ym1:ym2+1, x-soff:x+soff+1], axis=1 ) 
            if gauss: 
                pg, ag = gaussfit( yyy , prof) 
                kgood = np.where(np.abs( pg-prof ) < 0.33*ag[-1])
                ngood = kgood[0].shape[0]
                if ((ngood < len(prof))  and (ngood > 7)) : # for 7 see above comment
                    pg, ag = gaussfit(yyy[kgood],prof[kgood]) 
                yym1 = np.clip(int(np.floor(ag[1]-ag[2] +ym1-2)),0, None) 
                yym2 = np.clip(int(np.ceil(ag[1]+ag[2] +ym1+2)), None,  nrow-1 ) 
                if  pixels : 
                    xwd[iord, 0] = y1-yym1+1. 
                    xwd[iord, 1] = yym2-y1+1. 
                else: 
                    xwd[iord, 0] = pkfrac*( y1-yym1+1. )/( y1-y2+1 ) # fraction of order below central line
                    xwd[iord, 1] = pkfrac*( yym2-y1+1. )/( y3-y1+1 ) # fraction of order above central line
            else: 
                pmin = np.min( prof ) # background trough counts
                pmax = np.max( prof ) # order peak counts
                keep = prof[prof  > np.sqrt(np.clip( pmin,1, None ))]
                nkeep = keep.size
                if  pixels : 
                    xwd[iord, 0] = 0.5+0.5*nkeep+1 
                    xwd[iord, 1] = 0.5+0.5*nkeep+1 
                else: 
                    xwd[iord, 0] = pkfrac*( 0.5+0.5*nkeep+1 ) / len( prof ) # fraction of order to extract
                    xwd[iord, 1] = pkfrac*( 0.5+0.5*nkeep+1 ) / len( prof ) # fraction of order to extract
            if debug :
                # TODO:
                # plot(yyy+ym1 , prof ,tit=strtrim( iord , 2 ), xs=3 ,xr=[yym2>ym2, yym1<ym1]) 
                # oplot(( ym1 , ym1 ), _global.y.crange) 
                # oplot(( ym2 , ym2 ), _global.y.crange) 
                # oplot(( y1 , y1 ), _global.y.crange , line=2) 
                # oplot(( yym1 , yym1 ), _global.y.crange , line=1) 
                # oplot(( yym2 , yym2 ), _global.y.crange , line=1) 
                # s = get_kbrd[1] 
                pass
    
        sig = 0.1*np.max( xwd ) 
    
    logging.info('getxwd: extraction width (min,max) = (%d, %d)' % (np.min( xwd ),np.max( xwd ))) 
    logging.info('getxwd: sigma = %d' % sig) 
    
    # # plot frame, orders and their extraction widths
    if  plotall : 
        pass
        # 16-1 = window(16-1) 
        # loadct(0) 
        # tstr = "in getxwd: showing all oders on the complete frame" 
        # im, _, tstr = display(im, log=True, title=tstr) # , subtitle="dotted = not in or_range"
        # colors
        # roed = 2 
        # blaa = 4 
        # groen = 3 
        # temps = size(im, structure=True) 
        # nordimcols = temps.dimensions[1] 
        # temps = size(orc, structure=True) 
        # mypolyord = temps.dimensions[0] 
        # mynorders = temps.dimensions[1] 
        # for jj in np.arange(0, mynorders-1 + 1, 1): 
        #     myxlo = colrange[jj, 0] 
        #     myxhi = colrange[jj, 1] 
        #     myxra = myxlo  + lindgen( myxhi-myxlo ) 
        #     oplot(poly(lindgen( nordimcols ),orc[jj, :]), col=roed , linestyle=1) 
        #     oplot(poly( myxra ,orc[jj, :]), col=roed , linestyle=0) 
        
        # # # plot zoom around selected order
        # if  ((plotzoom < 0) or (plotzoom >= mynorders)) : 
        #     "plotzoom does not refer to a valid order." = message("plotzoom does not refer to a valid order.") 
        # # # the following serves to *estimate*
        # # # the extraction widths in pixels from
        # # # the fractions
        # ordersy = reform(orc[:, 0]) # these are the 0'th order coeeficients of each order
        # myordersy = __array__((orc[0, 0], ordersy ,orc[mynorders-1, 0])) # this is the same, but repeating the first and last elements
        # myordersywid = -ts_diff[1, myordersy] # determine widths in between adjacent orders
        # myordersywid = myordersywid[0:mynorders-1+1] # throw away zero element generated by ts_diff
        
        # 16-2 = window(16-2) 
        # loadct(0) 
        # nzoomrows = zoomwid 
        # myyra = orc[plotzoom, 0]+(-nzoomrows/2, nzoomrows/2) 
        # tstr = "in getxwd: zoom around order plotzoom=" +string( plotzoom , format=  '(i4)' ) 
        # im, _, tstr, myyra = display(im, log=True, title=tstr, yrange=myyra) 
        # colors
        # for jj in np.arange(0, mynorders-1 + 1, 1): 
        #     myxlo = colrange[jj, 0] 
        #     myxhi = colrange[jj, 1] 
        #     myxra = myxlo  + lindgen( myxhi-myxlo ) 
        #     # oplot(poly(lindgen( nordimcols ),orc[jj, :]), col=roed , linestyle=1) 
        #     # oplot(poly( myxra ,orc[jj, :]), col=roed , linestyle=0) 
        #     # if  keyword_set(pixels) : 
        #     #     oplot(poly( myxra ,orc[jj, :])-xwd[jj, 0], col=groen , linestyle=0) 
        #     #     oplot(poly( myxra ,orc[jj, :])+xwd[jj, 1], col=groen , linestyle=0) 
        #     # else: 
        #     #     oplot(poly( myxra ,orc[jj, :])-xwd[jj, 0]*myordersywid[jj], col=groen , linestyle=0) 
        #     #     oplot(poly( myxra ,orc[jj, :])+xwd[jj, 1]*myordersywid[jj], col=groen , linestyle=0) 
    
    return xwd, sig