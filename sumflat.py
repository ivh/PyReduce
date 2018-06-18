import numpy as np
from idl_lib import *
import _global


def sumflat(root, list, flat, head, bias=None, debug=None):
    # Combines flat field images, subtracting bias, fixing bad pixels, and
    # trimming, as needed.
    # root (string) root name to use in constructing file names.
    # ; Template (input string) template for constructing names of FITS format disk
    # ;   files containing images to be added. The only occurance of # in Template
    # ;   will be replaced succesively by integers in FNum.
    # List (input vector) list of FF filenames
    # Im (output array(# columns,# rows)) coadded, cosmic ray corrected image.
    # Head (optional output string vector(# cards x 80)) FITS header associated
    # with coadded image.
    # bias= (array) bias image **BEFORE TRIMMING**
    # xtrim= (vector(2)) first and last column to include in output
    # 08-Aug-91 JAV	Fixed bug in signed to unsigned I*2 logic.
    # 05-Sep-92 JAV	Fixed sign error in application of BZero.
    # 25-Feb-93 JAV	Changed
    # 29-Mar-93 JAV	Use FITS standard BZERO and BSCALE transformation:
    # Image = (BSCALE * ScaledImage) + BZERO
    # Detect older, nonstandard tranformations and abort.
    # 25-May-99 JAV  Use routines from astron library
    # 26-May-99 JAV  Adapted from cosmic.pro. Added bias subtraction, trimming,
    # noise model, and revamped bad pixel identification.
    # 26-Jan-2000  NP, removed common ham_aux, replaced with data from
    # inst_setup structure available in ham.common
    # 
    
    @ham.common
    
    if  n_params() < 2 : 
        print('syntax: sumflat, root, list [,flat ,head ,bias= ,/debug]') 
        return  
    
    if  not keyword_set(inst_setup) : 
        _ = message('run inst_setup before proceeding with reductions.', info=True) 
        return  
    
    # Defaults for external parameters.
    if  n_elements(debug) == 0 : # debug off by default
        debug = 0 
    
    # Program paramters
    smark = '#' # template substitution mark
    byteswap = 0 # 1=byte swap, 0=no byte swap
    thresh = 3.5 # bad pixel threshold (sigmas)
    hwin = 50 # half window for column avg
    
    # Verify sensibility of passed parameters.
    nfil = n_elements( list ) # number of files in list
    if  nfil < 3 : # true: too few images
        message('too few images to perform cosmic ray removal.') 
    
    # Initialize header information lists (one entry per file).
    nclist = intarr( nfil ) -  1 # # columns list (-1=not found)
    nrlist = nclist  +  2 # # rows list (1=default)
    hslist = nclist # header size list
    bzlist = fltarr( nfil ) # bzero list (0.0=default)
    bslist = bzlist  +  1.0 # bscale list (1.0=default)
    expolist = bzlist # exposure time list
    obslist = '' 
    obsnum = 0 # if not given
    totalexpo = 0 # init total exposure time
    
    # Loop through files in list, grabbing and parsing FITS headers.
    fnlen = max(strlen( list )) # length of longest filename
    print('  file'  + string(replicate( 32 , fnlen-3 )) +  'obs cols rows  object') 
    for ifil in np.arange(0, nfil-1 + 1, 1): # loop though files
        fname = list[ifil] # construct actual filename
        headin = headfits( fname ) # get fits header (as bytarr)
        if  ifil == 0 : # save first header for output
            head = headin 
        ncrd = n_elements( headin ) # number of header cards
        hsiz = 2880*ceil(80*ncrd/2880.0) 
        hslist[ifil] = hsiz # header size (in bytes)
        
        # Parse header for image information.
        obsnum = 0 
        nclist[ifil] = sxpar( headin , 'naxis1' ) 
        nrlist[ifil] = sxpar( headin , 'naxis2' ) 
        bzlist[ifil] = sxpar( headin , 'bzero' ) 
        if (_global.err != 0) : 
            bzlist[ifil] = 0. 
        bslist[ifil] = sxpar( headin , 'bscale' ) 
        if (_global.err != 0) : 
            bslist[ifil] = 1. 
        object = sxpar( headin , 'object' ) 
        obslist = obslist  + strtrim( ifil , 2 ) 
        exposure = sxpar( headin , 'exptime' ) 
        if (_global.err != 0) : 
            exp_start = sxpar( headin , 'tm-start' ) 
            exp_end = sxpar( headin , 'tm-end' ) 
            exposure = exp_end  -  exp_start 
        totalexpo = totalexpo  +  exposure # accumulate exposure times
        
        if  nclist(ifil) == -1 : # true: naxis1 not found
            message('unable to determine number of image columns from header.') 
        blanks = '' # null insert string
        if  strlen(fname) > fnlen : # true: need to pad with spaces
            blanks = string(replicate( 32 , fnlen  - strlen( fname ))) 
        print(format=  '(2x,2a,i4,i5,i5,2a)' , fname , blanks , obsnum ,nclist[ifil],nrlist[ifil], '  ' ,strtrim( object )) # summarize header info
    
    # Get gain and read noise from header.
    gain = sxpar( head , 'gain' ) 
    rdnoise = sxpar( head , 'rdnoise' ) 
    gain = 1.6 
    rdnoise = 8.97 
    
    # Verify comparable image sizes.
    if  min(nclist) != max(nclist) : # variable number of rows
        message('not all images have the same number of columns.') 
    ncol = long(nclist[0]) # # columns in all images
    if  min(nrlist) != max(nrlist) : # variable number of rows
        message('not all images have the same number of rows.') 
    nrow = long(nrlist[0]) # # rows in all images
    
    # Open all files to be processed and skip headers.
    ulist = intarr( nfil ) -  3 # init unit list
    for ifil in np.arange(0, nfil-1 + 1, 1): # loop thru filenames
        fname = list[ifil] # construct actual filename
        unit, fname, _ = openr(unit, fname, get_lun=True) # open file and get unit number
        ulist[ifil] = unit # store unit number
        unit, hslist[ifil] = point_lun(unit ,hslist[ifil]) # skip header
    
    # Setup for trimming.
    if  keyword_set(xtrim) : 
        it0 = xtrim[0] 
        it1 = xtrim[1] 
        ntrim = it1  -  it0  +  1 
    else: 
        it0 = -1 # flag value
        ntrim = ncol 
    
    # Loop thru processing sets, coadding data from all files in list and
    # correcting cosmic rays.
    nfix = 0 # init fixed pixel counter
    flat = fltarr( ntrim , nrow ) # init r*4 image array
    mbuff = fltarr( ntrim , nfil ) # init multi-file buffer
    prob = fltarr( ntrim , nfil ) # init probability function
    mfit = fltarr( ntrim , nfil ) # init fit to mbuff
    modamp = fltarr( ntrim ) # init amplitude of mbuff
    ibuff = intarr( ncol ) # init i*2 single-file buffer
    buff = fltarr( ntrim ) # init final image data buffer
    for irow in np.arange(0, nrow-1 + 1, 1): # loop thru processing units
        if  irow % 100 == 0 and irow > 0 : # report status
            strtrim(string( irow ), 2 ) +  ' rows processed - '  + strtrim(string( nfix ), 2 ) +  ' pixels fixed so far.', _ = message(strtrim(string( irow ), 2 ) +  ' rows processed - '  + strtrim(string( nfix ), 2 ) +  ' pixels fixed so far.', info=True) 
        
        # Read in current row for all files. Subtract bias, if available.
        for ifil in np.arange(0, nfil-1 + 1, 1): # loop thru files
            ulist[ifil], ibuff = readu(ulist[ifil], ibuff) # read unsigned i*2 as signed
            dummy = long( ibuff ) +2**15 
            # ii = where(iBuff lt 0, nii)
            # if(nii gt 0) then Dummy(ii)=(iBuff(ii)+2^15)
            dummy = dummy  * bslist[ifil] + bzlist[ifil] # rescale data
            # if(irow eq 45) then stop
            if  keyword_set(bias) : # true: need to subtract bias
                dummy = dummy  - bias[irow, :] 
            if  it0 != -1 : # true: need to trim
                dummy = dummy[it0:it1+1] 
            mbuff[ifil, :] = dummy # insert rescaled data into buffer
        
        # Construct a probability function based on mbuff data.
        for icol in np.arange(hwin, ntrim-hwin-1 + 1, 1): 
            filwt = total(mbuff[:, icol-hwin:icol+hwin+1], 1 ) 
            filwt = filwt  / total( filwt ) 
            prob[:, icol] = filwt 
        for icol in np.arange(0, hwin-1 + 1, 1): 
            prob[:, icol] = 2.0  * prob[:, hwin] - prob[:, 2*hwin-icol] 
        for icol in np.arange(ntrim-hwin, ntrim-1 + 1, 1): 
            prob[:, icol] = 2.0  * prob[:, ntrim-hwin-1] - prob[:, 2*( ntrim-hwin-1 ) -icol] 
        
        # Loop through columns, fitting data and constructing mfit.
        # Reject cosmic rays in amplitude fit by ignoring highest and lowest point.
        for icol in np.arange(0, ntrim-1 + 1, 1): 
            rat = mbuff[:, icol] / prob[:, icol] 
            isort = sort( rat ) 
            amp = total(rat[isort( 1  :  nfil-2 )])/( nfil-2 ) 
            modamp[icol] = amp 
            mfit[:, icol] = amp  * prob[:, icol] 
        
        # Construct noise model.
        predsig = sqrt(rdnoise**2 +(mfit/gain)) 
        
        # Identify outliers.
        ibad = where( mbuff-mfit  >  thresh*predsig , count='nbad' ) 
        
        # Debug plot.
        if  keyword_set(debug) and irow % 10 == 0 : 
            plot(mbuff-mfit , xsty=3 , ysty=3 , ps=3 , ytit=  'data - model  (adu)' , tit=  'row = '  + strtrim( irow , 2 ) +  ',  threshold = '  + fmtnum['(f9.4)', thresh]) 
            colors
            oplot(thresh*predsig , co=4) 
            oplot(-thresh*predsig , co=4) 
            if  nbad > 0 : 
                oplot(ibad ,mbuff[ibad]-mfit[ibad], ps=2 , syms=1.0 , co=2) 
            print(form=  '(a,)' , 'push space to continue...'  junk  = get_kbrd[1]) 
            print('') 
        
        # Fix bad pixels, if any.
        if  nbad > 0 : 
            mbuff[ibad] = mfit[ibad] 
            nfix = nfix  +  nbad 
        
        # Construct the summed flat.
        buff = total( mbuff , 2 ) /  nfil 
        plot(buff , xs=3 , ys=3 , title=  'row ' +strtrim( irow , 2 )) 
        flat[irow, :] = buff # insert final data into image
    
    2 ), _ = message('total cosmic ray hits identified and removed: ' + strtrim(string( nfix ), 2 ), info=True) 
    for ifil in np.arange(0, nfil-1 + 1, 1): # loop thru files
        ulist[ifil] = free_lun(ulist[ifil]) # free logical unit
    
    # Add info to header.
    head = sxaddpar(head , 'bzero' , 0.0) 
    head = sxaddpar(head , 'bscale' , 1.0) 
    head, totalexpo = sxaddpar(head , 'exptime' , totalexpo) 
    head, totalexpo = sxaddpar(head , 'darktime' , totalexpo) 
    head, rdnoise  / sqrt( nfil ) = sxaddpar(head, 'rdnoise', rdnoise  / sqrt( nfil ), ' noise in combined image, electrons') 
    head, obslist = sxaddpar(head , 'obslist' , obslist) 
    head, nfil = sxaddpar(head , 'nimages' , nfil , ' number of images summed') 
    head, nfix = sxaddpar(head , 'npixfix' , nfix , ' pixels corrected for cosmic rays') 
    head = sxaddpar(head , 'history' , 'images coadded by sumflat.pro on '  + systime()) 
    if  keyword_set(bias) : 
        head = sxaddpar(head , 'history' , 'bias subtracted during coaddition') 
    if  keyword_set(xtrim) : 
        head = sxaddpar(head , 'history' , 'original images trimmed to columns ['  + strtrim( it0 , 2 ) +  ','  + strtrim( it1 , 2 ) +  ']'  +  ' during coaddition') 
    
    # Write output FITS file.
    root  +  '.flat', flat, head = writefits(root  +  '.flat' , flat , head) 
    
    pass
    return root, list, flat, head, bias, debug 
