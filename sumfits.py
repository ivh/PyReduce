import numpy as np

import astropy.io.fits as fits
import datetime


#from modeinfo import modeinfo
# temporary fix
from modeinfo_uves import modeinfo_uves as modeinfo

# TODO find faster calculation ways
# TODO avoid reading files several times


def remove_bad_pixels(p, buff, row, nfil, ncol_a, rdnoise_amp, gain_amp, thresh):
    """ find and remove bad pixels """
    iprob = p > 0

    rat, rat.shape = buff[iprob, row] / p[iprob], p.shape
    amp = (np.sum(rat, axis=0) - np.min(rat, axis=0) -
           np.max(rat, axis=0)) / (nfil - 2)

    # make new array for prob, so p is not changed by reference
    prob, prob.shape = p[iprob], p.shape
    prob = amp * prob

    mfit = np.where(iprob, prob, 0)
    predsig = np.sqrt(rdnoise_amp**2 + (mfit / gain_amp))

    # Identify outliers.
    ibad = buff[:, :, row] - mfit > thresh * predsig
    nbad = len(np.nonzero(ibad.flat)[0])

    # Construct the summed flat.
    b = np.where(ibad, mfit, buff[:, :, row])
    return np.sum(b, axis=0), nbad


def calc_filwt(buff):
    """ Construct a probability function based on mBuff data. """
    # boxcar average for irow
    filwt = np.sum(buff, axis=2, dtype=np.float32)
    # norm for probability
    tot_filwt = np.sum(filwt, axis=0, dtype=np.float32)
    inorm = tot_filwt > 0
    tot_filwt, tot_filwt.shape = tot_filwt[inorm], tot_filwt.shape
    filwt[:, inorm] = filwt[:, inorm] / tot_filwt
    return filwt


def sumfits(list, instrument, summ=None, head=None, debug=None, err=None, thres=3.5, hwin=50, mask=None, xr=None, yr=None, exten=1):
    """
    Subroutine to correct cosmic rays blemishes, while adding otherwise
    # similar images.
    # List - (input string array) list of the filenames
    # Summ - (output array(# columns,# rows)) coadded, cosmic ray corrected image.
    # Head - (optional output string vector(# cards x 80)) FITS header associated
    # with coadded image.
    # Optional parameters:
    # debug - if set, plot scatter diagrams for cosmic ray corrections
    # err   - string, if set, it contains an error message
    """
    err = None
    np.seterr(all='raise')

    # Verify sensibility of passed parameters.
    lst = np.lib.arraysetops.unique(list)
    nfil = len(lst)  # number of files in list

# ===========================================================================
    if nfil < 2:  # true: only one image
        summ = fits.open(lst[0])  # , exten='all', pdu=True)
        head = summ[0].header
        head = modeinfo(head, instrument, xr=xr, yr=yr)
        if len(summ) == 1:
            err = 'sumfits: failed reading file ' + lst[0]
        return summ[exten], head
# ===========================================================================
    elif nfil == 2:  # ;true: only two images
        im = fits.open(lst[0])  # , exten=exten, pdu=True)
        h = im[0].header
        h.extend(im[exten].header, strip=False)
        h, exp1 = modeinfo(h, instrument, time=True)
        exp1 = exp1["time"]
        summ = fits.open(lst[1])  # , exten=exten, pdu=True)
        head = summ[0].header
        head.extend(summ[exten].header, strip=False)
        summ = summ[exten].data + im[exten].data
        
       

        head, exp2 = modeinfo(head, instrument, time=True,
                              readn=True, orient=True, xr=xr, yr=yr)
        exp2, rdnoise, orient = exp2["time"], exp2["readn"], exp2["orient"]
        totalexpo = exp1 + exp2
        if len(summ) == 1:
            raise Exception('sumfits: failed reading file ' + lst[0] + lst[1])
        # Add info to header.
        head['bzero'] = 0.0
        head['bscale'] = 1.0
        head['exptime'] = totalexpo
        head['darktime'] = totalexpo
        # Because we do not devide the signal by the number of files the
        # read-out noise goes up by the square root of the number of files
        head['rdnoise'] = (rdnoise * np.sqrt(nfil),
                           'noise in combined image, electrons')
        head['nimages'] = (nfil, 'number of images summed')
        head.add_history('images coadded by sumfits.pro on %s' %
                         datetime.datetime.now())
        head, _ = modeinfo(head, instrument, xr=xr, yr=yr)
        return summ, head

# ===========================================================================
# Initialize header information lists (one entry per file).
    bzlist = np.zeros(nfil)  # bzero list (0.0=default)
    bslist = np.full(nfil, 1.)  # bscale list (1.0=default)
    expolist = np.zeros(nfil)  # exposure time list
    totalexpo = 0  # init total exposure time
    # lunit = np.zeros(nfil, dtype=int)  # array of units

    # Loop through files in list, grabbing and parsing FITS headers.
    fnlen = np.max([len(l) for l in list])  # length of longest filename
    print('  file' + ' ' * (fnlen - 3) + 'obs cols rows  object  exposure')
    #lunit_pool = 128

    # TODO read files only once

    fname = lst[0]
    # , exten='all', header_only=True, pdu=True)
    hdu = fits.open(fname)
    head = hdu[0].header
    head.extend(hdu[exten].header, strip=False)

    head, kw = modeinfo(head, instrument, time=True, gain=True,
                        readn=True, orient=True, xr=xr, yr=yr)
    exp, gain, readnoise, orient = kw["time"], kw["gain"], kw["readn"], kw["orient"]

    ncol_old = head['naxis1']  # columns
    nrow_old = head['naxis2']  # rows

    # check if we deal with multiple amplifiers
    try:
        n_ampl = head['e_ampl']
    except KeyError:
        n_ampl = 1
    if n_ampl == 1:
        # section(s) of the detector to process
        xlo = [head['e_xlo']]
        xhi = [head['e_xhi']]
        ylo = [head['e_ylo']]
        yhi = [head['e_yhi']]
    else:
        # section(s) of the detector to process
        xlo = [head['e_xlo*']]
        xhi = [head['e_xhi*']]
        ylo = [head['e_ylo*']]
        yhi = [head['e_yhi*']]

    nfix = 0  # init fixed pixel counter
    summ = np.zeros((nrow_old, ncol_old))  # init r*4 output image array

    # check if non-linearity correction
    try:
        linear = head['e_linear']
    except KeyError:  # is needed and prepare for it.
        linear = True
    if not linear:
        pref = head['e_prefmo']

    # outer loop through amplifiers (note: 1,2 ...)
    for amplifier in range(n_ampl):
        for ifil in range(nfil):  # loop though files
            fname = lst[ifil]  # take the next filename

            im = fits.open(fname)
            head = im[0].header  # , exten='all', header_only=True, pdu=True)
            head.extend(im[1].header, strip=False)

            head, kw = modeinfo(head, instrument, time=True, gain=True,
                                readn=True, orient=True)
            exp, gain, readnoise, orient = kw["time"], kw["gain"], kw["readn"], kw["orient"]

            if (len(head) == 1):
                err = 'sumfits: ' + head + ' in file ' + fname
                raise ValueError(err)

            ncol = head['naxis1']  # columns
            nrow = head['naxis2']  # rows

            if (nrow != nrow_old or ncol != ncol_old):
                err = 'sumfits: file ' + fname + ' has different dimensions'
                raise ValueError(err)

            bitpix = head['bitpix']  # rows
            object = head['object']  # get object id
            bslist[ifil] = head['bscale']  # pixel scale
            if bslist[ifil] == 0:
                bslist[ifil] = 1
            bzlist[ifil] = head['bzero']  # pixel offset

            # of the correct extension

            im = im[exten]

            exposure = head['EXPTIME']
            totalexpo = totalexpo + exposure  # accumulate exposure times
            print(fname, ifil, ncol, nrow, '  ', object,
                  exposure)  # summarize header info

        xleft = xlo[amplifier]
        xright = xhi[amplifier]
        ybottom = ylo[amplifier]
        ytop = yhi[amplifier]

        if (n_ampl > 1):
            gain_amp = gain[amplifier]
            rdnoise_amp = rdnoise[amplifier]
        else:
            gain_amp = head['e_gain']
            rdnoise_amp = head['e_readn']

        orient = head['e_orient']
        if orient == 0 or orient == 2 or orient == 5 or orient == 7:
            raise NotImplementedError
            ncol_a = xright - xleft + 1
            nrow_a = ytop - ybottom + 1

            block = fits_read(lst, exten=1)
            block = block[:, :, xleft:xright + 1]

            if (ybottom > 0):  # skip ybottom rows
                for irow in range(0, ybottom - 1 + 1, 1):
                    for ifil in np.arange(0, nfil - 1 + 1, 1):  # loop though files
                        block = readu(lunit[ifil], dtype=idl_type, n=ncol)

            for irow in range(ybottom, ytop + 1, 1):  # loop through rows
                if irow % 100 == 0 and irow > 0:  # report status
                    message(strtrim(irow, 2) + ' rows processed - ' +
                            strtrim(string(nfix), 2) + ' pixels fixed so far.', info=True)
                for ifil in np.arange(0, nfil - 1 + 1, 1):  # loop though files
                    block = readu(lunit[ifil], dtype=idl_type, n=ncol)
                    blockf = block * bslist[ifil] + bzlist[ifil]
                    if (not linear):  # linearity needs fixing
                        blockf = call_function(
                            'nonlinear_' + pref, blockf, head, amplifier=amplifier, gain=gain_amp)
                        pass
                    if (keyword_set(mask)):
                        mbuff[ifil, :] = (
                            blockf * mask[irow, :])[xleft:xright + 1]
                    else:
                        mbuff[ifil, :] = blockf[xleft:xright + 1]

                # Construct a probability function based on mbuff data.
                for icol in range(hwin, ncol_a - hwin):
                    # filwt = total(mBuff[iCol-hwin:iCol+hwin,*], 1)
                    filwt = median(
                        mbuff[:, icol - hwin:icol + hwin + 1], dimension=1)

                    tot_filwt = total(filwt)  # norm for probability
                    if (tot_filwt > 0.0):
                        filwt = filwt / tot_filwt
                    prob[:, icol] = filwt

                prob[:, :hwin] = 2 * prob[:, hwin] - prob[:, 2 * hwin:hwin:-1]
                prob[:, ncol_a - hwin:ncol_a] = 2 * prob[:, ncola_a -
                                                         hwin] - prob[:, 2 * (ncol_a - hwin):ncol_a - hwin:-1]

                # Loop through columns, fitting data and constructing mfit.
                # Reject cosmic rays in amplitude fit by ignoring highest and lowest point.
                for icol in range(ncol_a):
                    pr = reform(prob[:, icol])
                    mfit[:, icol] = 0.0
                    iprob, nprob = where(pr > 0.0, count='nprob')
                    if (nprob > 0):
                        rat = reform(mbuff[iprob, icol]) / pr[iprob]
                        amp = (total(rat) - min(rat) - max(rat)) / (nfil - 2)
                        mfit[iprob, icol] = amp * pr[iprob]
                    pass

                # Construct noise model.
                predsig = np.sqrt(rdnoise_amp**2 + abs(mfit / gain_amp))

                # Identify outliers.
                ibad, nbad = where(mbuff - mfit > thresh *
                                   predsig, count='nbad')

                # Debug plot.
                if keyword_set(debug) and irow % 10 == 0:
                    plot(mbuff - mfit, xsty=3, ysty=3, ps=3, ytit='data - model  (adu)', tit='row = ' +
                         strtrim(irow, 2) + ',  threshold = ' + string(thresh, form='(f9.4)'))
                    oplot(thresh * predsig, co=4)
                    oplot(-thresh * predsig, co=4)
                    if nbad > 0:
                        oplot(ibad, mbuff[ibad] - mfit[ibad],
                              ps=2, syms=1.0, co=2)
                    print('push space to continue...')
                    print('')

                # Fix bad pixels, if any.
                if nbad > 0:
                    mbuff[ibad] = mfit[ibad]
                    nfix = nfix + nbad

                # Construct the summed FITS.
                buff = total(mbuff, 2)  # / nfil
                summ[irow, xleft:xright + 1] = buff
        elif orient == 1 or orient == 3 or orient == 4 or orient == 6:

            ncol_a = xright - xleft + 1
            mrow = 2 * hwin + 1  # of rows in the fifo buffer

            if (ytop - ybottom + 1 < mrow):
                err = 'sumfits: the number of rows should be larger than 2 * win = ' + \
                    str(mrow)
                raise ValueError(err)

            # The same as mbuff
            block = np.array([fits.open(f)[exten].data for f in lst])
            block = block[:, :, xleft:xright + 1]
            # Initial Window
            mbuff = np.swapaxes(block[:, :mrow, :], 1, 2)
            # For extrapolation later
            mmbuff = np.swapaxes(block[:, :hwin, :], 1, 2).astype(np.float32)

            fltbuff = np.swapaxes(
                block[:, :mrow + 1, :], 1, 2).astype(np.float32)
            crow = np.arange(hwin, ytop - ybottom + 1) % (mrow)
            jrow = np.arange(2 * hwin, ytop - ybottom + 1) % (mrow)

            # TODO Remove the loop, should be possible
            for irow in range(ybottom + hwin, ytop - hwin + 1):
                count = irow - ybottom - hwin
                if (irow) % 100 == 0:
                    print(irow, ' rows processed - ',
                          nfix, ' pixels fixed so far')

                # This simulates reading the file line by line
                mbuff[:, :, jrow[count]] = block[:, mrow + count - 1, :]
                filwt = calc_filwt(mbuff)

                if irow <= ybottom + 2 * hwin:
                    # for 1st special case: rows lesser than ybottom + hwin
                    fltbuff[:, :, irow - ybottom - hwin] = np.copy(filwt)
                if irow >= ytop - 2 * hwin:
                    # for 2nd special case: rows greater than ytop - hwin
                    fltbuff[:, :, irow + 3 * hwin - ytop + 1] = np.copy(filwt)

                #pr = np.copy(prob)
                summ[irow, xleft:xright + 1], nbad = remove_bad_pixels(
                    filwt, mbuff, crow[count], nfil, ncol_a, rdnoise_amp, gain_amp, thres)
                nfix += nbad

            # 1st special case: rows less than hwin from the 0th row
            for irow in range(ybottom + hwin + 1, ybottom + 2 * hwin + 1):
                lrow = 2 * hwin - irow + ybottom
                prob2 = 2 * fltbuff[:, :ncol_a, 0] - \
                    fltbuff[:, :ncol_a, irow - ybottom - hwin]

                summ[ybottom + lrow, xleft:xright + 1], nbad = remove_bad_pixels(
                    prob2, mmbuff, lrow, nfil, ncol_a, rdnoise_amp, gain_amp, thres)
                nfix += nbad

            # 2nd special case: rows greater than ytop-hwin
            for irow in range(ytop - hwin + 1, ytop + 1):
                prob2 = 2 * fltbuff[:, :ncol_a, 2 * hwin + 1] - \
                    fltbuff[:, :ncol_a, ytop - irow + hwin + 1]

                summ[irow, xleft:xright + 1], nbad = remove_bad_pixels(
                    prob2, mbuff, crow[irow - hwin], nfil, ncol_a, rdnoise_amp, gain_amp, thres)
                nfix += nbad

        print('total cosmic ray hits identified and removed: ', nfix)

        # Add info to header.
        head['bzero'] = 0.0
        head['bscale'] = 1.0
        head['exptime'] = totalexpo
        head['darktime'] = totalexpo
        # Because we do not devide the signal by the number of files the
        # read-out noise goes up by the square root of the number of files
        # sxaddpar, head, 'RDNOISE', rdnoise / sqrt(nFil) $
        # , ' noise in combined image, electrons'
        rdnoise = [rdnoise_amp]
        if (len(rdnoise) > 1):
            for amplifier in range(len(rdnoise) + 1):
                head['rdnoise{:0>1}'.format(amplifier)] = (
                    rdnoise[amplifier - 1] * np.sqrt(nfil), ' noise in combined image, electrons')
        else:
            head['rdnoise'] = (rdnoise[0] * np.sqrt(nfil),
                               ' noise in combined image, electrons')
        head['nimages'] = (nfil,
                           ' number of images summed')
        head['npixfix'] = (nfix,
                           ' pixels corrected for cosmic rays')
        head.add_history('images coadded by sumfits.pro on %s' % datetime.datetime.now())

        if (not linear):  # non-linearity was fixed. mark this in the header
            i = where(head['e_linear'] >= 0)
            if (len(i) > 0):
                head = np.array((head[0:i - 1 + 1], head[i + 1:]))
            head['e_linear'] = ('t', ' image corrected of non-linearity')

            ii = where(head['e_gain*'] >= 0)
            if (len(ii) > 0):
                for i in range(len(ii)):
                    k = ii[i]
                    head = np.array((head[0:k - 1 + 1], head[k + 1:]))
            head['e_gain'] = (1, ' image was converted to e-')

        head, _ = modeinfo(head, instrument, xr=xr, yr=yr)
        #summ = summ.swapaxes(0,1)
        return summ, head

    return summ, head
