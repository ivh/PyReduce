"""
Combine several fits files into one master frame
"""

import datetime

import astropy.io.fits as fits
import numpy as np

from modeinfo_uves import modeinfo_uves as modeinfo


def remove_bad_pixels(p, buff, row, nfil, ncol_a, rdnoise_amp, gain_amp, thresh):
    """
    find and remove bad pixels

    Parameters
    ----------
    p : array(float)
        probabilities
    buff : array(int)
        image buffer
    row : int
        current row
    nfil : int
        file number
    ncol_a : int
        number of columns
    rdnoise_amp : float
        readnoise of current amplifier
    gain_amp : float
        gain of current amplifier
    thresh : float
        threshold for bad pixels
    Returns
    -------
    array(int)
        input buff, with bad pixels removed
    """

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


def calc_filwt(buffer):
    """
    Construct a probability function based on buffer data.

    Parameters
    ----------
    buffer : array(float)
        buffer
    Returns
    -------
    array(float)
        probabilities
    """

    # boxcar average for irow
    filwt = np.sum(buffer, axis=2, dtype=np.float32)
    # norm for probability
    tot_filwt = np.sum(filwt, axis=0, dtype=np.float32)
    inorm = tot_filwt > 0
    tot_filwt, tot_filwt.shape = tot_filwt[inorm], tot_filwt.shape
    filwt[:, inorm] = filwt[:, inorm] / tot_filwt
    return filwt


def load_header(hdulist, exten=1):
    """
    load and combine primary header with extension header

    Parameters
    ----------
    hdulist : list(hdu)
        list of hdu, usually from fits.open
    exten : int, optional
        extension to use in addition to primary (default: 1)

    Returns
    -------
    header
        combined header, extension first
    """

    head = hdulist[exten].header
    head.extend(hdulist[0].header, strip=False)
    return head


def combine_frames(files, instrument, exten=1, **kwargs):
    """
    Subroutine to correct cosmic rays blemishes, while adding otherwise
    similar images.

    Parameters
    ----------
    files : list(str)
        list of fits files to combine
    instrument : str
        instrument id for modinfo
    exten : int, optional
        fits extension to load (default: 1)
    thresh : float, optional
        threshold for bad pixels (default: 3.5)
    hwin : int, optional
        horizontal window size (default: 50)
    mask : array(bool), optional
        mask for the fits image, not supported yet (default: None)
    xr : int, optional
        xrange (default: None)
    yr : int, optional
        yrange (default: None)
    debug : bool, optional
        show debug plot of noise distribution (default: False)
    """

    debug = kwargs.get("debug")
    thres = kwargs.get("thres", 3.5)
    hwin = kwargs.get("hwin", 50)
    mask = kwargs.get("mask")
    xr = kwargs.get("xr")
    yr = kwargs.get("yr")

    # Verify sensibility of passed parameters.
    files = np.lib.arraysetops.unique(files)

# ===========================================================================
    if len(files) < 2:  # true: only one image
        bias2 = fits.open(files[0])
        head2 = load_header(bias2)
        head2 = modeinfo(head2, instrument, xr=xr, yr=yr)
        return bias2[exten].data, head2
# ===========================================================================
    elif len(files) == 2:  # ;true: only two images
        bias1 = fits.open(files[0])
        head1 = load_header(bias1)
        head1, exp1 = modeinfo(head1, instrument, time=True)
        exp1 = exp1["time"]

        bias2 = fits.open(files[1])
        head2 = load_header(bias2)
        head2, exp2 = modeinfo(head2, instrument, time=True,
                               readn=True, orient=True, xr=xr, yr=yr)
        exp2, rdnoise, orient = exp2["time"], exp2["readn"], exp2["orient"]

        bias2 = bias2[exten].data + bias1[exten].data
        totalexpo = exp1 + exp2

        # Add info to header.
        head2['bzero'] = 0.0
        head2['bscale'] = 1.0
        head2['exptime'] = totalexpo
        head2['darktime'] = totalexpo
        # Because we do not devide the signal by the number of files the
        # read-out noise goes up by the square root of the number of files
        head2['rdnoise'] = (rdnoise * np.sqrt(len(files)),
                            'noise in combined image, electrons')
        head2['nimages'] = (len(files), 'number of images summed')
        head2.add_history('images coadded by sumfits.pro on %s' %
                          datetime.datetime.now())
        head2, _ = modeinfo(head2, instrument, xr=xr, yr=yr)
        return bias2, head2

# ===========================================================================
# Initialize header information lists (one entry per file).
    bzero = np.zeros(len(files))  # bzero list (0.0=default)
    bscale = np.full(len(files), 1.)  # bscale list (1.0=default)
    totalexpo = 0  # init total exposure time

    # Loop through files in list, grabbing and parsing FITS headers.
    # length of longest filename
    filename_length = np.max([len(l) for l in files])
    print('  file' + ' ' * (filename_length - 3) +
          'obs cols rows  object  exposure')

    fname = files[0]
    hdu = fits.open(fname)
    head2 = load_header(hdu)
    head2, kw = modeinfo(head2, instrument, gain=True,
                         orient=True, xr=xr, yr=yr)
    gain, orient = kw["gain"], kw["orient"]

    ncol_old = head2['naxis1']  # columns
    nrow_old = head2['naxis2']  # rows

    # check if we deal with multiple amplifiers
    try:
        n_ampl = head2['e_ampl']
    except KeyError:
        n_ampl = 1
    if n_ampl == 1:
        # section(s) of the detector to process
        xlow = [head2['e_xlo']]
        xhigh = [head2['e_xhi']]
        ylow = [head2['e_ylo']]
        yhigh = [head2['e_yhi']]
    else:
        # section(s) of the detector to process
        xlow = [i for i in head2['e_xlo*'].values()]
        xhigh = [i for i in head2['e_xhi*'].values()]
        ylow = [i for i in head2['e_ylo*'].values()]
        yhigh = [i for i in head2['e_yhi*'].values()]

    nfix = 0  # init fixed pixel counter
    bias2 = np.zeros((nrow_old, ncol_old))  # init r*4 output image array

    # check if non-linearity correction
    linear = head2.get("e_linear", True)
    if not linear:
        pref = head2['e_prefmo']

    # TODO: what happens for several amplifiers?
    # outer loop through amplifiers (note: 1,2 ...)
    for amplifier in range(n_ampl):
        for ifile, fname in enumerate(files):  # loop though files

            bias1 = fits.open(fname)
            head2 = load_header(bias1)
            head2, kw = modeinfo(head2, instrument, gain=True, orient=True)
            gain, orient = kw["gain"], kw["orient"]

            # Sanity check for the files
            ncol = head2['naxis1']  # columns
            nrow = head2['naxis2']  # rows
            if nrow != nrow_old or ncol != ncol_old:
                raise ValueError('sumfits: file ' + fname +
                                 ' has different dimensions')

            # load some header data for each file
            bscale[ifile] = head2.get("bscale", 1)  # pixel scale
            bzero[ifile] = head2.get('bzero', 0)  # pixel offset
            obj = head2['object']  # get object id
            exposure = head2['EXPTIME']
            totalexpo += exposure  # accumulate exposure times

            print(fname, ifile, ncol, nrow, '  ', obj,
                  exposure)  # summarize header info

        xleft = xlow[amplifier]
        xright = xhigh[amplifier]
        ybottom = ylow[amplifier]
        ytop = yhigh[amplifier]

        if n_ampl > 1:
            gain_amp = gain[amplifier]
            rdnoise_amp = rdnoise[amplifier]
        else:
            gain_amp = head2['e_gain']
            rdnoise_amp = head2['e_readn']

        orient = head2['e_orient']
        if orient == 0 or orient == 2 or orient == 5 or orient == 7:
            raise NotImplementedError #TODO implement, or better combine with other orientation
            ncol_a = xright - xleft + 1
            nrow_a = ytop - ybottom + 1

            block = fits_read(files, exten=1)
            block = block[:, :, xleft:xright + 1]

            if (ybottom > 0):  # skip ybottom rows
                for i_row in range(0, ybottom - 1 + 1, 1):
                    for ifile in np.arange(0, nfil - 1 + 1, 1):  # loop though files
                        block = readu(lunit[ifile], dtype=idl_type, n=ncol)

            for i_row in range(ybottom, ytop + 1, 1):  # loop through rows
                if i_row % 100 == 0 and i_row > 0:  # report status
                    message(strtrim(i_row, 2) + ' rows processed - ' +
                            strtrim(string(nfix), 2) + ' pixels fixed so far.', info=True)
                for ifile in np.arange(0, nfil - 1 + 1, 1):  # loop though files
                    block = readu(lunit[ifile], dtype=idl_type, n=ncol)
                    blockf = block * bscale[ifile] + bzero[ifile]
                    if (not linear):  # linearity needs fixing
                        blockf = call_function(
                            'nonlinear_' + pref, blockf, head2, amplifier=amplifier, gain=gain_amp)
                        pass
                    if (keyword_set(mask)):
                        img_buffer[ifile, :] = (
                            blockf * mask[i_row, :])[xleft:xright + 1]
                    else:
                        img_buffer[ifile, :] = blockf[xleft:xright + 1]

                # Construct a probability function based on mbuff data.
                for icol in range(hwin, ncol_a - hwin):
                    # filwt = total(mBuff[iCol-hwin:iCol+hwin,*], 1)
                    filwt = median(
                        img_buffer[:, icol - hwin:icol + hwin + 1], dimension=1)

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
                        rat = reform(img_buffer[iprob, icol]) / pr[iprob]
                        amp = (total(rat) - min(rat) - max(rat)) / (nfil - 2)
                        mfit[iprob, icol] = amp * pr[iprob]
                    pass

                # Construct noise model.
                predsig = np.sqrt(rdnoise_amp**2 + abs(mfit / gain_amp))

                # Identify outliers.
                ibad, nbad = where(img_buffer - mfit > thresh *
                                   predsig, count='nbad')

                # Debug plot.
                if keyword_set(debug) and i_row % 10 == 0:
                    plot(img_buffer - mfit, xsty=3, ysty=3, ps=3, ytit='data - model  (adu)', tit='row = ' +
                         strtrim(i_row, 2) + ',  threshold = ' + string(thresh, form='(f9.4)'))
                    oplot(thresh * predsig, co=4)
                    oplot(-thresh * predsig, co=4)
                    if nbad > 0:
                        oplot(ibad, img_buffer[ibad] - mfit[ibad],
                              ps=2, syms=1.0, co=2)
                    print('push space to continue...')
                    print('')

                # Fix bad pixels, if any.
                if nbad > 0:
                    img_buffer[ibad] = mfit[ibad]
                    nfix = nfix + nbad

                # Construct the summed FITS.
                buff = total(img_buffer, 2)  # / nfil
                bias2[i_row, xleft:xright + 1] = buff
        elif orient == 1 or orient == 3 or orient == 4 or orient == 6:

            ncol_a = xright - xleft + 1
            m_row = 2 * hwin + 1  # of rows in the fifo buffer

            if ytop - ybottom + 1 < m_row:
                raise ValueError(
                    'sumfits: the number of rows should be larger than 2 * win = %i' % m_row)

            # The same as mbuff
            block = np.array([fits.open(f)[exten].data for f in files])
            block = block[:, :, xleft:xright + 1]
            # Initial Window
            img_buffer = np.swapaxes(block[:, :m_row, :], 1, 2)
            # For extrapolation later
            extra_buffer = np.swapaxes(
                block[:, :hwin, :], 1, 2).astype(np.float32)

            float_buffer = np.swapaxes(
                block[:, :m_row + 1, :], 1, 2).astype(np.float32)
            c_row = np.arange(hwin, ytop - ybottom + 1) % (m_row)
            j_row = np.arange(2 * hwin, ytop - ybottom + 1) % (m_row)

            # TODO Remove the loop, should be possible
            for i_row in range(ybottom + hwin, ytop - hwin + 1):
                count = i_row - ybottom - hwin
                if (i_row) % 100 == 0:
                    print(i_row, ' rows processed - ',
                          nfix, ' pixels fixed so far')

                # This simulates reading the file line by line
                img_buffer[:, :, j_row[count]] = block[:, m_row + count - 1, :]
                filwt = calc_filwt(img_buffer)

                if i_row <= ybottom + 2 * hwin:
                    # for 1st special case: rows lesser than ybottom + hwin
                    float_buffer[:, :, i_row - ybottom - hwin] = np.copy(filwt)
                if i_row >= ytop - 2 * hwin:
                    # for 2nd special case: rows greater than ytop - hwin
                    float_buffer[:, :, i_row + 3 *
                                 hwin - ytop + 1] = np.copy(filwt)

                bias2[i_row, xleft:xright + 1], nbad = remove_bad_pixels(
                    filwt, img_buffer, c_row[count], len(files), ncol_a, rdnoise_amp, gain_amp, thres)
                nfix += nbad

            # 1st special case: rows less than hwin from the 0th row
            for i_row in range(ybottom + hwin + 1, ybottom + 2 * hwin + 1):
                lrow = 2 * hwin - i_row + ybottom
                prob2 = 2 * float_buffer[:, :ncol_a, 0] - \
                    float_buffer[:, :ncol_a, i_row - ybottom - hwin]

                bias2[ybottom + lrow, xleft:xright + 1], nbad = remove_bad_pixels(
                    prob2, extra_buffer, lrow, len(files), ncol_a, rdnoise_amp, gain_amp, thres)
                nfix += nbad

            # 2nd special case: rows greater than ytop-hwin
            for i_row in range(ytop - hwin + 1, ytop + 1):
                prob2 = 2 * float_buffer[:, :ncol_a, 2 * hwin + 1] - \
                    float_buffer[:, :ncol_a, ytop - i_row + hwin + 1]

                bias2[i_row, xleft:xright + 1], nbad = remove_bad_pixels(
                    prob2, img_buffer, c_row[i_row - hwin], len(files), ncol_a, rdnoise_amp, gain_amp, thres)
                nfix += nbad

        print('total cosmic ray hits identified and removed: ', nfix)

        # Add info to header.
        head2['bzero'] = 0.0
        head2['bscale'] = 1.0
        head2['exptime'] = totalexpo
        head2['darktime'] = totalexpo
        # Because we do not devide the signal by the number of files the
        # read-out noise goes up by the square root of the number of files
        rdnoise = [rdnoise_amp]

        if len(rdnoise) > 1:
            for amplifier in range(len(rdnoise) + 1):
                head2['rdnoise{:0>1}'.format(amplifier)] = (
                    rdnoise[amplifier - 1] * np.sqrt(len(files)), ' noise in combined image, electrons')
        else:
            head2['rdnoise'] = (rdnoise[0] * np.sqrt(len(files)),
                                ' noise in combined image, electrons')
        head2['nimages'] = (len(files),
                            ' number of images summed')
        head2['npixfix'] = (nfix,
                            ' pixels corrected for cosmic rays')
        head2.add_history('images coadded by sumfits.pro on %s' %
                          datetime.datetime.now())

        if not linear:  # non-linearity was fixed. mark this in the header
            raise NotImplementedError()  # TODO Nonlinear
            i = np.where(head2['e_linear'] >= 0)
            head2[i] = np.array((head2[0:i - 1 + 1], head2[i + 1:]))
            head2['e_linear'] = ('t', ' image corrected of non-linearity')

            ii = np.where(head2['e_gain*'] >= 0)
            if len(ii[0]) > 0:
                for i in range(len(ii[0])):
                    k = ii[i]
                    head2 = np.array((head2[0:k - 1 + 1], head2[k + 1:]))
            head2['e_gain'] = (1, ' image was converted to e-')

        head2, _ = modeinfo(head2, instrument, xr=xr, yr=yr)

    return bias2, head2
