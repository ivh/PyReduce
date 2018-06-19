"""
Combine several fits files into one master frame
"""

import datetime

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from clipnflip import clipnflip
from modeinfo_uves import modeinfo_uves as modeinfo


def gaussfit(x, y):
    """ gaussian fit to data """
    gauss = lambda x, A0, A1, A2: A0 * np.exp(-((x - A1) / A2)**2 / 2)
    popt, _ = curve_fit(gauss, x, y, p0=[max(y), 1, 1])
    return gauss(x, *popt), popt


def gaussbroad(x, y, hwhm):
    nw = len(x)
    dw = (x[-1] - x[0]) / (len(x) - 1)

    if hwhm > 5 * (x[-1] - x[0]):
        return np.full(len(x), sum(y) / len(x))

    nhalf = int(3.3972872 * hwhm / dw)
    ng = 2 * nhalf + 1				        # points in gaussian (odd!)
    # wavelength scale of gaussian
    wg = dw * (np.arange(0, ng, 1, dtype=float) - (ng - 1) / 2)
    xg = (0.83255461 / hwhm) * wg  # convenient absisca
    gpro = (0.46974832 * dw / hwhm) * \
        np.exp(-xg * xg)  # unit area gaussian w/ FWHM
    gpro = gpro / np.sum(gpro)

    # Pad spectrum ends to minimize impact of Fourier ringing.
    npad = nhalf + 2				# pad pixels on each end
    spad = np.concatenate((np.full(npad, y[0]), y, np.full(npad, y[-1]),))

    # Convolve and trim.
    sout = np.convolve(spad, gpro)  # convolve with gaussian
    sout = sout[npad: npad + nw]  # trim to original data / length
    return sout  # return broadened spectrum.


def combine_flat(files, inst_setting, **kwargs):
    """Combine several flat files into one master flat

    Parameters
    ----------
    files : list(str)
        flat files
    inst_setting : str
        instrument mode for modinfo
    bias: array(int, float), optional
        bias image to subtract from master flat (default: 0)
    exten: {int, str}, optional
        fits extension to use (default: 1)
    xr: 2-tuple(int), optional
        x range to use (default: None, i.e. whole image)
    yr: 2-tuple(int), optional
        y range to use (default: None, i.e. whole image)
    Returns
    -------
    flat, fhead
        image and header of master flat
    """

    flat, fhead = combine_frames(files, inst_setting, **kwargs)
    flat = clipnflip(flat, fhead)
    # Subtract master dark. We have to scale it by the number of Flats
    bias = kwargs.get("bias", 0)
    flat = flat - bias * len(files)  # subtract bias
    flat = flat.astype(np.float32)  # Cast to smaller type to save disk space
    return flat, fhead


def combine_bias(files, inst_setting, **kwargs):
    """
    Combine bias frames, determine read noise, reject bad pixels.
    Read noise calculation only valid if both lists yield similar noise.

    Parameters
    ----------
    files : list(str)
        bias files to combine
    inst_setting : str
        instrument mode for modinfo
    exten: {int, str}, optional
        fits extension to use (default: 1)
    xr: 2-tuple(int), optional
        x range to use (default: None, i.e. whole image)
    yr: 2-tuple(int), optional
        y range to use (default: None, i.e. whole image)
    Returns
    -------
    bias, bhead
        bias image and header
    """

    debug = kwargs.get("debug", False)

    n = len(files) // 2
    # necessary to maintain proper dimensionality, if there is just one element
    if n == 0:
        files = np.array([files, files])
        n = 1
    list1, list2 = files[:n], files[n:]

    # Lists of images.
    n1 = len(list1)
    n2 = len(list2)
    n = n1 + n2

    # Separately images in two groups.
    bias1, head1 = combine_frames(list1, inst_setting, **kwargs)
    bias1 = clipnflip(bias1 / n1, head1)

    bias2, head2 = combine_frames(list2, inst_setting, **kwargs)
    bias2 = clipnflip(bias2 / n2, head2)

    if bias1.ndim != bias2.ndim or bias1.shape[0] != bias2.shape[0]:
        raise Exception(
            'sumbias: bias frames in two lists have different dimensions')

    # Make sure we know the gain.
    head = head2
    try:
        gain = head['e_gain*']
        gain = np.array([gain[i] for i in range(len(gain))])
        gain = gain[0]
    except KeyError:
        gain = 1

    # Construct unnormalized sum.
    bias = bias1 * n1 + bias2 * n2

    # Normalize the sum.
    bias = bias / n

    # Compute noise in difference image by fitting Gaussian to distribution.
    diff = 0.5 * (bias1 - bias2)  # 0.5 like the mean...
    if np.min(diff) != np.max(diff):

        crude = np.median(np.abs(diff))  # estimate of noise
        hmin = -5.0 * crude
        hmax = +5.0 * crude
        bin_size = np.clip(2 / n, 0.5, None)
        nbins = int((hmax - hmin) / bin_size)

        h, _ = np.histogram(diff, range=(hmin, hmax), bins=nbins)
        xh = hmin + bin_size * (np.arange(0., nbins) + 0.5)

        hfit, par = gaussfit(xh, h)
        noise = abs(par[2])  # noise in diff, bias

        # Determine where wings of distribution become significantly non-Gaussian.
        contam = (h - hfit) / np.sqrt(np.clip(hfit, 1, None))
        imid = np.where(abs(xh) < 2 * noise)
        consig = np.std(contam[imid])

        smcontam = gaussbroad(xh, contam, 0.1 * noise)
        igood = np.where(smcontam < 3 * consig)
        gmin = np.min(xh[igood])
        gmax = np.max(xh[igood])

        # Find and fix bad pixels.
        ibad = np.where((diff <= gmin) | (diff >= gmax))
        nbad = len(ibad[0])

        bias[ibad] = np.clip(bias1[ibad], None, bias2[ibad])

        # Compute read noise.
        biasnoise = gain * noise
        bgnoise = biasnoise * np.sqrt(n)

        # Print diagnostics.
        print('change in bias between image sets= %f electrons' %
              (gain * par[1],))
        print('measured background noise per image= %f' % bgnoise)
        print('background noise in combined image= %f' % biasnoise)
        print('fixing %i bad pixels' % nbad)

        if debug:
            # Plot noise distribution.
            plt.subplot(211)
            plt.plot(xh, h)
            plt.plot(xh, hfit, c='r')
            plt.title('noise distribution')
            plt.axvline(gmin, c='b')
            plt.axvline(gmax, c='b')

            # Plot contamination estimation.
            plt.subplot(212)
            plt.plot(xh, contam)
            plt.plot(xh, smcontam, c='r')
            plt.axhline(3 * consig, c='b')
            plt.axvline(gmin, c='b')
            plt.axvline(gmax, c='b')
            plt.title('contamination estimation')
            plt.show()
    else:
        diff = 0
        biasnoise = 1.
        nbad = 0

    obslist = files[0][0]
    for i in range(1, len(files)):
        obslist = obslist + ' ' + files[i][0]

    try:
        del head['tapelist']
    except KeyError:
        pass

    head['bzero'] = 0.0
    head['bscale'] = 1.0
    head['obslist'] = obslist
    head['nimages'] = (n, 'number of images summed')
    head['npixfix'] = (nbad, 'pixels corrected for cosmic rays')
    head['bgnoise'] = (biasnoise, 'noise in combined image, electrons')
    bias = bias.astype(np.float32)
    return bias, head


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
    b = np.sum(b, axis=0)
    return b, nbad


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


def combine_frames(files, instrument, **kwargs):
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

    exten = kwargs.get("exten", 1)
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
    # Loop through files in list, grabbing and parsing FITS headers.
    # length of longest filename
    filename_length = np.max([len(l) for l in files])
    print('  file' + ' ' * (filename_length - 3) +
          'obs cols rows  object  exposure')

    fname = files[0]
    hdu = fits.open(fname)
    head2 = load_header(hdu)
    head2, kw = modeinfo(head2, instrument, xr=xr,
                         yr=yr, gain=True, readn=True, orient=True)

    # check if we deal with multiple amplifiers
    n_ampl = head2.get('e_ampl', 1)

    # section(s) of the detector to process
    xlow = np.array(list(head2['e_xlo*'].values()), ndmin=1)
    xhigh = np.array(list(head2['e_xhi*'].values()), ndmin=1)
    ylow = np.array(list(head2['e_ylo*'].values()), ndmin=1)
    yhigh = np.array(list(head2['e_yhi*'].values()), ndmin=1)

    gain = np.array(list(head2["e_gain*"].values()), ndmin=1)
    rdnoise = np.array(list(head2["e_readn*"].values()), ndmin=1)

    nfix = 0  # init fixed pixel counter

    ncol_old = head2['naxis1']  # columns
    nrow_old = head2['naxis2']  # rows
    bias2 = np.zeros((nrow_old, ncol_old))  # init r*4 output image array

    # check if non-linearity correction
    linear = head2.get("e_linear", True)
    if not linear:
        pref = head2['e_prefmo']

    # TODO: what happens for several amplifiers?
    # outer loop through amplifiers (note: 1,2 ...)
    for amplifier in range(n_ampl):
        heads = [fits.open(f) for f in files]
        heads = [load_header(h) for h in heads]
        heads = [modeinfo(h, instrument, readn=True, gain=True, orient=True)[
            0] for h in heads]

        # Sanity Check
        ncol = np.array([h['naxis1'] for h in heads])
        nrow = np.array([h['naxis2'] for h in heads])
        if np.any(ncol != ncol_old) or np.any(nrow != nrow_old):
            raise Exception('Not all files have the same dimensions')

        bscale = np.array([h.get("bscale", 1) for h in heads])
        bzero = np.array([h.get("bzero", 0) for h in heads])
        obj = [h["object"] for h in heads]
        exposure = [h["exptime"] for h in heads]
        totalexpo = sum(exposure)

        # loop though files
        for ifile, fname, nc, nr, ob, exp in zip(range(len(files)), files, ncol, nrow, obj, exposure):
            print(fname, ifile, nc, nr, '  ', ob, exp)  # summarize header info

        xleft = xlow[amplifier]
        xright = xhigh[amplifier]
        ybottom = ylow[amplifier]
        ytop = yhigh[amplifier]

        gain_amp = gain[amplifier]
        rdnoise_amp = rdnoise[amplifier]

        orient = heads[0]['e_orient']
        if orient == 0 or orient == 2 or orient == 5 or orient == 7:
            raise NotImplementedError  # TODO implement, or better combine with other orientation
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
