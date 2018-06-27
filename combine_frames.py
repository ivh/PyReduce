"""
Combine several fits files into one master frame

Used to create master bias and master flat
"""

import datetime

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import median_filter
from scipy.optimize import curve_fit

from clipnflip import clipnflip
from modeinfo_uves import modeinfo_uves as modeinfo


def load_fits(fname, extension, instrument, header_only=False, **kwargs):
    """
    load a fits file
    merge primary and extension header
    fix reduce specific header values
    mask array
    """
    data = fits.open(fname)
    head = load_header(data)
    head, kw = modeinfo(head, instrument, **kwargs)

    if header_only:
        return head, kw

    data = data[extension].data
    data = np.ma.masked_array(data, mask=kwargs.get("mask"))
    return data, head, kw


def load_header(hdulist, extension=1):
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

    head = hdulist[extension].header
    head.extend(hdulist[0].header, strip=False)
    return head


def gaussfit(x, y):
    """
    Fit a simple gaussian to data

    gauss(x, a, mu, sigma) = a * exp(-z**2/2)
    with z = (x - mu) / sigma

    Parameters
    ----------
    x : array(float)
        x values
    y : array(float)
        y values
    Returns
    -------
    gauss(x), parameters
        fitted values for x, fit paramters (a, mu, sigma)
    """

    gauss = lambda x, A0, A1, A2: A0 * np.exp(-((x - A1) / A2) ** 2 / 2)
    popt, _ = curve_fit(gauss, x, y, p0=[max(y), 1, 1])
    return gauss(x, *popt), popt


def gaussbroad(x, y, hwhm):
    """
    Apply gaussian broadening to x, y data with half width half maximum hwhm

    Parameters
    ----------
    x : array(float)
        x values
    y : array(float)
        y values
    hwhm : float > 0
        half width half maximum
    Returns
    -------
    array(float)
        broadened y values
    """

    # alternatively use:
    # from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
    # but that doesn't have an x coordinate

    nw = len(x)
    dw = (x[-1] - x[0]) / (len(x) - 1)

    if hwhm > 5 * (x[-1] - x[0]):
        return np.full(len(x), sum(y) / len(x))

    nhalf = int(3.3972872 * hwhm / dw)
    ng = 2 * nhalf + 1  # points in gaussian (odd!)
    # wavelength scale of gaussian
    wg = dw * (np.arange(0, ng, 1, dtype=float) - (ng - 1) / 2)
    xg = (0.83255461 / hwhm) * wg  # convenient absisca
    gpro = (0.46974832 * dw / hwhm) * np.exp(-xg * xg)  # unit area gaussian w/ FWHM
    gpro = gpro / np.sum(gpro)

    # Pad spectrum ends to minimize impact of Fourier ringing.
    npad = nhalf + 2  # pad pixels on each end
    spad = np.concatenate((np.full(npad, y[0]), y, np.full(npad, y[-1])))

    # Convolve and trim.
    sout = np.convolve(spad, gpro)  # convolve with gaussian
    sout = sout[npad : npad + nw]  # trim to original data / length
    return sout  # return broadened spectrum.


def running_median(arr, size):
    """Calculate the running median of a 2D sequence

    Parameters
    ----------
    seq : 2d array [n, l]
        n datasets of length l
    size : int
        number of elements to consider for each median
    Returns
    -------
    2d array [n, l-size]
        running median
    """

    ret = np.array([median_filter(s, size=size, mode="constant") for s in arr])
    m = size // 2
    return ret[:, m:-m]


def running_sum(arr, size):
    """Calculate the running sum over the 2D sequence

    Parameters
    ----------
    arr : array[n, l]
        sequence to calculate running sum over, n datasets of length l
    size : int
        number of elements to sum
    Returns
    -------
    2D array
        running sum
    """

    ret = np.cumsum(arr, axis=1)
    ret[:, size:] -= ret[:, :-size]
    return ret[:, size - 1 :]


def calculate_probability(buffer, window, method="sum"):
    """
    Construct a probability function based on buffer data.

    Parameters
    ----------
    buffer : array(float)
        buffer
    window : int
        size of the running window
    method : {"sum", "median"}, optional
        which method to use to average the probabilities (default: "sum")
        "sum" is much faster, but "median" is more resistant to outliers
    Returns
    -------
    array(float)
        probabilities
    """

    # Take the median/sum for each file
    if method == "median":
        # Running median is slow
        weights = running_median(buffer, 2 * window + 1)
        sum_of_weights = np.mean(weights, axis=0)
    if method == "sum":
        # Running sum is fast
        weights = running_sum(buffer, 2 * window + 1)
        sum_of_weights = np.sum(weights, axis=0)

    # norm probability
    weights = np.where(sum_of_weights > 0, weights / sum_of_weights, weights)
    return weights


def fix_bad_pixels(probability, buffer, readnoise, gain, threshold):
    """
    find and fix bad pixels

    Parameters
    ----------
    probability : array(float)
        probabilities
    buffer : array(int)
        image buffer
    readnoise : float
        readnoise of current amplifier
    gain : float
        gain of current amplifier
    threshold : float
        sigma threshold between observation and fit for bad pixels
    Returns
    -------
    array(int)
        input buffer, with bad pixels fixed
    """
    # Fit signal
    ratio = np.where(probability > 0, buffer / probability, 0.)
    amplitude = (
        np.sum(ratio, axis=0) - np.min(ratio, axis=0) - np.max(ratio, axis=0)
    ) / (buffer.shape[0] - 2)

    fitted_signal = np.where(probability > 0, amplitude[None, :] * probability, 0)
    predicted_noise = np.sqrt(readnoise ** 2 + (fitted_signal / gain))

    # Identify outliers
    badpixels = buffer - fitted_signal > threshold * predicted_noise
    nbad = len(np.nonzero(badpixels.flat)[0])

    # Construct the summed flat
    corrected_signal = np.where(badpixels, fitted_signal, buffer)
    corrected_signal = np.sum(corrected_signal, axis=0)
    return corrected_signal, nbad


def combine_frames(files, instrument, extension=1, threshold=3.5, window=50, **kwargs):
    """
    Subroutine to correct cosmic rays blemishes, while adding otherwise
    similar images.

    combine_frames co-adds a group of FITS files with 2D images of identical dimensions.
    In the process it rejects cosmic ray, detector defects etc. It is capable of
    handling images that have strip pattern (e.g. echelle spectra) using the REDUCE
    modinfo conventions to figure out image orientation and useful pixel ranges.
    It can handle many frames. Special cases: 1 file in the list (the input is returned as output)
    and 2 files (straight sum is returned).

    If the image orientation is not predominantly vertical, the image is rotated 90 degrees (and rotated back afterwards).

    Open all FITS files in the list.
    Loop through the rows. Read next row from each file into a row buffer mBuff[nCol, nFil].
    Optionally correct the data for non-linearity.

    calc_probability
    ----------------
    Go through the row creating "probability" vector. That is for column iCol take the median of
    the part of the row mBuff[iCol-win:iCol+win,iFil] for each file and divide these medians by the
    mean of them computed across the stack of files. In other words:
    >>> filwt[iFil] = median(mBuff[iCol-win:iCol+win,iFil])
    >>> norm_filwt = mean(filwt)
    >>> prob[iCol,iFil] = (norm_filtwt>0)?filwt[iCol]/norm_filwt:filwt[iCol]

    This is done for all iCol in the range of [win:nCol-win-1]. It is then linearly extrapolated to
    the win zones of both ends. E.g. for iCol in [0:win-1] range:
    >>> prob[iCol,iFil]=2*prob[win,iFil]-prob[2*win-iCol,iFil]

    For the other end ([nCol-win:nCol-1]) it is similar:
    >>> prob[iCol,iFil]=2*prob[nCol-win-1,iFil]-prob[2*(nCol-win-1)-iCol,iFil]

    fix_bad_pixels
    -----------------
    Once the probailities are constructed we can do the fitting, measure scatter and detect outliers.
    We ignore negative or zero probabilities as it should not happen. For each iCol with (some)
    positive probabilities we compute tha ratios of the original data to the probabilities and get
    the mean amplitude of these ratios after rejecting extreme values:
    >>> ratio = mBuff[iCol,iFil]/prob[iCol,iFil]
    >>> amp = (total(ratio)-min(ratio)-max(ratio))/(nFil-2)
    >>> mFit[iCol,iFil] = amp*prob[iCol,iFil]

    Note that for iFil whereprob[iCol,iFil] is zero we simply set mFit to zero. The scatter (noise)
    consists readout noise and shot noise of the model (fit) co-added in quadratures:
    >>> sig=sqrt(rdnoise*rdnoise + abs(mFit[iCol,iFil]/gain))

    and the outliers are defined as:
    >>> iBad=where(mBuff-mFit gt thresh*sig)

    >>> Bad values are replaced from the fit:
    >>> mBuff[iBad]=mFit[iBad]

    and mBuff is summed across the file dimension to create an output row.

    Parameters
    ----------
    files : list(str)
        list of fits files to combine
    instrument : str
        instrument id for modinfo
    extension : int, optional
        fits extension to load (default: 1)
    threshold : float, optional
        threshold for bad pixels (default: 3.5)
    window : int, optional
        horizontal window size (default: 50)
    mask : array(bool), optional
        mask for the fits image (default: None)
    xr : int, optional
        xrange (default: None)
    yr : int, optional
        yrange (default: None)
    debug : bool, optional
        show debug plot of noise distribution (default: False)
    dtype : np.dtype, optional
        datatype of the combined image (default float32)

    Returns
    -------
    combined_data, header
        combined image data, header
    """

    DEBUG_NROWS = 1000  # print status update every DEBUG_NROWS rows (if debug is True)
    debug = kwargs.get("debug", False)
    dtype = kwargs.get("dtype", np.float32)

    # summarize file info
    print("Files:")
    for i, fname in zip(range(len(files)), files):
        print(i, fname)

    # Only one image
    if len(files) < 2:
        result, head, _ = load_fits(files[0], extension, instrument, **kwargs)
        return result, head
    # Two images
    elif len(files) == 2:
        bias1, _, kw = load_fits(files[0], extension, instrument, **kwargs)
        exp1 = kw["time"]

        bias2, head, kw = load_fits(files[0], extension, instrument, **kwargs)
        exp2, readnoise = kw["time"], kw["readn"]

        result = bias2 + bias1
        total_exposure_time = exp1 + exp2
        readnoise = np.atleast_1d(readnoise)
        n_fixed = 0
        linear = head.get("e_linear", True)
    # More than two images
    else:
        # Get information from headers

        # TODO: assume all values are the same in all the headers
        heads = [
            load_fits(f, extension, instrument, header_only=True, **kwargs)[0]
            for f in files
        ]
        head = heads[0]

        # if sizes vary, it will show during loading of the data
        n_columns = head["naxis1"]
        n_rows = head["naxis2"]

        # check if we deal with multiple amplifiers
        n_amplifier = head.get("e_ampl", 1)
        # check orientation of the image
        # orient 0, 2, 5, 7: orders are horizontal
        # orient 1, 3, 4, 6: orders are vertical
        orientation = head["e_orient"]
        # check if non-linearity correction
        linear = head.get("e_linear", True)

        # section(s) of the detector to process, x_low, x_high, y_low, y_high
        # head["e_xlo*"] will find all entries with * as a wildcard
        # we also ensure that we will have one dimensional arrays (not just the value)
        x_low = [c[1] for c in head["e_xlo*"].cards]
        x_high = [c[1] for c in head["e_xhi*"].cards]
        y_low = [c[1] for c in head["e_ylo*"].cards]
        y_high = [c[1] for c in head["e_yhi*"].cards]

        gain = np.array(list(head["e_gain*"].values()), ndmin=1)
        readnoise = np.array(list(head["e_readn*"].values()), ndmin=1)
        total_exposure_time = sum([h["exptime"] for h in heads])

        result = np.zeros((n_rows[0], n_columns[0]), dtype=dtype)  # the combined image
        n_fixed = 0  # number of fixed pixels

        # Load all image data
        data = np.empty((len(files), n_rows[0], n_columns[0]), dtype="uint16")
        for i, f in enumerate(files):
            data[i] = fits.open(f)[extension].data

        if orientation in [1, 3, 4, 6]:
            data = np.rot90(data, k=-1, axes=(1, 2))
            result = np.rot90(result, k=-1)
            x_low, x_high, y_low, y_high = y_low, y_high, x_low, x_high

        # For several amplifiers, different sections of the image are set
        # One for each amplifier, each amplifier is treated seperately
        for amplifier in range(n_amplifier):
            x_left = x_low[amplifier]
            x_right = x_high[amplifier]
            y_bottom = y_low[amplifier]
            y_top = y_high[amplifier]

            gain_amp = gain[amplifier]
            readnoise_amp = readnoise[amplifier]

            buffer = np.zeros((len(files), x_right - x_left))
            probability = np.zeros((len(files), x_right - x_left))

            # for each row
            for row in range(y_bottom, y_top):
                if debug and (row) % DEBUG_NROWS == 0:
                    print(row, " rows processed - ", n_fixed, " pixels fixed so far")

                # load current row
                buffer[:, :] = data[:, row, x_left:x_right]

                # Calculate probabilities
                probability[:, window:-window] = calculate_probability(buffer, window)

                # extrapolate to edges
                probability[:, :window] = (
                    2 * probability[:, window][:, None]
                    - probability[:, 2 * window : window : -1]
                )
                probability[:, -window:] = (
                    2 * probability[:, -window - 1][:, None]
                    - probability[:, -window - 1 : -2 * window - 1 : -1]
                )

                # fix bad pixels
                result[row, x_left:x_right], n_bad = fix_bad_pixels(
                    probability, buffer, readnoise_amp, gain_amp, threshold
                )
                n_fixed += n_bad

        # rotate back
        if orientation in [1, 3, 4, 6]:
            result = np.rot90(result, 1)

        print("total cosmic ray hits identified and removed: ", n_fixed)

    # Add info to header.
    head["bzero"] = 0.0
    head["bscale"] = 1.0
    head["exptime"] = total_exposure_time
    head["darktime"] = total_exposure_time
    # Because we do not divide the signal by the number of files the
    # read-out noise goes up by the square root of the number of files

    for n_amp, rdn in enumerate(readnoise):
        head["rdnoise{:0>1}".format(n_amp + 1)] = (
            rdn * np.sqrt(len(files)),
            " noise in combined image, electrons",
        )

    head["nimages"] = (len(files), " number of images summed")
    head["npixfix"] = (n_fixed, " pixels corrected for cosmic rays")
    head.add_history(
        "images coadded by combine_frames.py on %s" % datetime.datetime.now()
    )

    if not linear:  # non-linearity was fixed. mark this in the header
        raise NotImplementedError()  # TODO Nonlinear
        i = np.where(head["e_linear"] >= 0)
        head[i] = np.array((head[0 : i - 1 + 1], head[i + 1 :]))
        head["e_linear"] = ("t", " image corrected of non-linearity")

        ii = np.where(head["e_gain*"] >= 0)
        if len(ii[0]) > 0:
            for i in range(len(ii[0])):
                k = ii[i]
                head = np.array((head[0 : k - 1 + 1], head[k + 1 :]))
        head["e_gain"] = (1, " image was converted to e-")

    return result, head


def combine_flat(files, instrument, extension=1, **kwargs):
    """
    Combine several flat files into one master flat

    Parameters
    ----------
    files : list(str)
        flat files
    instrument : str
        instrument mode for modinfo
    extension: {int, str}, optional
        fits extension to use (default: 1)
    bias: array(int, float), optional
        bias image to subtract from master flat (default: 0)
    
    xr: 2-tuple(int), optional
        x range to use (default: None, i.e. whole image)
    yr: 2-tuple(int), optional
        y range to use (default: None, i.e. whole image)
    dtype : np.dtype, optional
        datatype of the combined bias frame (default: float32) 
    Returns
    -------
    flat, fhead
        image and header of master flat
    """

    flat, fhead = combine_frames(files, instrument, extension, **kwargs)
    flat = clipnflip(flat, fhead)
    # Subtract master dark. We have to scale it by the number of Flats
    bias = kwargs.get("bias", 0)
    flat = flat - bias * len(files)  # subtract bias, if passed
    return flat, fhead


def combine_bias(files, instrument, extension=1, **kwargs):
    """
    Combine bias frames, determine read noise, reject bad pixels.
    Read noise calculation only valid if both lists yield similar noise.

    Parameters
    ----------
    files : list(str)
        bias files to combine
    instrument : str
        instrument mode for modinfo
    extension : {int, str}, optional
        fits extension to use (default: 1)
    xr : 2-tuple(int), optional
        x range to use (default: None, i.e. whole image)
    yr : 2-tuple(int), optional
        y range to use (default: None, i.e. whole image)
    dtype : np.dtype, optional
        datatype of the combined bias frame (default: float32)
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
    bias1, head1 = combine_frames(list1, instrument, extension, **kwargs)
    bias1 = clipnflip(bias1 / n1, head1)

    bias2, head2 = combine_frames(list2, instrument, extension, **kwargs)
    bias2 = clipnflip(bias2 / n2, head2)

    if bias1.ndim != bias2.ndim or bias1.shape[0] != bias2.shape[0]:
        raise Exception("sumbias: bias frames in two lists have different dimensions")

    # Make sure we know the gain.
    head = head2
    try:
        gain = head["e_gain*"]
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
        print("change in bias between image sets= %f electrons" % (gain * par[1],))
        print("measured background noise per image= %f" % bgnoise)
        print("background noise in combined image= %f" % biasnoise)
        print("fixing %i bad pixels" % nbad)

        if debug:
            # Plot noise distribution.
            plt.subplot(211)
            plt.plot(xh, h)
            plt.plot(xh, hfit, c="r")
            plt.title("noise distribution")
            plt.axvline(gmin, c="b")
            plt.axvline(gmax, c="b")

            # Plot contamination estimation.
            plt.subplot(212)
            plt.plot(xh, contam)
            plt.plot(xh, smcontam, c="r")
            plt.axhline(3 * consig, c="b")
            plt.axvline(gmin, c="b")
            plt.axvline(gmax, c="b")
            plt.title("contamination estimation")
            plt.show()
    else:
        diff = 0
        biasnoise = 1.
        nbad = 0

    obslist = files[0][0]
    for i in range(1, len(files)):
        obslist = obslist + " " + files[i][0]

    try:
        del head["tapelist"]
    except KeyError:
        pass

    head["bzero"] = 0.0
    head["bscale"] = 1.0
    head["obslist"] = obslist
    head["nimages"] = (n, "number of images summed")
    head["npixfix"] = (nbad, "pixels corrected for cosmic rays")
    head["bgnoise"] = (biasnoise, "noise in combined image, electrons")
    return bias, head
