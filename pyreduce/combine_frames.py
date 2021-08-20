"""
Combine several fits files into one master frame

Used to create master bias and master flat
"""

import datetime
import logging
import os

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from dateutil import parser
from scipy.ndimage.filters import median_filter
from tqdm import tqdm

from .clipnflip import clipnflip
from .instruments.instrument_info import load_instrument
from .util import gaussbroad, gaussfit, remove_bias

logger = logging.getLogger(__name__)


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
    buffer : array of shape (nx, ny)
        buffer
    window : int
        size of the running window
    method : {"sum", "median"}, optional
        which method to use to average the probabilities (default: "sum")
        "sum" is much faster, but "median" is more resistant to outliers

    Returns
    -------
    weights : array of shape (nx, ny - 2 * window)
        probabilities
    """

    buffer = np.require(buffer, dtype=float)

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
    np.divide(weights, sum_of_weights, where=sum_of_weights > 0, out=weights)
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
    ratio = np.zeros_like(probability)
    np.divide(buffer, probability, where=probability > 0, out=ratio)
    # ratio = np.where(probability > 0, buffer / probability, 0.)
    amplitude = (
        np.sum(ratio, axis=0) - np.min(ratio, axis=0) - np.max(ratio, axis=0)
    ) / (buffer.shape[0] - 2)

    fitted_signal = np.where(probability > 0, amplitude[None, :] * probability, 0)
    predicted_noise = np.zeros_like(fitted_signal)
    tmp = readnoise ** 2 + (fitted_signal / gain)
    np.sqrt(tmp, where=tmp >= 0, out=predicted_noise)

    # Identify outliers
    badpixels = buffer - fitted_signal > threshold * predicted_noise
    nbad = len(np.nonzero(badpixels.flat)[0])

    # Construct the summed flat
    corrected_signal = np.where(badpixels, fitted_signal, buffer)
    corrected_signal = np.sum(corrected_signal, axis=0)
    return corrected_signal, nbad


def combine_frames(
    files,
    instrument,
    mode,
    extension=None,
    threshold=3.5,
    window=50,
    dtype=np.float32,
    **kwargs,
):
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

    calc_probability::

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

    fix_bad_pixels::

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
    mode : str
        instrument mode
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

    DEBUG_NROWS = 100  # print status update every DEBUG_NROWS rows (if debug is True)
    if instrument is None or isinstance(instrument, str):
        instrument = load_instrument(instrument)

    # summarize file info
    logger.debug("Files:")
    for i, fname in zip(range(len(files)), files):
        logger.debug("%i\t%s", i, fname)

    # Only one image
    if len(files) == 0:
        raise ValueError("No files given for combine frames")
    elif len(files) == 1:
        result, head = instrument.load_fits(
            files[0], mode, dtype=dtype, extension=extension, **kwargs
        )
        return result, head
    # Two images
    elif len(files) == 2:
        bias1, head1 = instrument.load_fits(
            files[0], mode, dtype=dtype, extension=extension, **kwargs
        )
        exp1 = head1.get("exptime", 0)

        bias2, head2 = instrument.load_fits(
            files[1], mode, dtype=dtype, extension=extension, **kwargs
        )
        exp2 = head2.get("exptime", 0)
        readnoise = head2.get("e_readn", 0)

        result = bias2 + bias1
        head = head2

        total_exposure_time = exp1 + exp2
        readnoise = np.atleast_1d(readnoise)
        n_fixed = 0
        linear = head.get("e_linear", True)
    # More than two images
    else:
        # Get information from headers
        # TODO: check if all values are the same in all the headers?

        heads = [
            instrument.load_fits(
                f, mode, header_only=True, dtype=dtype, extension=extension, **kwargs
            )
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
        transpose = head.get("e_transpose", False)
        orientation = orientation % 8
        # check if non-linearity correction
        linear = head.get("e_linear", True)

        # section(s) of the detector to process, x_low, x_high, y_low, y_high
        # head["e_xlo*"] will find all entries with * as a wildcard
        # we also ensure that we will have one dimensional arrays (not just the value)
        x_low = [c[1] for c in head["e_xlo*"].cards]
        x_high = [c[1] for c in head["e_xhi*"].cards]
        y_low = [c[1] for c in head["e_ylo*"].cards]
        y_high = [c[1] for c in head["e_yhi*"].cards]

        gain = [c[1] for c in head["e_gain*"].cards]
        readnoise = [c[1] for c in head["e_readn*"].cards]
        total_exposure_time = sum([h.get("exptime", 0) for h in heads])

        # Scaling for image data
        bscale = [h.get("bscale", 1) for h in heads]
        bzero = [h.get("bzero", 0) for h in heads]

        result = np.zeros((n_rows, n_columns), dtype=dtype)  # the combined image
        n_fixed = 0  # number of fixed pixels

        # Load all image hdus, but leave the data on the disk, using memmap
        # Need to scale data later
        if extension is None:
            extension = [instrument.get_extension(h, mode) for h in heads]
        else:
            extension = [extension] * len(heads)

        data = [
            fits.open(f, memmap=True, do_not_scale_image_data=True)[e]
            for f, e in zip(files, extension)
        ]

        if window >= n_columns / 2:
            window = n_columns // 10
            logger.warning("Reduce Window size to fit the image")

        # depending on the orientation the indexing changes and the borders of the image change
        if orientation in [1, 3, 4, 6]:
            # idx gives the index for accessing the data in the image, which is rotated depending on the orientation
            # We could just rotate the whole image, but that requires reading the whole image at once
            index = lambda row, x_left, x_right: (slice(x_left, x_right), row)
            # Exchange the borders of the image
            x_low, x_high, y_low, y_high = y_low, y_high, x_low, x_high
        else:
            index = lambda row, x_left, x_right: (row, slice(x_left, x_right))

        # For several amplifiers, different sections of the image are set
        # One for each amplifier, each amplifier is treated seperately
        for amplifier in range(n_amplifier):
            # Pick data for current amplifier
            x_left = x_low[amplifier]
            x_right = x_high[amplifier]
            y_bottom = y_low[amplifier]
            y_top = y_high[amplifier]

            gain_amp = gain[amplifier]
            readnoise_amp = readnoise[amplifier]

            # Prepare temporary arrays
            buffer = np.zeros((len(files), x_right - x_left))
            probability = np.zeros((len(files), x_right - x_left))

            # for each row
            for row in tqdm(range(y_bottom, y_top), desc="Rows"):
                if (row) % DEBUG_NROWS == 0:
                    logger.debug(
                        "%i rows processed - %i pixels fixed so far", row, n_fixed
                    )

                # load current row
                idx = index(row, x_left, x_right)
                for i in range(len(files)):
                    # TODO: does memmap not work with compressed files?
                    buffer[i, :] = data[i].data[idx] * bscale[i] + bzero[i]

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
                result[idx], n_bad = fix_bad_pixels(
                    probability, buffer, readnoise_amp, gain_amp, threshold
                )
                n_fixed += n_bad

        logger.debug("total cosmic ray hits identified and removed: %i", n_fixed)

        result = clipnflip(result, head)
        result = np.ma.masked_array(result, mask=kwargs.get("mask"))

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

    if not linear:  # pragma: no cover
        # non-linearity was fixed. mark this in the header
        raise NotImplementedError()  # TODO Nonlinear
        # i = np.where(head["e_linear"] >= 0)
        # head[i] = np.array((head[0 : i - 1 + 1], head[i + 1 :]))
        # head["e_linear"] = ("t", " image corrected of non-linearity")

        # ii = np.where(head["e_gain*"] >= 0)
        # if len(ii[0]) > 0:
        #     for i in range(len(ii[0])):
        #         k = ii[i]
        #         head = np.array((head[0 : k - 1 + 1], head[k + 1 :]))
        # head["e_gain"] = (1, " image was converted to e-")

    return result, head


def combine_flat(
    files,
    instrument,
    mode,
    extension=None,
    bhead=None,
    bias=None,
    plot=False,
    plot_title=None,
    bias_scaling="number_of_files",
    **kwargs,
):
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
    flat, fhead = combine_frames(files, instrument, mode, extension, **kwargs)
    # Subtract master dark
    # TODO: Why do we scale with number of files and not exposure time?
    if bias is not None:
        if bias_scaling == "number_of_files":
            flat -= bias * len(files)
        elif bias_scaling == "exposure_time":
            flat -= bias * fhead["exptime"] / bhead["exptime"]
        elif bias_scaling == "mean":
            flat -= bias * np.ma.mean(flat) / np.ma.mean(bias)
        elif bias_scaling == "median":
            flat -= bias * np.ma.median(flat) / np.ma.median(bias)
        else:
            raise ValueError("Unexpected value for 'bias_scaling', expected one of ['number_of_files', 'exposure_time'], but got %s" % bias_scaling)
    

    if plot:  # pragma: no cover
        title = "Master Flat"
        if plot_title is not None:
            title = f"{plot_title}\n{title}"
        plt.title(title)
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        bot, top = np.percentile(flat, (10, 90))
        plt.imshow(flat, vmin=bot, vmax=top, origin="lower")
        plt.show()

    return flat, fhead


def combine_bias(
    files,
    instrument,
    mode,
    extension=None,
    plot=False,
    plot_title=None,
    science_observation_time=None,
    **kwargs,
):
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

    n = len(files)
    if n == 0:
        raise FileNotFoundError("No bias file(s) given")
    elif n == 1:
        # if there is just one element compare it with itself, not really useful, but it works
        list1 = list2 = files
        n = 2
    # elif science_observation_time is not None:
    #     # TODO split bias into before and after observation sets, if possible
    #     try:
    #         kw = get_instrument_info(instrument)["date"]
    #         times = np.array([parser.parse(fits.open(f)[0].header[kw]) for f in files])
    #         list1 = files[times < science_observation_time]
    #         list2 = files[times > science_observation_time]
    #         # np.digitize(science_observation_time, sorted(times))
    #     except KeyError:
    #         logger.info("Could not sort files by observation time")
    #         list1, list2 = files[: n // 2], files[n // 2 :]
    else:
        list1, list2 = files[: n // 2], files[n // 2 :]

    # Lists of images.
    n1 = len(list1)
    n2 = len(list2)

    # Separately images in two groups.
    bias1, head1 = combine_frames(list1, instrument, mode, extension, **kwargs)
    bias1 /= n1

    bias2, head = combine_frames(list2, instrument, mode, extension, **kwargs)
    bias2 /= n2

    # Make sure we know the gain.
    gain = head.get("e_gain*", (1,))[0]

    # Construct normalized sum.
    bias = (bias1 * n1 + bias2 * n2) / n
    exptime_1 = head1.get("exptime", 0)
    exptime_2 = head.get("exptime", 0)
    head["exptime"] = (exptime_1 + exptime_2) / n

    # Compute noise in difference image by fitting Gaussian to distribution.
    diff = 0.5 * (bias1 - bias2)
    if np.min(diff) != np.max(diff):
        crude = np.ma.median(np.abs(diff))  # estimate of noise
        hmin = -5.0 * crude
        hmax = +5.0 * crude
        bin_size = np.clip(2 / n, 0.5, None)
        nbins = int((hmax - hmin) / bin_size)

        h, _ = np.histogram(diff, range=(hmin, hmax), bins=nbins)
        xh = hmin + bin_size * (np.arange(0.0, nbins) + 0.5)

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
        logger.debug("change in bias between image sets= %f electrons", gain * par[1])
        logger.debug("measured background noise per image= %f", bgnoise)
        logger.debug("background noise in combined image= %f", biasnoise)
        logger.debug("fixing %i bad pixels", nbad)

        if debug:  # pragma: no cover
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
        biasnoise = 1.0
        nbad = 0

    if plot:  # pragma: no cover
        title = "Master Bias"
        if plot_title is not None:
            title = f"{plot_title}\n{title}"
        plt.title(title)
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        bot, top = np.percentile(bias, (1, 99))
        plt.imshow(bias, vmin=bot, vmax=top, origin="lower")
        plt.show()

    head["obslist"] = " ".join([os.path.basename(f) for f in files])
    head["nimages"] = (n, "number of images summed")
    head["npixfix"] = (nbad, "pixels corrected for cosmic rays")
    head["bgnoise"] = (biasnoise, "noise in combined image, electrons")
    return bias, head
