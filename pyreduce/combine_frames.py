"""
Combine several fits files into one master frame

Used to create master bias and master flat
"""

import datetime
import logging
import os
import warnings

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import median_filter
from tqdm import tqdm

from . import util
from .clipnflip import clipnflip
from .instruments.instrument_info import load_instrument
from .util import gaussbroad, gaussfit

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
    tmp = readnoise**2 + (fitted_signal / gain)
    np.sqrt(tmp, where=tmp >= 0, out=predicted_noise)

    # Identify outliers
    badpixels = buffer - fitted_signal > threshold * predicted_noise
    nbad = len(np.nonzero(badpixels.flat)[0])

    # Construct the summed flat
    corrected_signal = np.where(badpixels, fitted_signal, buffer)
    corrected_signal = np.sum(corrected_signal, axis=0)
    return corrected_signal, nbad


def combine_frames_simple(
    files, instrument, channel, extension=None, dtype=np.float32, **kwargs
):
    """
    Simple addition of similar images.

    Parameters
    ----------
    files : list(str)
        list of fits files to combine
    instrument : str
        instrument id for modinfo
    channel : str
        instrument channel
    extension : int, optional
        fits extension to load (default: 1)
    dtype : np.dtype, optional
        datatype of the combined image (default float32)

    Returns
    -------
    combined_data, header
        combined image data, header
    """

    if len(files) == 0:
        raise ValueError("No files given for combine frames")

    # Load the first file to get the shape and header
    result, head = instrument.load_fits(
        files[0], channel, dtype=dtype, extension=extension, **kwargs
    )

    # Sum the remaining files
    for fname in files[1:]:
        data, _ = instrument.load_fits(
            fname, channel, dtype=dtype, extension=extension, **kwargs
        )
        result += data

    # Update the header
    head["NIMAGES"] = (len(files), "number of images summed")
    head["EXPTIME"] = (head["EXPTIME"] * len(files), "total exposure time")
    head["DARKTIME"] = (
        head.get("DARKTIME", head["EXPTIME"]) * len(files),
        "total dark time",
    )

    # Update the readout noise
    if "RDNOISE" in head:
        head["RDNOISE"] = (
            head["RDNOISE"] * np.sqrt(len(files)),
            "readout noise in combined image",
        )

    head.add_history(f"Combined {len(files)} images by simple addition")

    return result, head


def combine_frames(
    files,
    instrument,
    channel,
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
    arminfo conventions to figure out image orientation and useful pixel ranges.
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
        instrument id for arminfo
    channel : str
        instrument channel
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
    for i, fname in zip(range(len(files)), files, strict=False):
        logger.debug("%i\t%s", i, fname)

    # Only one image
    if len(files) == 0:
        raise ValueError("No files given for combine frames")
    elif len(files) == 1:
        result, head = instrument.load_fits(
            files[0], channel, dtype=dtype, extension=extension, **kwargs
        )
        readnoise = np.atleast_1d(head.get("e_readn", 0))
        total_exposure_time = head.get("exptime", 0)
        n_fixed = 0
        linear = head.get("e_linear", True)

    # Two images
    elif len(files) == 2:
        bias1, head1 = instrument.load_fits(
            files[0], channel, dtype=dtype, extension=extension, **kwargs
        )
        exp1 = head1.get("exptime", 0)

        bias2, head2 = instrument.load_fits(
            files[1], channel, dtype=dtype, extension=extension, **kwargs
        )
        exp2 = head2.get("exptime", 0)
        readnoise = head2.get("e_readn", 0)

        result = bias2 + bias1
        head = head2

        total_exposure_time = exp1 + exp2
        readnoise = np.atleast_1d(readnoise)
        n_fixed = 0
        linear = head.get("e_linear", True)

    else:  # More than two images
        # Get information from headers
        # TODO: check if all values are the same in all the headers?

        heads = [
            instrument.load_fits(
                f, channel, header_only=True, dtype=dtype, extension=extension, **kwargs
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
        orientation = head.get("e_orient", 0)
        head.get("e_transpose", False)
        orientation = orientation % 8
        # check if non-linearity correction
        linear = head.get("e_linear", True)

        # section(s) of the detector to process, x_low, x_high, y_low, y_high
        # head["e_xlo*"] will find all entries with * as a wildcard
        # we also ensure that we will have one dimensional arrays (not just the value)
        cards = sorted(head["e_xlo*"].cards, key=lambda c: c[0])
        x_low = [c[1] for c in cards]
        cards = sorted(head["e_xhi*"].cards, key=lambda c: c[0])
        x_high = [c[1] for c in cards]
        cards = sorted(head["e_ylo*"].cards, key=lambda c: c[0])
        y_low = [c[1] for c in cards]
        cards = sorted(head["e_yhi*"].cards, key=lambda c: c[0])
        y_high = [c[1] for c in cards]

        cards = sorted(head["e_gain*"].cards, key=lambda c: c[0])
        gain = [c[1] for c in cards]
        cards = sorted(head["e_readn*"].cards, key=lambda c: c[0])
        readnoise = [c[1] for c in cards]
        total_exposure_time = sum(h.get("exptime", 0) for h in heads)

        # Scaling for image data
        bscale = [h.get("bscale", 1) for h in heads]
        bzero = [h.get("bzero", 0) for h in heads]

        result = np.zeros((n_rows, n_columns), dtype=dtype)  # the combined image
        n_fixed = 0  # number of fixed pixels

        # Load all image hdus, but leave the data on the disk, using memmap
        # Need to scale data later
        if extension is None:
            extension = [instrument.get_extension(h, channel) for h in heads]
        else:
            extension = [extension] * len(heads)

        data = [
            fits.open(f, memmap=True, do_not_scale_image_data=True)[e]
            for f, e in zip(files, extension, strict=False)
        ]

        if window >= n_columns / 2:
            window = n_columns // 10
            logger.warning("Reduce Window size to fit the image")

        # depending on the orientation the indexing changes and the borders of the image change
        if orientation in [1, 3, 4, 6]:
            # idx gives the index for accessing the data in the image, which is rotated depending on the orientation
            # We could just rotate the whole image, but that requires reading the whole image at once
            def index(row, x_left, x_right):
                return (slice(x_left, x_right), row)

            # Exchange the borders of the image
            x_low, x_high, y_low, y_high = y_low, y_high, x_low, x_high
        else:

            def index(row, x_left, x_right):
                return (row, slice(x_left, x_right))

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
                    # If the following causes int16 overflow, add .astype('float64')
                    # to the first term. The receiving buffer is f64 anyway.
                    buffer[i, :] = (
                        data[i].data[idx].astype("float64") * bscale[i] + bzero[i]
                    )

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

        for d in data:
            d._file.close()  # Close open FITS files

    # Add info to header.
    head["bzero"] = 0.0
    head["bscale"] = 1.0
    head["exptime"] = total_exposure_time
    head["darktime"] = total_exposure_time
    # Because we do not divide the signal by the number of files the
    # read-out noise goes up by the square root of the number of files

    for n_amp, rdn in enumerate(readnoise):
        head[f"rdnoise{n_amp + 1:0>1}"] = (
            rdn * np.sqrt(len(files)),
            " noise in combined image, electrons",
        )

    head["nimages"] = (len(files), " number of images summed")
    head["npixfix"] = (n_fixed, " pixels corrected for cosmic rays")
    head.add_history(
        f"images coadded by combine_frames.py on {datetime.datetime.now()}"
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


def combine_calibrate(
    files,
    instrument,
    channel,
    mask=None,
    bias=None,
    bhead=None,
    norm=None,
    bias_scaling="exposure_time",
    norm_scaling="divide",
    plot=False,
    plot_title=None,
    **kwargs,
):
    """
    Combine the input files and then calibrate the image with the bias
    and normalized flat field if provided

    Parameters
    ----------
    files : list
        list of file names to load
    instrument : Instrument
        PyReduce instrument object with load_fits method
    channel : str
        descriptor of the instrument channel
    mask : array
        2D Bad Pixel Mask to apply to the master image
    bias : tuple(bias, bhead), optional
        bias correction to apply to the combiend image, if bias has 3 dimensions
        it is used as polynomial coefficients scaling with the exposure time, by default None
    norm_flat : tuple(norm, blaze), optional
        normalized flat to divide the combined image with after
        the bias subtraction, by default None
    bias_scaling : str, optional
        defines how the bias is subtracted, by default "exposure_time"
    plot : bool, optional
        whether to plot the results, by default False
    plot_title : str, optional
        Name to put on the plot, by default None

    Returns
    -------
    orig : array
        combined image with calibrations applied
    thead : Header
        header of the combined image

    Raises
    ------
    ValueError
        Unrecognised bias_scaling option
    """
    # Combine the images and try to remove bad pixels
    orig, thead = combine_frames(files, instrument, channel, mask=mask, **kwargs)

    # Subtract bias
    if bias is not None and bias_scaling is not None and bias_scaling != "none":
        if bias.ndim == 2:
            if bhead["exptime"] == 0 and bias_scaling == "exposure_time":
                logger.warning(
                    "No exposure time set in bias, using number of files instead"
                )
                bias_scaling = "number_of_files"
            if bias_scaling == "exposure_time":
                orig -= bias * thead["exptime"] / bhead["exptime"]
            elif bias_scaling == "number_of_files":
                orig -= bias * len(files)
            elif bias_scaling == "mean":
                orig -= bias * np.ma.mean(orig) / np.ma.mean(bias)
            elif bias_scaling == "median":
                orig -= bias * np.ma.median(orig) / np.ma.median(bias)
            else:
                raise ValueError(
                    f"Unexpected value for 'bias_scaling', expected one of ['exposure_time', 'number_of_files', 'mean', 'median', 'none'], but got {bias_scaling}"
                )
        else:
            bias.shape[0]
            if bias_scaling == "exposure_time":
                orig -= np.polyval(bias, thead["exptime"])
            # elif bias_scaling == "number_of_files":
            #     flat -= bias * len(files)
            # elif bias_scaling == "mean":
            #     flat -= bias * np.ma.mean(flat) / np.ma.mean(bias)
            # elif bias_scaling == "median":
            #     flat -= bias * np.ma.median(flat) / np.ma.median(bias)
            else:
                raise ValueError(
                    f"Unexpected value for 'bias_scaling', expected one of ['exposure_time'], but got {bias_scaling}"
                )

    # Remove the Flat
    if norm is not None and norm_scaling != "none":
        if norm_scaling == "divide":
            orig /= norm
        else:
            raise ValueError(
                f"Unexpected value for 'norm_scaling', expected one of ['divide', 'none'], but got {norm_scaling}"
            )

    if plot:  # pragma: no cover
        title = "Master"
        if plot_title is not None:
            title = f"{plot_title}\n{title}"
        plt.title(title)
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        bot, top = np.percentile(orig[orig != 0], (10, 90))
        plt.imshow(orig, vmin=bot, vmax=top, origin="lower")
        util.show_or_save("combine_master")

    return orig, thead


def combine_polynomial(
    files, instrument, channel, mask, degree=1, plot=False, plot_title=None
):
    """
    Combine the input files by fitting a polynomial of the pixel value versus
    the exposure time of each pixel

    Parameters
    ----------
    files : list
        list of file names
    instrument : Instrument
        PyReduce instrument object with load_fits method
    channel : str
        channel identifier for this instrument
    mask : array
        bad pixel mask to apply to the coefficients
    degree : int, optional
        polynomial degree of the fit, by default 1
    plot : bool, optional
        whether to plot the results, by default False
    plot_title : str, optional
        Title of the plot, by default None

    Returns
    -------
    bias : array
        3d array with the coefficients for each pixel
    bhead : Header
        combined FITS header of the coefficients
    """
    hdus = [instrument.load_fits(f, channel) for f in tqdm(files)]
    data = np.array([h[0] for h in hdus])
    exptimes = np.array([h[1]["EXPTIME"] for h in hdus])
    # Numpy polyfit can fit all polynomials at the same time
    # but we need to flatten the pixels into 1 dimension
    data_flat = data.reshape((len(exptimes), -1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        coeffs = np.polyfit(exptimes, data_flat, degree)
    # Afterwards we reshape the coefficients into the image shape
    shape = (degree + 1, data.shape[1], data.shape[2])
    coeffs = coeffs.reshape(shape)
    # And apply the mask to each image of coefficients
    if mask is not None:
        bias = np.ma.masked_array(coeffs, mask=[mask for _ in range(degree + 1)])
    # We arbitralily pick the first header as the bias header
    # and change the exposure time
    bhead = hdus[0][1]
    bhead["EXPTIME"] = np.sum(exptimes)

    if plot:
        title = "Master"
        if plot_title is not None:
            title = f"{plot_title}\n{title}"

        for i in range(degree + 1):
            plt.subplot(1, degree + 1, i + 1)
            plt.title("Coefficient %i" % (degree - i))
            plt.xlabel("x [pixel]")
            plt.ylabel("y [pixel]")
            bot, top = np.percentile(bias[i], (10, 90))
            plt.imshow(bias[i], vmin=bot, vmax=top, origin="lower")

        plt.suptitle(title)
        util.show_or_save("combine_polynomial")

    return bias, bhead


def combine_bias(
    files,
    instrument,
    channel,
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
        instrument channel for arminfo
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
    else:
        list1, list2 = files[: n // 2], files[n // 2 :]

    # Lists of images.
    n1 = len(list1)
    n2 = len(list2)

    # Separately images in two groups.
    bias1, head1 = combine_frames(list1, instrument, channel, extension, **kwargs)
    bias1 /= n1

    bias2, head = combine_frames(list2, instrument, channel, extension, **kwargs)
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
            util.show_or_save("bias_contamination")
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
        # Handle non-finite values for plotting
        plot_data = np.where(np.isfinite(bias), bias, np.nan)
        valid = np.isfinite(plot_data)
        if np.any(valid):
            bot, top = np.percentile(plot_data[valid], (1, 99))
            if bot >= top:
                bot, top = None, None
        else:
            bot, top = None, None
        plt.imshow(plot_data, vmin=bot, vmax=top, origin="lower")
        util.show_or_save("bias_master")

    head["obslist"] = " ".join([os.path.basename(f) for f in files])
    head["nimages"] = (n, "number of images summed")
    head["npixfix"] = (nbad, "pixels corrected for cosmic rays")
    head["bgnoise"] = (biasnoise, "noise in combined image, electrons")
    return bias, head
