"""
Combine bias frames into a master bias
"""

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad

from combine_frames import combine_frames
from clipnflip import clipnflip


def gaussfit(x, y):
    """ gaussian fit to data """
    gauss = lambda x, A0, A1, A2: A0 * np.exp(-((x - A1) / A2)**2 / 2)
    popt, _ = curve_fit(gauss, x, y, p0=[max(y), 1, 1])
    return gauss(x, *popt), popt


def combine_bias(list1, list2, inst_setting, exten=1, **kwargs):
    """
    Combine bias frames, determine read noise, reject bad pixels.
    Read noise calculation only valid if both lists yield similar noise.
    """
    xr = kwargs.get("xr")
    yr = kwargs.get("yr")
    debug = kwargs.get("debug", False)

    # Lists of images.
    files = np.concatenate((list1, list2))
    n1 = len(list1)
    n2 = len(list2)
    n = n1 + n2

    # Separately images in two groups.
    bias1, head1 = combine_frames(list1, inst_setting, xr=xr,
                                  yr=yr, exten=exten)
    bias1 = clipnflip(bias1 / n1, head1)

    bias2, head2 = combine_frames(list2, inst_setting, xr=xr,
                                  yr=yr, exten=exten)
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

        smcontam = gaussbroad(contam, 0.1 * noise)
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
            plt.subplot(121)
            plt.plot(xh, h)
            plt.plot(xh, hfit, c='r')
            plt.title('noise distribution')
            plt.axvline(gmin, c='b')
            plt.axvline(gmax, c='b')

            # Plot contamination estimation.
            plt.subplot(122)
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
    return bias, head
