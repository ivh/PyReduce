import numpy as np

from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad

from sumfits import sumfits
from clipnflip import clipnflip


def gaussfit(x, y, nterm=3):

    if nterm == 3:
        gauss = lambda x, A0, A1, A2: A0 * np.exp(-((x - A1) / A2)**2 / 2)

    popt, pcov = curve_fit(gauss, x, y, p0=[max(y), 1, 1])
    return gauss(x, *popt), popt


def sumbias(list1, list2, inst_setting, bias=None, head=None, debug=None, xr=None, yr=None, err=None, exten=1):
    """
    # Combine bias frames, determine read noise, reject bad pixels.
    # Read noise calculation only valid if both lists yield similar noise.
    """
    err = None

    # Lists of images.
    list = np.concatenate((list1, list2))
    n1 = len(list1)
    n2 = len(list2)
    n = n1 + n2

    # Separately images in two groups.
    bias1, head1 = sumfits(list1, inst_setting, xr=xr,
                           yr=yr, err=err, exten=exten)
    if err is not None:
        err = 'sumbias: error ' + err
        raise Exception(err)
    bias1 = clipnflip(bias1 / float(n1), head1)

    bias2, head2 = sumfits(list2, inst_setting, xr=xr,
                           yr=yr, err=err, exten=exten)
    if err is not None:
        err = 'sumbias: error ' + err
        raise Exception(err)
    bias2 = clipnflip(bias2 / float(n2), head2)

    if (bias1.ndim != bias2.ndim or bias1.shape[0] != bias2.shape[0]):
        err = 'sumbias: bias frames in two lists have different dimensions'
        raise Exception(err)

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
    if (np.min(diff) != np.max(diff)):

        crude = np.median(np.abs(diff))  # estimate of noise
        hmin = -5.0 * crude
        hmax = +5.0 * crude
        bsiz = np.clip(2 / n, 0.5, None)
        nbins = int((hmax - hmin) / bsiz)

        h, _ = np.histogram(diff, range=(hmin, hmax), bins=nbins)
        nh = nbins  # histogram points
        xh = hmin + bsiz * (np.arange(0., nh) + 0.5)

        hfit, par = gaussfit(xh, h, nterm=3)
        noise = abs(par[2])  # noise in diff, bias

        # Determine where wings of distribution become significantly non-Gaussian.
        contam = (h - hfit) / np.sqrt(np.clip(hfit, 1, None))
        imid = np.where(abs(xh) < 2 * noise)
        consig = np.std(contam[imid])

        #TODO That good?
        smcontam = gaussbroad(contam, 0.1 * noise)
        igood = np.where(smcontam < 3.0 * consig)
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

        # Plot noise distribution.
        if debug is not None:
            raise NotImplementedError()
            plt.subplots()
            _global.p.multi = __array__((0, 1, 2))
            plot(xh, h, xsty=3, ps=10, title='noise distribution')
            # colors
            oplot(xh, hfit, co=2)
            oplot(gmin + (0, 0), _global.y.crange, co=6)
            oplot(gmax + (0, 0), _global.y.crange, co=6)

            # Plot contamination estimation.
            plot(xh, contam, xsty=3, ps=10, title='contamination estimation')
            oplot(_global.x.crange, thresh + (0, 0), co=2)
            oplot(xh, smcontam, co=3)
            oplot(gmin + (0, 0), _global.y.crange, co=6)
            oplot(gmax + (0, 0), _global.y.crange, co=6)
            _global.p.multi = 0
            print('push space to continue...')
            print('')
    else:
        diff = 0
        biasnoise = 1.
        nbad = 0

    # Toss blank header lines.
    #iwhr = where(strtrim( head , 2 ) !=  '' )
    #head = head[iwhr]

    obslist = list[0][0]
    for i in range(1, len(list)):
        obslist = obslist + ' ' + list[i][0]

    try:
        del head['tapelist']
    except KeyError:
        pass

    del head['naxis1']
    del head['naxis2']

    head['bzero'] = 0.0
    head['bscale'] = 1.0
    head['obslist'] = obslist
    head['nimages'] = (n, ' number of images summed')
    head['npixfix'] = (nbad, ' pixels corrected for cosmic rays')
    head['bgnoise'] = (biasnoise,
                       ' noise in combined image, electrons')
    return bias, head
