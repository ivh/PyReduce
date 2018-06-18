"""
REDUCE script for spectrograph data
"""
import glob
import json
import os.path

import astropy.io.fits as fits
import numpy as np

from clipnflip import clipnflip
# from hamdord import hamdord
# from modeinfo import modeinfo
from modeinfo_uves import modeinfo_uves as modeinfo
from combine_bias import combine_bias
from combine_frames import combine_frames

# some basic settings
base_dir = './Test'
mask_dir = './Test/UVES/HD132205'
instrument = 'UVES'
target = 'HD132205'

# load configuration for the current instrument
with open('settings_%s.json' % instrument) as f:
    config = json.load(f)
modes = config['modes'][:2]


def create_flat(flatlist, inst_mode, exten, bias):
    """ Create a master flat from a list of flats """
    flat, fhead = combine_frames(
        flatlist, inst_mode, exten=exten)
    flat = clipnflip(flat, fhead)
    # = Subtract master dark. We have to scale it by the number of FF's
    flat = flat - bias * len(flatlist)  # subtract bias

    flat = flat[:, ::-1]

    return flat.astype(np.float32), fhead

def create_bias(biaslist, inst_mode, exten):
    """ Create a Master bias from a list of bias files """
    n = int(np.floor(len(biaslist) / 2))
    # necessary to maintain proper dimensionality, if there is just one element
    if n == 0:
        biaslist = np.array([biaslist, biaslist])
        n = 1
    bias, bhead = combine_bias(biaslist[:n], biaslist[n:], inst_mode, exten=exten, debug=True)

    return bias.astype(np.float32), bhead

if __name__ == '__main__':

    # Search the available days
    dates = os.path.join(base_dir, instrument, target, 'raw', '????-??-??')
    dates = glob.glob(dates)
    dates = [r + os.sep for r in dates if os.path.isdir(r)]

    for night in dates:
        night = os.path.basename(night[:-1])
        print(night)
        for counter_mode, inst_mode in enumerate(modes):
            print(inst_mode)

            # read configuration settings
            mode = config["modes"][counter_mode]
            inst_mode = config["small"] + '_' + mode
            exten = config["extensions"][counter_mode]
            prefix = inst_mode
            current_mode_value = config["mode_value"][counter_mode]

            # define paths
            raw_path = os.path.join(base_dir, instrument, target, 'raw', night) + os.sep
            reduced_path = os.path.join(base_dir, instrument, target, 'reduced', night, 'Reduced_' + mode) + os.sep

            # define files
            mask_file = os.path.join(mask_dir, 'mask_%s.fits.gz' % inst_mode)
            bias_file = os.path.join(reduced_path, prefix + '.bias.fits')
            flat_file = os.path.join(reduced_path, prefix + '.flat.fits')
            norm_flat_file = os.path.join(reduced_path, prefix + '.flat.norm.fits')
            ord_norm_file = os.path.join(reduced_path, prefix + '.ord_norm.sav')
            ord_default_file = os.path.join(reduced_path, prefix + '.ord_default.sav')

            # create output folder structure if necessary
            if not os.path.exists(reduced_path):
                os.makedirs(reduced_path)

            # find input files and sort them by type
            files = np.array(glob.glob(raw_path + '%s.*.fits.gz' % instrument))

            ob = np.zeros(len(files), dtype='U20')
            ty = np.zeros(len(files), dtype='U20')
            #mo = np.zeros(len(files), dtype='U20')
            exptime = np.zeros(len(files))

            for i, f in enumerate(files):
                h = fits.open(f)[0].header
                ob[i] = h['OBJECT']
                ty[i] = h['ESO DPR TYPE']
                exptime[i] = h['EXPTIME']
                #mo[i] = h['ESO INS MODE']

            biaslist = files[ty == config["id_bias"]]
            flatlist = files[ty == config["id_flat"]]
            tharlist = files[ob == config["id_wave"]]
            orderlist = files[ob == config["id_orders"]]
            orderdef_fiber_a = files[ob == config["id_fiber_a"]]
            orderdef_fiber_b = files[ob == config["id_fiber_b"]]
            spec = files[ob == 'HD-132205']

            # TODO: to have the same order as idl, for easier control
            biaslist = biaslist[[4, 0, 3, 2, 1]]

            # Which parts of the reduction to perform
            steps_to_take = ['bias', 'flat', 'orders']

            #
            # ==========================================================================
            # Creat master bias
            #
            if 'bias' in steps_to_take:
                if len(biaslist) > 0:
                    print('Creating master bias')
                    bias, bhead = create_bias(biaslist, inst_mode, exten=exten)
                    fits.writeto(filename=bias_file, data=bias, header=bhead, overwrite=True)  # save master bias
                else:
                    raise FileNotFoundError('No BIAS files found')
            else:
                print('Loading master bias')
                bias = fits.open(bias_file)[0]

            # ==========================================================================
            # = Create master flat
            if 'flat' in steps_to_take:
                if len(flatlist) > 0:
                    print('Creating flat field')
                    flat, fhead = create_flat(flatlist, inst_mode, exten, bias)
                    fits.writeto(filename=flat_file, data=flat, header=fhead, overwrite=True)  # save master flat
                else:
                    raise FileNotFoundError('No FLAT files found')
            else:
                print('Loading flat field')
                flat = fits.open(flat_file)


            # This is how far it should work
            continue
            # ==========================================================================
            # = Find default orders.

            # if file_test(ord_default_file) then begin
            # restore,ord_default_file
            # endif else begin
            if 'orders' in steps_to_take:
                mask = fits.open(mask_file)  # read ccd mask
                mhead = modeinfo(mask[0].header, inst_mode)
                mask = clipnflip(mask, mhead)

                dord = flat_file
                ordim, hrd = fits.open(dord)

                # manual selection of what to do with clusters disabled until i figure out how it works
                ordim, orders, ord_err, or_range, col_range, opower, mask = hamdord(
                    ordim, filter=30., power=opower, plot=True, mask=mask, thres=200, manual=True, polarim=True)

                # = Determine extraction width, blaze center column, and base order. Save to disk.
                ordim, orders, def_xwd, def_sxwd, col_range = getxwd(
                    ordim, orders, def_xwd, def_sxwd, colrange=col_range, gauss=True)  # get extraction width

                # = Save image format description
                orders, or_range, ord_err, col_range, def_xwd, def_sxwd, ord_default_file = save(
                    orders, or_range, ord_err, col_range, def_xwd, def_sxwd, file=ord_default_file)

            # ==========================================================================
            # = Construct normalized flat field.
            # restore,ord_default_file

            flat = readfits(flat_file, fhead)
            mask_file, mask, mhead = fits_read(
                mask_file, mask, mhead)  # read order definition frame
            mhead = modeinfo(mhead, inst_mode)
            mask = clipnflip(mask, mhead)

            flat, orders, def_xwd, def_sxwd, col_range, _ = getxwd(
                flat, orders, def_xwd, def_sxwd, colrange=col_range, gauss=True)  # get extraction width

            flat, fhead, orders, blzcoef, col_range, def_xwd, _, _, _, mask, _, _ = hamflat(
                flat, fhead, orders, blzcoef, colrange=col_range, fxwd=def_xwd, sf_smooth=4., osample=10, swath_width=200, mask=mask, threshold=10000, plot=True)

            orders, or_range, ord_err, col_range, def_xwd, def_sxwd, blzcoef, ord_norm_file = save(
                orders, or_range, ord_err, col_range, def_xwd, def_sxwd, blzcoef, file=ord_norm_file)
            norm_flat_file, flat, fhead = writefits(
                norm_flat_file, flat, fhead)

            # ==========================================================================
            # Extract thorium spectra.
            # restore,ord_norm_file

            for n in np.arange(0, n_elements(tharlist) + 1, 1):
                nameout = strmid(tharlist[n], strpos(
                    tharlist[n], '/', reverse_search=True) + 1)
                nameout = reduced_path + \
                    strmid(nameout, 0, strlen(nameout) - 8)

                tharlist[n], im, head, exten, _ = fits_read(
                    tharlist[n], im, head, exten=exten, pdu=True)  # read stellar spectrum
                # modify header with instrument setup
                head = modeinfo(head, inst_mode)
                im = clipnflip(im, head)  # clip
                flip
                xwd = def_xwd
                xwd[:, :] = 6
                im, head, orders, xwd, def_sxwd, or_range[0], thar, sunc, col_range, _, _ = hamspec(
                    im, head, orders, xwd, def_sxwd, or_range[0], thar, sig=sunc, colrange=col_range, osample=10, thar=True)

                im = 0
                head, or_range[0] = sxaddpar(
                    head, 'obase', or_range[0], ' base order number')  # , before='comment'
                # save spectrum to disk
                wdech(nameout + '.thar.ech', head,
                      thar, bary=True, overwrite=True)

            # ==========================================================================
            # Prepare for science spectra extractionord_norm_file
            # restore,ord_norm_file

            mask = readfits(mask_file, mhead)
            mhead = modeinfo(mhead, inst_mode)
            mask = clipnflip(mask, mhead)
            flat = readfits(norm_flat_file, fhead)
            bias = readfits(bias_file, bhead)

            nord = n_elements(orders[:, 0])
            xwd = replicate(8, 2, nord)

            for n in np.arange(0, n_elements(spec) + 1, 1):
                nameout = strmid(spec[n], strpos(
                    spec[n], '/', reverse_search=True) + 1)
                nameout = reduced_path + \
                    strmid(nameout, 0, strlen(nameout) - 8)
                spec[n], im, head, exten, _ = fits_read(
                    spec[n], im, head, exten=exten, pdu=True)  # read stellar spectrum
                loadct(0)
                im, _ = display(
                    im, log=True, tit=spec[n] + '(' + inst_mode + ')')
                # modify header with instrument setup
                head = modeinfo(head, inst_mode)
                im = clipnflip(im, head)  # clip
                flip
                im = im - bias

                # Extract frame information from the header
                readn = sxpar(head, 'e_readn')
                dark = sxpar(head, 'e_backg')
                gain = sxpar(head, 'e_gain')

                # Fit the scattered light. The approximation is returned in 2D array bg for each
                # inter-order troff
                im, orders, bg, ybg, col_range, _, _, _, mask, _, gain, readn, _ = mkscatter(
                    im, orders, bg, ybg, colrange=col_range, lambda_sf=60., swath_width=300, osample=10, mask=mask, lambda_sp=10., gain=gain, readn=readn, subtract=True)

                # Flat fielding
                im = im / flat

                # Optimally extract science spectrum
                im, head, orders, xwd, sxwd, or_range[0], sp, sunc, col_range, _, _, _, mask = hamspec(
                    im, head, orders, xwd, sxwd, or_range[0], sp, sig=sunc, colrange=col_range, sf_smooth=3., osample=10, swath_width=200, mask=mask, filename=spec[n])

                im = 0

                #_global.p.multi = __array__((4, 1, 4))

                sigma = sunc
                cont = sunc * 0. + 1.
                for i in np.arange(0, nord - 1 + 1, 1):
                    x = indgen(col_range[i, 1] -
                               col_range[i, 0] + 1) + col_range[i, 0]

                    # convert uncertainty to relative error
                    sigma[i, x] = sunc[i, x] / (sp[i, x] > 1.)

                    s = sp[i, x] / (blzcoef[i, x] > 0.001)
                    c = top(s, 1, eps=0.0002, poly=True)
                    s = s / c
                    c = sp[i, x] / s
                    cont[i, x] = c

                    yr = __array__((0., 2))
                    plot(x, s, xs=1, ys=3, xr=[n_elements(
                        sp[0, :]), 100], charsize=2, yr=yr, title='order:' + strtrim(i + or_range[0], 2))

                #_global.p.multi = 0

                head, or_range[0] = sxaddpar(
                    head, 'obase', or_range[0], ' base order number')  # , before='comment'
                # save spectrum to disk
                wdech(nameout + '.ech', head, sp,
                      sig=sigma, cont=cont, overwrite=True)
                pol_angle = hierarch(
                    head, 'hierarch eso ins ret25 pos', count=nval)
                if (nval == 0):
                    pol_angle = hierarch(
                        head, 'hierarch eso ins ret50 pos', count=nval)
                    if (nval == 0):
                        pol_angle = 'no polarimeter'
                    else:
                        pol_angle = 'lin ' + strtrim(pol_angle, 2)
                else:
                    pol_angle = 'cir ' + strtrim(pol_angle, 2)

                openw(1, reduced_path + night + '.log', append=True)
                printf(1, 'star: ' + hierarch(head, 'object') + ', polarization: ' +
                       pol_angle + ', mean s/n=' + strtrim(round(1e0 / mean(sigma))))
                printf(
                    1, 'file: ' + strmid(spec[n], strpos(spec[n], '/', reverse_search=True) + 1))
                printf(
                    1, '---------------------------------------------------------------------')
                close(1)
                print('completed: star ' + hierarch(head, 'object') + ', polarization: ' +
                      pol_angle + ', mean s/n=' + strtrim(round(1e0 / mean(sigma))))
                print(
                    'file: ' + strmid(spec[n], strpos(spec[n], '/', reverse_search=True) + 1))
