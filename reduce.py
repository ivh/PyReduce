"""
REDUCE script for spectrograph data
"""
import glob
import json
import os.path
import pickle
import argparse
import sys

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np

from clipnflip import clipnflip
from modeinfo_uves import modeinfo_uves as modeinfo
from combine_frames import combine_bias, combine_flat


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


def sort_files(files, config):
    """
    Sort a set of fits files into different categories
    types are: bias, flat, wavecal, orderdef, orderdef_fiber_a, orderdef_fiber_b, spec

    Parameters
    ----------
    files : list(str)
        files to sort
    Returns
    -------
    biaslist, flatlist, wavelist, orderlist, orderdef_fiber_a, orderdef_fiber_b, speclist
        lists of files, one per type
    """

    ob = np.zeros(len(files), dtype='U20')
    ty = np.zeros(len(files), dtype='U20')
    # mo = np.zeros(len(files), dtype='U20')
    exptime = np.zeros(len(files))

    for i, f in enumerate(files):
        h = fits.open(f)[0].header
        ob[i] = h['OBJECT']
        ty[i] = h['ESO DPR TYPE']
        exptime[i] = h['EXPTIME']
        # mo[i] = h['ESO INS MODE']

    biaslist = files[ty == config["id_bias"]]
    flatlist = files[ty == config["id_flat"]]
    wavelist = files[ob == config["id_wave"]]
    orderlist = files[ob == config["id_orders"]]
    orderdef_fiber_a = files[ob == config["id_fiber_a"]]
    orderdef_fiber_b = files[ob == config["id_fiber_b"]]
    speclist = files[ob == 'HD-132205']

    return biaslist, flatlist, wavelist, orderlist, orderdef_fiber_a, orderdef_fiber_b, speclist

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="General REDUCE script")
    parser.add_argument('-b', '--bias', action='store_true', help='Create master bias')
    parser.add_argument('-f', '--flat', action='store_true', help='Create master flat')
    parser.add_argument('-o', '--orders', action='store_true', help='Trace orders')
    parser.add_argument('-n', '--norm_flat', action='store_true', help='Normalize flat')
    parser.add_argument('-w', '--wavecal', action='store_true', help='Prepare wavelength calibration')
    parser.add_argument('-s', '--science', action='store_true', help='Extract science spectrum')

    parser.add_argument("instrument", type=str, help="instrument used")
    parser.add_argument("target", type=str, help="target star")

    args = parser.parse_args()
    instrument = args.instrument.upper()
    target = args.target.upper()

    steps_to_take = {'bias': args.bias, 'flat': args.flat, 'orders': args.orders,
                        'norm_flat': args.norm_flat, 'wavecal': args.wavecal, 'science': args.science}
    steps_to_take = [k for k, v in steps_to_take.items() if v]

    # if no steps are specified use all
    if len(steps_to_take) == 0:
        steps_to_take = ['bias', 'flat', 'orders',
                            'norm_flat', 'wavecal', 'science']

    return instrument, target, steps_to_take



if __name__ == '__main__':
    # some basic settings
    # Expected Folder Structure: base_dir/instrument/target/raw/night/*.fits.gz
    base_dir = './Test'
    mask_dir = './Test/UVES/HD132205'

    if len(sys.argv) > 1:
        instrument, target, steps_to_take = parse_args()
    else:
        # Manual settings
        # Instrument
        instrument = "UVES"
        # target star
        target = "HD132205"
        # Which parts of the reduction to perform
        steps_to_take = ['bias',
                         'flat',
                         'orders',
                         'norm_flat',
                         # 'wavecal',
                         'science']



    

    # load configuration for the current instrument
    with open('settings_%s.json' % instrument) as f:
        config = json.load(f)

    modes = config['modes'][1:2]  # TODO: just middle for testing

    # Search the available days
    dates = os.path.join(base_dir, instrument, target, 'raw', '????-??-??')
    dates = glob.glob(dates)
    dates = [r + os.sep for r in dates if os.path.isdir(r)]

    print('Instrument: ', instrument)
    print('Target: ', target)
    for night in dates:
        night = os.path.basename(night[:-1])
        print('Observation Date: ', night)
        for counter_mode, inst_mode in enumerate(modes):
            print('Instrument Mode: ', inst_mode)

            # read configuration settings
            mode = config["modes"][counter_mode]
            inst_mode = config["small"] + '_' + mode
            exten = config["extensions"][counter_mode]
            prefix = inst_mode
            current_mode_value = config["mode_value"][counter_mode]

            # define paths
            raw_path = os.path.join(
                base_dir, instrument, target, 'raw', night) + os.sep
            reduced_path = os.path.join(
                base_dir, instrument, target, 'reduced', night, 'Reduced_' + mode) + os.sep

            # define files
            mask_file = os.path.join(mask_dir, 'mask_%s.fits.gz' % inst_mode)
            bias_file = os.path.join(reduced_path, prefix + '.bias.fits')
            flat_file = os.path.join(reduced_path, prefix + '.flat.fits')
            norm_flat_file = os.path.join(
                reduced_path, prefix + '.flat.norm.fits')
            ord_norm_file = os.path.join(
                reduced_path, prefix + '.ord_norm.sav')
            ord_default_file = os.path.join(
                reduced_path, prefix + '.ord_default.sav')

            # create output folder structure if necessary
            if not os.path.exists(reduced_path):
                os.makedirs(reduced_path)

            # find input files and sort them by type
            files = np.array(glob.glob(raw_path + '%s.*.fits.gz' % instrument))
            f_bias, f_flat, f_wave, f_order, f_order_a, f_order_b, f_spec = sort_files(
                files, config)

            # TODO same order as idl for testing
            f_bias = f_bias[[4, 0, 3, 2, 1]]

            # ==========================================================================
            # Read mask
            mask = fits.open(mask_file)
            mhead, _ = modeinfo(mask[0].header, inst_mode)
            mask = clipnflip(mask[0].data, mhead)
            # TODO apply mask to all files, as masked arrays

            # ==========================================================================
            # Creat master bias
            if 'bias' in steps_to_take:
                print('Creating master bias')
                bias, bhead = combine_bias(f_bias, inst_mode, exten=exten)
                fits.writeto(bias_file, data=bias.data,
                             header=bhead, overwrite=True)
            else:
                print('Loading master bias')
                bias = fits.open(bias_file)[0]
                bias, bhead = bias.data, bias.header
                bias = np.ma.masked_array(bias, mask=mask)

            # ==========================================================================
            # Create master flat
            if 'flat' in steps_to_take:
                print('Creating flat field')
                flat, fhead = combine_flat(f_flat, inst_mode, exten=exten)
                fits.writeto(flat_file, data=flat.data,
                             header=fhead, overwrite=True)
            else:
                print('Loading flat field')
                flat = fits.open(flat_file)[0]
                flat, fhead = flat.data, flat.header
                flat = np.ma.masked_array(flat, mask=mask)

            # ==========================================================================
            # Find default orders.

            if 'orders' in steps_to_take:
                print('Order Tracing')
                # load mask

                if config["use_fiber"] == "A":
                    ordim = fits.open(f_order_a[0])
                    ordim, hrd = ordim[exten].data, ordim[exten].header
                if config["use_fiber"] == "B":
                    ordim = fits.open(f_order_b[0])
                    ordim, hrd = ordim[exten].data, ordim[exten].header
                if config["use_fiber"] == "AB":
                    ordim, hrd = flat, fhead

                # Mark Orders
                orders, or_range, ord_err, col_range = hamdord(
                    ordim, plot=True, manual=True, **config)

                # Determine extraction width, blaze center column, and base order
                def_xwd, def_sxwd = getxwd(
                    ordim, orders, colrange=col_range, gauss=True)

                # Save image format description
                with open(ord_default_file, 'w') as file:
                    pickle.dump(file, orders, or_range, ord_err, col_range,
                                def_xwd, def_sxwd)
            else:
                print('Load order tracing data')
                with open(ord_default_file) as file:
                    pickle.load(file)

            # ==========================================================================
            # = Construct normalized flat field.

            if "norm_flat" in steps_to_take:
                print("Normalize flat field")
                def_xwd, def_sxwd = getxwd(
                    flat, orders, colrange=col_range, gauss=True)  # get extraction width

                flat, fhead, blzcoef = hamflat(
                    flat, fhead, orders, colrange=col_range, fxwd=def_xwd, mask=mask, plot=True, **config)

                # Save data
                # with open(ord_norm_file, 'w') as file:
                #    pickle.dump(file, def_xwd, def_sxwd)
                fits.writeto(norm_flat_file, data=flat,
                             header=fhead, overwrite=True)
            else:
                print("Load normalized flat field")
                flat = fits.open(norm_flat_file)
                flat = flat[0].data, flat[0].header
            # ==========================================================================
            # Prepare wavelength calibration

            if "wavecal" in steps_to_take:
                print("Prepare wavelength calibration")
                for f in f_wave:
                    # Load wavecal image
                    im = fits.open(f)
                    head = load_header(im, exten=exten)
                    head = modeinfo(head, inst_mode)

                    im = im[exten].data
                    im = clipnflip(im, head)

                    # Extract wavecal spectrum
                    thar, head, sunc = hamspec(im, head, orders, def_xwd, def_sxwd,
                                               or_range[0], colrange=col_range, thar=True, **config)

                    head['obase'] = (or_range[0], 'base order number')

                    nameout = os.path.basename(f)
                    nameout, _ = os.path.splitext(nameout)
                    nameout = os.path.join(reduced_path, nameout + '.thar.ech')
                    fits.writeto(nameout, data=thar,
                                 header=head, overwrite=True)

            # ==========================================================================
            # Prepare for science spectra extractionord_norm_file

            if "science" in steps_to_take:
                nord = len(orders[:, 0])

                for f in f_spec:

                    im = fits.open(f)  # read stellar spectrum
                    head = load_header(im, exten=exten)
                    head, _ = modeinfo(head, inst_mode)

                    im = im[exten].data
                    im = clipnflip(im, head)
                    im = im - bias

                    # plt.imshow(im)
                    # plt.title(nameout + '(%s)' % inst_mode)
                    # plt.show()

                    # Extract frame information from the header
                    readn = head['e_readn']
                    dark = head['e_backg']
                    gain = head['e_gain']

                    # Fit the scattered light. The approximation is returned in 2D array bg for each
                    # inter-order troff
                    bg, ybg = mkscatter(im, orders, colrange=col_range, mask=mask,
                                        gain=gain, readn=readn, subtract=True, **config)

                    # Flat fielding
                    im = im / flat

                    # Optimally extract science spectrum
                    sp = hamspec(im, head, orders, def_xwd, def_sxwd,
                                 or_range[0], sig=sunc, colrange=col_range, mask=mask, **config)

                    sigma = sunc
                    cont = np.full_like(sunc, 1.)
                    for i in range(nord):
                        x = np.arange(col_range[i, 0], col_range[i, 1] + 1)

                        # convert uncertainty to relative error
                        sigma[i, x] = sunc[i, x] / np.clip(sp[i, x], 1., None)

                        s = sp[i, x] / np.clip(blzcoef[i, x], 0.001, None)
                        c = top(s, 1, eps=0.0002, poly=True)
                        s = s / c
                        c = sp[i, x] / s
                        cont[i, x] = c

                        yr = (0., 2)
                        plt.plot(x, s)
                        plt.title('order:%i % i' % (i, or_range[0]))

                    head["obase"] = (or_range[0], ' base order number')
                    # save spectrum to disk

                    nameout = os.path.basename(f)
                    nameout, _ = os.path.splitext(f) + '.ech'
                    nameout = os.path.join(reduced_path, nameout)

                    fits.writeto(nameout, data=sp, header=head,
                                 overwrite=True)  # sig=sigma, cont=cont
                    pol_angle = head.get("eso ins ret25 pos")
                    if pol_angle is None:
                        pol_angle = head.get('hierarch eso ins ret50 pos')
                        if pol_angle is None:
                            pol_angle = 'no polarimeter'
                        else:
                            pol_angle = 'lin %i' % pol_angle
                    else:
                        pol_angle = 'cir %i' % pol_angle

                    log_file = os.path.join(reduced_path, night + '.log')
                    with open(log_file, mode='w+') as log:
                        log.write('star: %s, polarization: %i, mean s/n=%.2f\n' %
                                  (head['object'], pol_angle, 1 / np.mean(sigma)))
                        log.write('file: %s\n' % os.path.basename(nameout))
                        log.write('----------------------------------\n')

                    print('star: %s, polarization: %i, mean s/n=%.2f\n' %
                          (head['object'], pol_angle, 1 / np.mean(sigma)))
                    print('file: %s\n' % os.path.basename(nameout))
                    print('----------------------------------\n')
