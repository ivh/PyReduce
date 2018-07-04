"""
REDUCE script for spectrograph data
"""
import argparse
import glob
import json
import os.path
import pickle
import sys
from os.path import join
from trace import mark_orders

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np

from getxwd import getxwd
from combine_frames import combine_bias, combine_flat
from extract import extract
from normalize_flat import normalize_flat
from util import find_first_index, load_fits, save_fits, swap_extension, top

# TODO turn dicts into numpy structured array
# TODO proper logging


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

    # TODO use instrument info instead of settings for labels?
    ob = np.zeros(len(files), dtype="U20")
    ty = np.zeros(len(files), dtype="U20")
    # mo = np.zeros(len(files), dtype='U20')
    exptime = np.zeros(len(files))

    for i, f in enumerate(files):
        h = fits.open(f)[0].header
        ob[i] = h["OBJECT"]
        ty[i] = h["ESO DPR TYPE"]
        exptime[i] = h["EXPTIME"]
        # mo[i] = h['ESO INS MODE']

    biaslist = files[ty == config["id_bias"]]
    flatlist = files[ty == config["id_flat"]]
    wavelist = files[ob == config["id_wave"]]
    orderlist = files[ob == config["id_orders"]]
    speclist = files[ob == "HD-132205"]

    return (biaslist, flatlist, wavelist, orderlist, speclist)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="General REDUCE script")
    parser.add_argument("-b", "--bias", action="store_true", help="Create master bias")
    parser.add_argument("-f", "--flat", action="store_true", help="Create master flat")
    parser.add_argument("-o", "--orders", action="store_true", help="Trace orders")
    parser.add_argument("-n", "--norm_flat", action="store_true", help="Normalize flat")
    parser.add_argument(
        "-w", "--wavecal", action="store_true", help="Prepare wavelength calibration"
    )
    parser.add_argument(
        "-s", "--science", action="store_true", help="Extract science spectrum"
    )

    parser.add_argument("instrument", type=str, help="instrument used")
    parser.add_argument("target", type=str, help="target star")

    args = parser.parse_args()
    instrument = args.instrument.upper()
    target = args.target.upper()

    steps_to_take = {
        "bias": args.bias,
        "flat": args.flat,
        "orders": args.orders,
        "norm_flat": args.norm_flat,
        "wavecal": args.wavecal,
        "science": args.science,
    }
    steps_to_take = [k for k, v in steps_to_take.items() if v]

    # if no steps are specified use all
    if len(steps_to_take) == 0:
        steps_to_take = ["bias", "flat", "orders", "norm_flat", "wavecal", "science"]

    return instrument, target, steps_to_take


def main(target, instrument, mode, night, config, steps="all"):
    counter_mode = find_first_index(config["modes"], mode)

    # read configuration settings
    inst_mode = config["small"] + "_" + mode
    extension = config["extensions"][counter_mode]
    prefix = inst_mode

    # define paths
    raw_path = join(base_dir, instrument, target, "raw", night) + os.sep
    reduced_path = (
        join(base_dir, instrument, target, "reduced", night, "Reduced_" + mode) + os.sep
    )

    # define files
    mask_file = join(mask_dir, "mask_%s.fits.gz" % inst_mode)
    bias_file = join(reduced_path, prefix + ".bias.fits")
    flat_file = join(reduced_path, prefix + ".flat.fits")
    norm_flat_file = join(reduced_path, prefix + ".flat.norm.fits")
    ord_norm_file = join(reduced_path, prefix + ".ord_norm.sav")
    ord_default_file = join(reduced_path, prefix + ".ord_default.sav")

    # create output folder structure if necessary
    if not os.path.exists(reduced_path):
        os.makedirs(reduced_path)

    # find input files and sort them by type
    files = glob.glob(raw_path + "%s.*.fits" % instrument)
    files += glob.glob(raw_path + "%s.*.fits.gz" % instrument)
    files = np.array(files)
    f_bias, f_flat, f_wave, f_order, f_spec = sort_files(files, config)

    # ==========================================================================
    # Read mask
    # the mask is not stored with the data files (it is not supported by astropy)
    mask, _ = load_fits(mask_file, inst_mode, extension=0)
    mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks

    # ==========================================================================
    # Create master bias
    if "bias" in steps or steps == "all":
        print("Creating master bias")
        bias, bhead = combine_bias(f_bias, inst_mode, mask=mask, extension=extension)
        fits.writeto(bias_file, data=bias.data, header=bhead, overwrite=True)
    else:
        print("Loading master bias")
        bias = fits.open(bias_file)[0]
        bias, bhead = bias.data, bias.header
        bias = np.ma.masked_array(bias, mask=mask)

    # ==========================================================================
    # Create master flat
    if "flat" in steps or steps == "all":
        print("Creating master flat")
        flat, fhead = combine_flat(
            f_flat, inst_mode, mask=mask, extension=extension, bias=bias
        )
        fits.writeto(flat_file, data=flat.data, header=fhead, overwrite=True)
    else:
        print("Loading master flat")
        flat = fits.open(flat_file)[0]
        flat, fhead = flat.data, flat.header
        flat = np.ma.masked_array(flat, mask=mask)

    # ==========================================================================
    # Find default orders.

    if "orders" in steps or steps == "all":
        print("Order Tracing")

        order_img, _ = load_fits(f_order[0], inst_mode, extension, mask=mask)

        # Mark Orders
        orders, column_range = mark_orders(
            order_img,
            min_cluster=config.get("orders_threshold", 500),
            filter_size=config.get("orders_filter", 120),
            noise=config.get("orders_noise", 8),
            opower=config.get("orders_opower", 4),
            manual=True,
        )

        # Save image format description
        with open(ord_default_file, "wb") as file:
            pickle.dump((orders, column_range), file)
    else:
        print("Load order tracing data")
        with open(ord_default_file, "rb") as file:
            orders, column_range = pickle.load(file)

    # ==========================================================================
    # = Construct normalized flat field.

    if "norm_flat" in steps or steps == "all":
        print("Normalize flat field")
        xwd = 8
        #xwd, sxwd = getxwd(flat, orders, colrange=column_range, gauss = True, pixels=True)

        flat, blzcoef = normalize_flat(
            flat,
            fhead,
            orders,
            column_range=column_range,
            xwd=xwd,
            threshold=config.get("normflat_threshold", 10000),
            lambda_sf=config.get("normflat_sf_smooth", 8),
            lambda_sp=config.get("normflat_sp_smooth", 4),
            swath_width=config.get("normflat_swath_width", None),
        )

        # Save data
        with open(ord_norm_file, "wb") as file:
            pickle.dump(blzcoef, file)
        fits.writeto(norm_flat_file, data=flat.data, header=fhead, overwrite=True)
    else:
        print("Load normalized flat field")
        flat = fits.open(norm_flat_file)[0]
        flat, fhead = flat.data, flat.header
        flat = np.ma.masked_array(flat, mask=mask)

        with open(ord_norm_file, "rb") as file:
            blzcoef = pickle.load(file)
    # ==========================================================================
    # Prepare wavelength calibration

    if "wavecal" in steps or steps == "all":
        print("Prepare wavelength calibration")
        for f in f_wave:
            # Load wavecal image
            im, head = load_fits(f, inst_mode, extension, mask=mask)

            # Determine extraction width, blaze center column, and base order
            xwd, sxwd = np.full((len(orders), 2), 2), 0
            order_range = [0, len(orders) - 1]

            # Extract wavecal spectrum
            thar, _ = extract(
                im,
                head,
                orders,
                xwd=xwd,
                sxwd=sxwd,
                order_range=order_range,
                column_range=column_range,
                thar=True,  # Thats the important difference to science extraction, TODO split it into two different functions?
                osample=config.get("wavecal_osample", 1),
            )

            head["obase"] = (order_range[0], "base order number")

            nameout = swap_extension(f, ".thar.ech", reduced_path)
            save_fits(nameout, head, spec=thar)
            # fits.writeto(nameout, data=thar, header=head, overwrite=True)

    # ==========================================================================
    # Prepare for science spectra extraction

    if "science" in steps or steps == "all":
        print("Extract science spectra")
        for f in f_spec:
            im, head = load_fits(f, inst_mode, extension, mask=mask, dtype=np.float32)
            # Correct for bias and flat field
            im -= bias
            im /= flat

            #TODO may or may not work
            #xwd, sxwd = getxwd(im, orders, colrange=column_range, pixels=True)
            xwd, sxwd = 25, 0

            # Optimally extract science spectrum
            spec, sigma = extract(
                im,
                head,
                orders,
                xwd=xwd,
                sxwd=sxwd,
                column_range=column_range,
                lambda_sf=config.get("science_lambda_sf", 0.1),
                lambda_sp=config.get("science_lambda_sp", 0),
                osample=config.get("science_osample", 1),
                swath_width=config.get("science_swath_width", 300),
                plot=True,
            )

            # Calculate Continuum and Error
            # TODO plotting
            cont = np.full_like(sigma, 1.)

            # convert uncertainty to relative error
            sigma /= np.clip(spec, 1., None)
            s = spec / np.clip(blzcoef, 0.001, None)

            # fit simple continuum
            for i in range(len(orders)):
                c = top(s[i], 1, eps=0.0002, poly=True)
                s[i] = s[i] / c
                c = spec[i] / s[i]
                cont[i] = c

            sigma *= cont  # Scale Error with Continuum

            head["obase"] = (order_range[0], " base order number")

            # save spectrum to disk
            nameout = swap_extension(f, ".ech", path=reduced_path)
            save_fits(nameout, head, spec=spec, sig=sigma, cont=cont)

            pol_angle = head.get("eso ins ret25 pos")
            if pol_angle is None:
                pol_angle = head.get("eso ins ret50 pos")
                if pol_angle is None:
                    pol_angle = "no polarimeter"
                else:
                    pol_angle = "lin %i" % pol_angle
            else:
                pol_angle = "cir %i" % pol_angle

            log_file = os.path.join(reduced_path, night + ".log")
            with open(log_file, mode="w+") as log:
                log.write(
                    "star: %s, polarization: %s, mean s/n=%.2f\n"
                    % (head["object"], pol_angle, 1 / np.mean(sigma))
                )
                log.write("file: %s\n" % os.path.basename(nameout))
                log.write("----------------------------------\n")

            print(
                "star: %s, polarization: %s, mean s/n=%.2f\n"
                % (head["object"], pol_angle, 1 / np.mean(sigma))
            )
            print("file: %s\n" % os.path.basename(nameout))
            print("----------------------------------\n")


if __name__ == "__main__":
    # some basic settings
    # Expected Folder Structure: base_dir/instrument/target/raw/night/*.fits.gz
    base_dir = "./Test"
    mask_dir = "./Test/UVES/HD132205"

    if len(sys.argv) > 1:
        instrument, target, steps_to_take = parse_args()
    else:
        # Manual settings
        # Instrument
        instrument = "UVES"
        # target star
        target = "HD132205"
        # Which parts of the reduction to perform
        steps_to_take = [
            #"bias",
            #"flat",
            #"orders",
            #"norm_flat",
            #"wavecal",
            "science",
        ]

    # load configuration for the current instrument
    with open("settings_%s.json" % instrument) as f:
        config = json.load(f)

    # TODO: Test settings
    config["plot"] = False
    config["manual"] = True
    modes = config["modes"][1:2]

    # Search the available days
    dates = join(base_dir, instrument, target, "raw", "????-??-??")
    dates = glob.glob(dates)
    dates = [r + os.sep for r in dates if os.path.isdir(r)]

    print("Instrument: ", instrument)
    print("Target: ", target)
    for night in dates:
        night = os.path.basename(night[:-1])
        print("Observation Date: ", night)
        for mode in modes:
            print("Instrument Mode: ", mode)
            main(target, instrument, mode, night, config, steps=steps_to_take)
