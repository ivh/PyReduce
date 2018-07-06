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
import logging

import astropy.io.fits as fits
import matplotlib.pyplot as plt

import numpy as np

from getxwd import getxwd
from combine_frames import combine_bias, combine_flat
from extract import extract
from normalize_flat import normalize_flat
from util import find_first_index, load_fits, save_fits, swap_extension, top
from instruments.instrument_info import sort_files
from trace import mark_orders  # TODO: trace is a standard library name

# TODO turn dicts into numpy structured array


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


def start_logging(log_file="log.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Command Line output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(ch_formatter)

    # Log file settings
    file = logging.FileHandler(log_file)
    file.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file.setFormatter(file_formatter)

    logger.addHandler(ch)
    logger.addHandler(file)

    logging.captureWarnings(True)

    logging.debug("----------------------")


def main(target, instrument, mode, night, config, steps="all"):
    counter_mode = find_first_index(config["modes"], mode)

    # read configuration settings
    extension = config["extensions"][counter_mode]
    prefix = instrument.lower() + "_" + mode

    # define paths
    raw_path = input_dir.format(
        instrument=instrument, target=target, night=night, mode=mode
    )
    reduced_path = output_dir.format(
        instrument=instrument, target=target, night=night, mode=mode
    )

    mask_file = join(mask_dir, "mask_%s_%s.fits.gz" % (instrument.lower(), mode))

    # define intermediary product files
    bias_file = join(reduced_path, prefix + ".bias.fits")
    flat_file = join(reduced_path, prefix + ".flat.fits")
    norm_flat_file = join(reduced_path, prefix + ".flat_norm.fits")
    blaze_file = join(reduced_path, prefix + ".ord_norm.sav")
    order_file = join(reduced_path, prefix + ".ord_default.sav")

    # create output folder structure if necessary
    if not os.path.exists(reduced_path):
        os.makedirs(reduced_path)

    # find input files and sort them by type
    files = glob.glob(join(raw_path, "%s.*.fits" % instrument))
    files += glob.glob(join(raw_path, "%s.*.fits.gz" % instrument))
    files = np.array(files)

    f_bias, f_flat, f_wave, f_order, f_spec = sort_files(
        files, target, instrument, mode
    )

    # ==========================================================================
    # Read mask
    # the mask is not stored with the data files (it is not supported by astropy)
    mask, _ = load_fits(mask_file, instrument, mode, extension=0)
    mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks

    # ==========================================================================
    # Create master bias
    if "bias" in steps or steps == "all":
        logging.info("Creating master bias")
        bias, bhead = combine_bias(
            f_bias, instrument, mode, mask=mask, extension=extension
        )
        fits.writeto(bias_file, data=bias.data, header=bhead, overwrite=True)
    else:
        logging.info("Loading master bias")
        bias = fits.open(bias_file)[0]
        bias, bhead = bias.data, bias.header
        bias = np.ma.masked_array(bias, mask=mask)

    # ==========================================================================
    # Create master flat
    if "flat" in steps or steps == "all":
        logging.info("Creating master flat")
        flat, fhead = combine_flat(
            f_flat, instrument, mode, mask=mask, extension=extension, bias=bias
        )
        fits.writeto(flat_file, data=flat.data, header=fhead, overwrite=True)
    else:
        logging.info("Loadinging master flat")
        flat = fits.open(flat_file)[0]
        flat, fhead = flat.data, flat.header
        flat = np.ma.masked_array(flat, mask=mask)

    # ==========================================================================
    # Find default orders.

    if "orders" in steps or steps == "all":
        logging.info("Tracing orders")

        order_img, _ = load_fits(f_order[0], instrument, mode, extension, mask=mask)

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
        with open(order_file, "wb") as file:
            pickle.dump((orders, column_range), file)
    else:
        logging.info("Loading order tracing data")
        with open(order_file, "rb") as file:
            orders, column_range = pickle.load(file)

    # ==========================================================================
    # = Construct normalized flat field.

    if "norm_flat" in steps or steps == "all":
        logging.info("Normalizing flat field")
        xwd = 40
        # xwd, sxwd = getxwd(flat, orders, colrange=column_range, gauss = True, pixels=True)

        flat, blzcoef = normalize_flat(
            flat,
            fhead,
            orders,
            column_range=column_range,
            xwd=xwd,
            threshold=config.get("normflat_threshold", 10000),
            lambda_sf=config.get("normflat_sf_smooth", 8),
            lambda_sp=config.get("normflat_sp_smooth", 0),
            swath_width=config.get("normflat_swath_width", None),
        )

        # Save data
        with open(blaze_file, "wb") as file:
            pickle.dump(blzcoef, file)
        fits.writeto(norm_flat_file, data=flat.data, header=fhead, overwrite=True)
    else:
        logging.info("Loading normalized flat field")
        flat = fits.open(norm_flat_file)[0]
        flat, fhead = flat.data, flat.header
        flat = np.ma.masked_array(flat, mask=mask)

        with open(blaze_file, "rb") as file:
            blzcoef = pickle.load(file)
    # ==========================================================================
    # Prepare wavelength calibration

    if "wavecal" in steps or steps == "all":
        logging.info("Preparing wavelength calibration")
        for f in f_wave:
            # Load wavecal image
            im, head = load_fits(f, instrument, mode, extension, mask=mask)

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
        logging.info("Extracting science spectra")
        for f in f_spec:
            im, head = load_fits(
                f, instrument, mode, extension, mask=mask, dtype=np.float32
            )
            # Correct for bias and flat field
            im -= bias
            im /= flat

            # TODO may or may not work
            # xwd, sxwd = getxwd(im, orders, colrange=column_range, pixels=True)
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
                plot=False,
            )

            # Calculate Continuum and Error
            # TODO plotting
            # cont = np.full_like(sigma, 1.)

            # TODO do we even want this to happen?
            # convert uncertainty to relative error
            # Temp copy for calculation
            # sunc = np.copy(sigma)
            # sunc /= np.clip(spec, 1., None)
            # s = spec / np.clip(blzcoef, 0.001, None)

            # # fit simple continuum
            # for i in range(len(orders)):
            #     c = top(s[i][s[i] != 0], 1, eps=0.0002, poly=True)
            #     s[i][s[i] != 0] = s[i][s[i] != 0] / c
            #     c = spec[i][s[i] != 0] / s[i][s[i] != 0]
            #     cont[i][s[i] != 0] = np.copy(c)

            # sigma *= cont  # Scale Error with Continuum

            order_range = (0, len(orders))  # TODO
            head["obase"] = (order_range[0], " base order number")

            # save spectrum to disk
            nameout = swap_extension(f, ".ech", path=reduced_path)
            save_fits(nameout, head, spec=spec, sig=sigma, cont=blzcoef)

            pol_angle = head.get("eso ins ret25 pos")
            if pol_angle is None:
                pol_angle = head.get("eso ins ret50 pos")
                if pol_angle is None:
                    pol_angle = "no polarimeter"
                else:
                    pol_angle = "lin %i" % pol_angle
            else:
                pol_angle = "cir %i" % pol_angle

            logging.info(
                "star: %s, polarization: %s, mean s/n=%.2f",
                head["object"],
                pol_angle,
                1 / np.mean(sigma),
            )
            logging.info("file: %s", os.path.basename(nameout))
            logging.debug("--------------------------------")


if __name__ == "__main__":
    # some basic settings
    # Expected Folder Structure: base_dir/instrument/target/raw/night/*.fits.gz
    input_dir = "./Test/{instrument}/{target}/raw/{night}"
    output_dir = "./Test/{instrument}/{target}/reduced/{night}/Reduced_{mode}"

    mask_dir = "./masks"
    log_file = "log.log"
    start_logging(log_file)

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
            # "bias",
            # "flat",
            # "orders",
            # "norm_flat",
            # "wavecal",
            "science"
        ]

    # load configuration for the current instrument
    with open("settings_%s.json" % instrument) as f:
        config = json.load(f)

    # TODO: Test settings
    config["plot"] = False
    config["manual"] = True
    modes = config["modes"][1:2]

    # Search the available days
    dates = join("./Test", instrument, target, "raw", "????-??-??")
    dates = glob.glob(dates)
    dates = [r + os.sep for r in dates if os.path.isdir(r)]

    logging.info("Instrument: %s", instrument)
    logging.info("Target: %s", target)
    for night in dates:
        night = os.path.basename(night[:-1])
        logging.info("Observation Date: %s", night)
        for mode in modes:
            logging.info("Instrument Mode: %s", mode)
            main(target, instrument, mode, night, config, steps=steps_to_take)
