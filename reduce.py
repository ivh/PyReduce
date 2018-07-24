"""
REDUCE script for spectrograph data
"""
import glob
import json
import os.path
from os.path import join
import pickle
import sys
import logging
import time

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav

# PyReduce subpackages
from combine_frames import combine_bias, combine_flat
from extract import extract
from normalize_flat import normalize_flat
from util import (
    find_first_index,
    load_fits,
    save_fits,
    swap_extension,
    top,
    parse_args,
    start_logging,
)
from instruments import instrument_info
from trace import mark_orders  # TODO: trace is a standard library name
from getxwd import getxwd
from wavelength_calibration import wavecal
from continuum import splice_orders

# TODO turn dicts into numpy structured array
# TODO use masked array instead of column_range ?


def main(
    instrument="UVES",
    target="HD132205",
    steps=(
        # "bias", 
        # "flat",
        # "orders",
        "norm_flat",
        # "wavecal",
        # "science",
        "continuum"),
):
    """
    Main entry point for REDUCE scripts,
    default values can be changed as required if reduce is used as a script

    Finds input directories, and loops over observation nights and instrument modes

    instrument : str
        instrument used for the observation (e.g. UVES, HARPS)
    target : str
        the observed star, as named in the folder structure/fits headers
    steps : {tuple(str), "all"}, optional
        which steps of the reduction process to perform
        the possible steps are: "bias", "flat", "orders", "norm_flat", "wavecal", "science"
        alternatively set steps to "all", which is equivalent to setting all steps
        Note that the later steps require the previous intermediary products to exist and raise an exception otherwise
    """

    # some basic settings
    # Expected Folder Structure: base_dir/instrument/target/raw/night/*.fits.gz
    # Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}
    input_dir = "./Test/{instrument}/{target}/raw/{night}"
    output_dir = "./Test/{instrument}/{target}/reduced/{night}/Reduced_{mode}"

    log_file = "%s.log" % target
    start_logging(log_file)

    # config: paramters for the current reduction
    # info: constant, instrument specific parameters
    with open("settings_%s.json" % instrument.upper()) as f:
        config = json.load(f)
    info = instrument_info.get_instrument_info(instrument)

    # TODO: Test settings
    config["plot"] = True
    config["manual"] = True
    modes = info["modes"][1:2]

    # Search the available days
    dates = input_dir.format(instrument=instrument, target=target, night="????-??-??")
    dates = glob.glob(dates)
    dates = [r for r in dates if os.path.isdir(r)]

    logging.info("Instrument: %s", instrument)
    logging.info("Target: %s", target)
    for night in dates:
        night = os.path.basename(night)
        logging.info("Observation Date: %s", night)
        for mode in modes:
            logging.info("Instrument Mode: %s", mode)
            run_steps(
                input_dir,
                output_dir,
                target,
                instrument,
                mode,
                night,
                config,
                info,
                steps=steps,
            )


def run_steps(
    input_dir,
    output_dir,
    target,
    instrument,
    mode,
    night,
    config,
    info,
    steps="all",
    mask_dir="./masks",
):
    """Reduce all observations from a single night and instrument mode

    Parameters
    ----------
    input_dir : str
        input directory, may contain tags {instrument}, {night}, {target}, {mode}
    output_dir : str
        output directory, may contain tags {instrument}, {night}, {target}, {mode}
    target : str
        observed targets as used in directory names/fits headers
    instrument : str
        instrument used for observations
    mode : str
        instrument mode used (e.g. "red" or "blue" for HARPS)
    night : str
        Observation night, in the same format as used in the directory structure/file sorting
    config : dict
        numeric reduction specific settings, like pixel threshold, which may change between runs
    info : dict
        fixed instrument specific values, usually header keywords for gain, readnoise, etc.
    steps : {tuple(str), "all"}, optional
        which steps of the reduction process to perform
        the possible steps are: "bias", "flat", "orders", "norm_flat", "wavecal", "science"
        alternatively set steps to "all", which is equivalent to setting all steps
        Note that the later steps require the previous intermediary products to exist and raise an exception otherwise
    mask_dir : str, optional
        directory containing the masks, defaults to predefined REDUCE masks
    """

    imode = find_first_index(info["modes"], mode)

    # read configuration settings
    extension = info["extension"][imode]
    prefix = "%s_%s" % (instrument.lower(), mode.lower())

    # define paths
    input_dir = input_dir.format(
        instrument=instrument, target=target, night=night, mode=mode
    )
    output_dir = output_dir.format(
        instrument=instrument, target=target, night=night, mode=mode
    )

    mask_file = join(mask_dir, "mask_%s_%s.fits.gz" % (instrument.lower(), mode))

    # define intermediary product files
    bias_file = join(output_dir, prefix + ".bias.fits")
    flat_file = join(output_dir, prefix + ".flat.fits")
    norm_flat_file = join(output_dir, prefix + ".flat_norm.fits")
    blaze_file = join(output_dir, prefix + ".ord_norm.sav")
    order_file = join(output_dir, prefix + ".ord_default.sav")

    # create output folder structure if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # find input files and sort them by type
    files = glob.glob(join(input_dir, "%s.*.fits" % instrument))
    files += glob.glob(join(input_dir, "%s.*.fits.gz" % instrument))
    files = np.array(files)

    f_bias, f_flat, f_wave, f_order, f_spec = instrument_info.sort_files(
        files, target, instrument, mode, **config
    )
    logging.debug("Bias files:\n%s", str(f_bias))
    logging.debug("Flat files:\n%s", str(f_flat))
    logging.debug("Wavecal files:\n%s", str(f_wave))
    logging.debug("Orderdef files:\n%s", str(f_order))
    logging.debug("Science files:\n%s", str(f_spec))

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
        logging.info("Loading master flat")
        flat = fits.open(flat_file)[0]
        flat, fhead = flat.data, flat.header
        flat = np.ma.masked_array(flat, mask=mask)

    # ==========================================================================
    # Find default orders.

    if "orders" in steps or steps == "all":
        logging.info("Tracing orders")

        order_img, _ = load_fits(f_order[0], instrument, mode, extension, mask=mask)

        orders, column_range = mark_orders(
            order_img,
            min_cluster=config.get("orders_threshold", 500),
            filter_size=config.get("orders_filter", 120),
            noise=config.get("orders_noise", 8),
            opower=config.get("orders_opower", 4),
            manual=config.get("orders_manual", True),
            plot=config.get("plot", False),
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
        extraction_width = 0.5
        order_range = (0, len(orders) - 1)

        flat, blzcoef = normalize_flat(
            flat,
            fhead,
            orders,
            column_range=column_range,
            extraction_width=extraction_width,
            order_range=order_range,
            threshold=config.get("normflat_threshold", 10000),
            lambda_sf=config.get("normflat_sf_smooth", 8),
            lambda_sp=config.get("normflat_sp_smooth", 0),
            swath_width=config.get("normflat_swath_width", None),
            plot=config.get("plot", False),
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
        logging.info("Creating wavelength calibration")
        for f in f_wave:
            # Load wavecal image
            thar, thead = load_fits(f, instrument, mode, extension, mask=mask)

            # Determine extraction width, blaze center column, and base order
            extraction_width = 0.25
            order_range = (0, len(orders) - 1)

            # Extract wavecal spectrum
            thar, _ = extract(
                thar,
                thead,
                orders,
                extraction_type= "arc",
                extraction_width=extraction_width,
                order_range=order_range,
                column_range=column_range,
                osample=config.get("wavecal_osample", 1),
                plot=config.get("plot", False),
            )
            thead["obase"] = (order_range[0], "base order number")

            # Create wavelength calibration fit
            reference = instrument_info.get_wavecal_filename(thead, instrument, mode)
            reference = readsav(reference)
            cs_lines = reference["cs_lines"]
            wave = wavecal(
                thar,
                cs_lines,
                plot=config.get("plot", False),
                manual=config.get("wavecal_manual", False),
            )

            nameout = swap_extension(f, ".thar.ech", output_dir)
            save_fits(nameout, thead, spec=thar, wave=wave)
    else:
        fname = swap_extension(f_wave[-1], ".thar.ech", output_dir)
        thar = fits.open(fname)
        wave = thar[1].data["WAVE"][0]

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

            # Set extraction width
            extraction_width = 25
            order_range = (0, len(orders) - 1)

            # Optimally extract science spectrum
            spec, sigma = extract(
                im,
                head,
                orders,
                extraction_width=extraction_width,
                column_range=column_range,
                order_range=order_range,
                lambda_sf=config.get("science_lambda_sf", 0.1),
                lambda_sp=config.get("science_lambda_sp", 0),
                osample=config.get("science_osample", 1),
                swath_width=config.get("science_swath_width", 300),
                plot=config.get("plot", False),
            )
            head["obase"] = (order_range[0], " base order number")

            # Calculate Continuum and Error
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

            # save spectrum to disk
            nameout = swap_extension(f, ".ech", path=output_dir)
            save_fits(nameout, head, spec=spec, sig=sigma, cont=blzcoef, wave=wave)
            logging.info("science file: %s", os.path.basename(nameout))
    else:
        f = f_spec[-1]
        nameout = swap_extension(f, ".ech", path=output_dir)
        science = fits.open(nameout)[1]
        spec = science.data["SPEC"][0]
        sigma = science.data["SIG"][0]

    if "continuum" in steps or steps == "all":
        logging.info("Continuum normalization")
        for f in f_spec:
            splice_orders(spec, wave, blzcoef, sigma, column_range=column_range)


    logging.debug("--------------------------------")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command Line arguments passed
        args = parse_args()
    else:
        # Use "default" values set in main function
        args = {}

    start = time.time()
    main(**args)
    finish = time.time()
    print("Execution time: %f s" % (finish - start))
