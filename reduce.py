"""
REDUCE script for spectrograph data

Authors
-------
Ansgar Wehrhahn  (ansgar.wehrhahn@physics.uu.se)
Thomas Marquart  (thomas.marquart@physics.uu.se)
Alexis Lavail    (alexis.lavail@physics.uu.se)
Nikolai Piskunov (nikolai.piskunov@physics.uu.se)

Version
-------
1.0 - Initial PyReduce

License
--------
...

"""

import glob
import json
import logging
import os.path
from os.path import join
import pickle
import sys
import time

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav

import PyReduce.echelle as echelle
import PyReduce.util as util

# PyReduce subpackages
from PyReduce.combine_frames import combine_bias, combine_flat
from PyReduce.continuum_normalization import splice_orders, continuum_normalize
from PyReduce.extract import extract
from PyReduce.instruments import instrument_info
from PyReduce.normalize_flat import normalize_flat
from PyReduce.trace_orders import mark_orders
from PyReduce.wavelength_calibration import wavecal
from PyReduce.make_shear import make_shear

# from getxwd import getxwd

# TODO Jupyter Notebook with example usage
# TODO turn dicts into numpy structured array
# TODO use masked array instead of column_range ? or use a mask instead of column range
# TODO figure out relative imports
# TODO Naming of functions and modules
# TODO License

# TODO order tracing: does it work well enough?
# TODO wavelength calibration: automatic alignment parameters, use gaussian process to model wavelengths?


def main(
    instrument="UVES",
    target="HD132205",
    steps=(
        # "bias",
        # "flat",
        # "orders",
        # "norm_flat",
        "wavecal",
        # "science",
        # "continuum",
    ),
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
    # base_dir = "/DATA/ESO_Archive/"
    base_dir = "./Test/"
    input_dir = base_dir + "{instrument}/{target}/raw/{night}"
    output_dir = base_dir + "{instrument}/{target}/reduced/{night}/Reduced_{mode}"

    log_file = "logs/%s.log" % target
    util.start_logging(log_file)

    # config: paramters for the current reduction
    # info: constant, instrument specific parameters
    with open("settings_%s.json" % instrument.upper()) as f:
        config = json.load(f)
    info = instrument_info.get_instrument_info(instrument)

    # TODO: Test settings
    config["plot"] = True
    config["manual"] = True
    modes = [info["modes"][1]]

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

            input_dir_night = input_dir.format(
                instrument=instrument, target=target, night=night, mode=mode
            )

            # find input files and sort them by type
            files = glob.glob(join(input_dir_night, "%s.*.fits" % instrument))
            files += glob.glob(join(input_dir_night, "%s.*.fits.gz" % instrument))
            files = np.array(files)

            f_bias, f_flat, f_wave, f_order, f_spec = instrument_info.sort_files(
                files, target, night, instrument, mode, **config
            )
            logging.debug("Bias files:\n%s", str(f_bias))
            logging.debug("Flat files:\n%s", str(f_flat))
            logging.debug("Wavecal files:\n%s", str(f_wave))
            logging.debug("Orderdef files:\n%s", str(f_order))
            logging.debug("Science files:\n%s", str(f_spec))

            if isinstance(f_spec, dict):
                for key, _ in f_spec.items():
                    run_steps(
                        f_bias[key],
                        f_flat[key],
                        f_wave[key],
                        f_order[key],
                        f_spec[key],
                        output_dir,
                        target,
                        instrument,
                        mode,
                        night,
                        config,
                        info,
                        steps=steps,
                    )
            else:
                run_steps(
                    f_bias,
                    f_flat,
                    f_wave,
                    f_order,
                    f_spec,
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
    f_bias,
    f_flat,
    f_wave,
    f_order,
    f_spec,
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
        the possible steps are: "bias", "flat", "orders", "norm_flat", "wavecal", "science", "continuum"
        alternatively set steps to "all", which is equivalent to setting all steps
        Note that the later steps require the previous intermediary products to exist and raise an exception otherwise
    mask_dir : str, optional
        directory containing the masks, defaults to predefined REDUCE masks
    """

    imode = util.find_first_index(info["modes"], mode)

    # read configuration settings
    extension = info["extension"][imode]
    prefix = "%s_%s" % (instrument.lower(), mode.lower())

    # define paths
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

    # ==========================================================================
    # Read mask
    # the mask is not stored with the data files (it is not supported by astropy)
    mask, _ = util.load_fits(mask_file, instrument, mode, extension=0)
    mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks

    # ==========================================================================
    # Create master bias
    if "bias" in steps or steps == "all":
        logging.info("Creating master bias")
        bias, bhead = combine_bias(
            f_bias,
            instrument,
            mode,
            mask=mask,
            extension=extension,
            plot=config.get("plot", False),
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
            f_flat,
            instrument,
            mode,
            mask=mask,
            extension=extension,
            bias=bias,
            plot=config.get("plot", False),
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

        order_img, _ = util.load_fits(
            f_order[0], instrument, mode, extension, mask=mask
        )

        orders, column_range = mark_orders(
            order_img,
            min_cluster=config.get("orders_threshold", 500),
            filter_size=config.get("orders_filter", 120),
            noise=config.get("orders_noise", 8),
            opower=config.get("orders_opower", 4),
            manual=config.get("orders_manual", True),
            plot=config.get("plot", True),
        )

        # Save results
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
        order_range = (0, len(orders) - 1)

        flat, blaze = normalize_flat(
            flat,
            orders,
            gain=fhead["e_gain"],
            readnoise=fhead["e_readn"],
            dark=fhead["e_drk"],
            column_range=column_range,
            order_range=order_range,
            extraction_width=config.get("normflat_extraction_width", 0.2),
            degree=config.get("normflat_scatter_degree", 4),
            threshold=config.get("normflat_threshold", 10000),
            lambda_sf=config.get("normflat_lambda_sf", 8),
            lambda_sp=config.get("normflat_lambda_sp", 0),
            swath_width=config.get("normflat_swath_width", None),
            plot=config.get("plot", True),
        )

        # Save data
        with open(blaze_file, "wb") as file:
            pickle.dump(blaze, file)
        fits.writeto(norm_flat_file, data=flat.data, header=fhead, overwrite=True)
    else:
        logging.info("Loading normalized flat field")
        flat = fits.open(norm_flat_file)[0]
        flat, fhead = flat.data, flat.header
        flat = np.ma.masked_array(flat, mask=mask)

        with open(blaze_file, "rb") as file:
            blaze = pickle.load(file)

    # Fix column ranges
    for i in range(blaze.shape[0]):
        column_range[i] = np.where(blaze[i] != 0)[0][[0, -1]]

    # ==========================================================================
    # Prepare wavelength calibration

    if "wavecal" in steps or steps == "all":
        logging.info("Creating wavelength calibration")
        for f in f_wave:
            # Load wavecal image
            thar, thead = util.load_fits(f, instrument, mode, extension, mask=mask)
            orig = thar
            order_range = (0, len(orders) - 1)

            # Extract wavecal spectrum
            thar, _ = extract(
                thar,
                orders,
                gain=thead["e_gain"],
                readnoise=thead["e_readn"],
                dark=thead["e_drk"],
                extraction_type="arc",
                order_range=order_range,
                column_range=column_range,
                extraction_width=config.get("wavecal_extraction_width", 0.25),
                osample=config.get("wavecal_osample", 1),
                plot=config.get("plot", True),
            )
            thead["obase"] = (order_range[0], "base order number")

            # TODO: where to put this?
            # shear = make_shear(
            #     thar,
            #     orig,
            #     orders,
            #     extraction_width=config.get("wavecal_extraction_width", 0.25),
            #     column_range=column_range,
            #     plot=config.get("plot", True),
            # )
            shear = np.zeros_like(thar)

            # Create wavelength calibration fit
            # TODO just save the coefficients?
            reference = instrument_info.get_wavecal_filename(thead, instrument, mode)
            reference = readsav(reference)
            cs_lines = reference["cs_lines"]
            wave = wavecal(
                thar,
                cs_lines,
                plot=config.get("plot", True),
                manual=config.get("wavecal_manual", False),
            )

            nameout = util.swap_extension(f, ".thar.ech", output_dir)
            echelle.save(nameout, thead, spec=thar, wave=wave, shear=shear)
    else:
        fname = util.swap_extension(f_wave[-1], ".thar.ech", output_dir)
        thar = echelle.read(fname, raw=True)
        wave = thar.wave
        shear = thar.shear

    # ==========================================================================
    # Prepare for science spectra extraction

    if "science" in steps or steps == "all":
        logging.info("Extracting science spectra")
        for f in f_spec:
            im, head = util.load_fits(
                f, instrument, mode, extension, mask=mask, dtype=np.float32
            )
            # Correct for bias and flat field
            im -= bias
            im /= flat

            order_range = (0, len(orders) - 1)

            # Optimally extract science spectrum
            spec, sigma = extract(
                im,
                orders,
                shear=shear,
                gain=head["e_gain"],
                readnoise=head["e_readn"],
                dark=head["e_drk"],
                column_range=column_range,
                order_range=order_range,
                extraction_width=config.get("science_extraction_width", 25),
                lambda_sf=config.get("science_lambda_sf", 0.1),
                lambda_sp=config.get("science_lambda_sp", 0),
                osample=config.get("science_osample", 1),
                swath_width=config.get("science_swath_width", 300),
                plot=config.get("plot", True),
            )
            head["obase"] = (order_range[0], " base order number")

            # save spectrum to disk
            nameout = util.swap_extension(f, ".science.ech", path=output_dir)
            echelle.save(nameout, head, spec=spec, sig=sigma)
    else:
        f = f_spec[-1]
        fname = util.swap_extension(f, ".science.ech", path=output_dir)
        science = echelle.read(fname, raw=True)
        head = science.head
        spec = science.spec
        sigma = science.sig

    if "continuum" in steps or steps == "all":
        logging.info("Continuum normalization")
        for f in f_spec:
            # fix column ranges
            for i in range(spec.shape[0]):
                column_range[i] = np.where(spec[i] != 0)[0][[0, -1]] + [0, 1]

            logging.info("Splicing orders")
            spec, wave, blaze, sigma = splice_orders(
                spec,
                wave,
                blaze,
                sigma,
                column_range=column_range,
                scaling=True,
                plot=config.get("plot", True),
            )

            # spec = continuum_normalize(spec, wave, blaze, sigma)

    # Combine science with wavecal and continuum
    for f in f_spec:
        head["e_error_scale"] = "absolute"

        rv_corr, bjd = util.helcorr(
            head["e_obslon"],
            head["e_obslat"],
            head["e_obsalt"],
            head["ra"],
            head["dec"],
            head["e_jd"],
        )
        head["barycorr"] = rv_corr
        head["e_jd"] = bjd

        fname = "{instrument}.{night}.ech".format(
            instrument=instrument.upper(), night=night
        )
        fname = os.path.join(output_dir, fname)
        echelle.save(
            fname,
            head,
            spec=spec,
            sig=sigma,
            cont=blaze,
            wave=wave,
            columns=column_range,
        )
        logging.info("science file: %s", os.path.basename(fname))
        logging.debug("--------------------------------")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command Line arguments passed
        args = util.parse_args()
    else:
        # Use "default" values set in main function
        args = {}

    start = time.time()
    main(**args)
    finish = time.time()
    print("Execution time: %f s" % (finish - start))
