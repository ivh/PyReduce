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

from PyReduce import util, echelle, instruments

# PyReduce subpackages
from PyReduce.combine_frames import combine_bias, combine_flat
from PyReduce.continuum_normalization import splice_orders, continuum_normalize
from PyReduce.extract import extract
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
    night="????-??-??",
    modes="middle",
    steps=(
        "bias",
        "flat",
        "orders",
        "norm_flat",
        "wavecal",
        "science",
        "continuum",
    ),
    base_dir=None,
    input_dir=None,
    output_dir=None,
    configuration=None,
):
    """
    Main entry point for REDUCE scripts,
    default values can be changed as required if reduce is used as a script

    Finds input directories, and loops over observation nights and instrument modes

    instrument : str, list[str]
        instrument used for the observation (e.g. UVES, HARPS)
    target : str, list[str]
        the observed star, as named in the folder structure/fits headers
    night : str, list[str]
        the observation nights to reduce, as named in the folder structure. Accepts bash wildcards (i.e. *, ?), but then relies on the folder structure for restricting the nights
    modes : str, list[str], None, optional
        the instrument modes to use, if None will use all known modes for the current instrument. See instruments for possible options
    steps : {tuple(str), "all"}, optional
        which steps of the reduction process to perform
        the possible steps are: "bias", "flat", "orders", "norm_flat", "wavecal", "science"
        alternatively set steps to "all", which is equivalent to setting all steps
        Note that the later steps require the previous intermediary products to exist and raise an exception otherwise
    base_dir : str, optional
        base data directory that Reduce should work in, is prefixxed on input_dir and output_dir (default: use settings_pyreduce.json)
    input_dir : str, optional
        input directory containing raw files. Can contain placeholders {instrument}, {target}, {night}, {mode} as well as wildcards. If relative will use base_dir as root (default: use settings_pyreduce.json)
    output_dir : str, optional
        output directory for intermediary and final results. Can contain placeholders {instrument}, {target}, {night}, {mode}, but no wildcards. If relative will use base_dir as root (default: use settings_pyreduce.json)
    configuration : dict, str, list[str], dict[str], optional
        configuration file for the current run, contains parameters for different parts of reduce. Can be a path to a json file, or a dict with configurations for the different instruments. When a list, the order must be the same as instruments (default: settings_{instrument.upper()}.json)
    """
    if isinstance(instrument, str):
        instrument = [instrument]
    if isinstance(target, str):
        target = [target]
    if isinstance(night, str):
        night = [night]
    if isinstance(modes, str):
        modes = [modes]

    m_was_None = modes is None

    # some basic settings
    # Expected Folder Structure: base_dir/instrument/target/raw/night/*.fits.gz
    # Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}
    settings = util.read_config()
    if base_dir is None:
        base_dir = settings["reduce.base_dir"]
    if input_dir is None:
        input_dir = settings["reduce.input_dir"]
    if output_dir is None:
        output_dir = settings["reduce.output_dir"]

    input_dir = join(base_dir, input_dir)
    output_dir = join(base_dir, output_dir)

    # Loop over everything
    for j, i in enumerate(instrument):
        # config: paramters for the current reduction
        # info: constant, instrument specific parameters
        if configuration is None:
            config = "settings_%s.json" % i.upper()
        elif isinstance(configuration, dict):
            if i in configuration.keys():
                config = configuration[i]
        elif isinstance(configuration, list):
            config = configuration[j]

        if isinstance(config, str):
            with open(config) as f:
                config = json.load(f)

        info = instruments.instrument_info.get_instrument_info(i)

        if m_was_None:
            modes = info["modes"]

        for t in target:
            log_file = join(base_dir, "logs/%s.log" % t)
            util.start_logging(log_file)

            for n in night:
                for m in modes:
                    # find input files and sort them by type
                    files, nights = instruments.instrument_info.sort_files(
                        input_dir, t, n, i, m, **config
                    )
                    for f, k in zip(files, nights):
                        logging.info("Instrument: %s", i)
                        logging.info("Target: %s", t)
                        logging.info("Observation Date: %s", k)
                        logging.info("Instrument Mode: %s", m)

                        if not isinstance(f, dict):
                            f = {1: f}
                        for key, _ in f.items():
                            logging.info("Group Identifier: %s", key)
                            logging.debug("Bias files:\n%s", f[key]["bias"])
                            logging.debug("Flat files:\n%s", f[key]["flat"])
                            logging.debug("Wavecal files:\n%s", f[key]["wave"])
                            logging.debug("Orderdef files:\n%s", f[key]["order"])
                            logging.debug("Science files:\n%s", f[key]["spec"])
                            run_steps(
                                f[key], output_dir, t, i, m, k, config, steps=steps
                            )


def run_steps(files, output_dir, target, instrument, mode, night, config, steps="all"):
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
    info = instruments.instrument_info.get_instrument_info(instrument)
    imode = util.find_first_index(info["modes"], mode)

    # read configuration settings
    extension = info["extension"][imode]
    prefix = "%s_%s" % (instrument.lower(), mode.lower())

    # define paths
    output_dir = output_dir.format(
        instrument=instrument, target=target, night=night, mode=mode
    )

    # define intermediary product files
    bias_file = join(output_dir, prefix + ".bias.fits")
    flat_file = join(output_dir, prefix + ".flat.fits")
    norm_flat_file = join(output_dir, prefix + ".flat_norm.fits")
    blaze_file = join(output_dir, prefix + ".ord_norm.sav")
    order_file = join(output_dir, prefix + ".ord_default.sav")

    out_file = "{instrument}.{night}.ech".format(
        instrument=instrument.upper(), night=night
    )
    out_file = os.path.join(output_dir, out_file)

    # create output folder structure if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==========================================================================
    # Read mask
    # the mask is not stored with the data files (it is not supported by astropy)
    mask_dir = os.path.dirname(__file__)
    mask_dir = os.path.join(mask_dir, "masks")
    mask_file = join(mask_dir, "mask_%s_%s.fits.gz" % (instrument.lower(), mode))

    mask, _ = util.load_fits(mask_file, instrument, mode, extension=0)
    mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks

    # ==========================================================================
    # Create master bias
    if "bias" in steps or steps == "all":
        logging.info("Creating master bias")
        bias, bhead = combine_bias(
            files["bias"],
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
            files["flat"],
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
            files["order"][0], instrument, mode, extension, mask=mask
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
        for f in files["wave"]:
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
            reference = instruments.instrument_info.get_wavecal_filename(
                thead, instrument, mode
            )
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
        fname = util.swap_extension(files["wave"][-1], ".thar.ech", output_dir)
        thar = echelle.read(fname, raw=True)
        wave = thar.wave
        shear = thar.shear

    # ==========================================================================
    # Prepare for science spectra extraction

    if "science" in steps or steps == "all":
        logging.info("Extracting science spectra")
        for f in files["spec"]:
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
        f = files["spec"][-1]
        fname = util.swap_extension(f, ".science.ech", path=output_dir)
        science = echelle.read(fname, raw=True)
        head = science.head
        spec = science.spec
        sigma = science.sig

    if "continuum" in steps or steps == "all":
        logging.info("Continuum normalization")
        for f in files["spec"]:
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
    for f in files["spec"]:
        head["e_error_scale"] = "absolute"

        # Add heliocentric correction
        rv_corr, bjd = util.helcorr(
            head["e_obslon"],
            head["e_obslat"],
            head["e_obsalt"],
            head["ra"],
            head["dec"],
            head["e_jd"],
        )
        
        logging.debug("Heliocentric correction: %f km/s", rv_corr)
        logging.debug("Heliocentric Julian Date: %s", str(bjd))

        head["barycorr"] = rv_corr
        head["e_jd"] = bjd

        echelle.save(
            out_file,
            head,
            spec=spec,
            sig=sigma,
            cont=blaze,
            wave=wave,
            columns=column_range,
        )
        logging.info("science file: %s", os.path.basename(out_file))
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
