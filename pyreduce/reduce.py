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

import json
import logging
import os.path
import pickle
import sys
import time
from os.path import join

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

from . import echelle, instruments, util

# PyReduce subpackages
from .combine_frames import combine_bias, combine_flat
from .continuum_normalization import continuum_normalize, splice_orders
from .extract import extract
from .make_shear import make_shear
from .normalize_flat import normalize_flat
from .trace_orders import mark_orders
from .wavelength_calibration import wavecal, WavelengthCalibration

from .extraction_width import estimate_extraction_width

# TODO turn dicts into numpy structured array
# TODO use masked array instead of column_range ? or use a mask instead of column range
# TODO Naming of functions and modules
# TODO License

# TODO wavelength calibration: automatic alignment parameters


def main(
    instrument="UVES",
    target="HD132205",
    night="????-??-??",
    modes="middle",
    steps=("bias", "flat", "orders", "norm_flat", "wavecal", "science", "continuum"),
    base_dir=None,
    input_dir=None,
    output_dir=None,
    configuration=None,
    order_range=None,
):
    """
    Main entry point for REDUCE scripts,
    default values can be changed as required if reduce is used as a script
    Finds input directories, and loops over observation nights and instrument modes

    Parameters
    ----------
    instrument : str, list[str]
        instrument used for the observation (e.g. UVES, HARPS)
    target : str, list[str]
        the observed star, as named in the folder structure/fits headers
    night : str, list[str]
        the observation nights to reduce, as named in the folder structure. Accepts bash wildcards (i.e. \*, ?), but then relies on the folder structure for restricting the nights
    modes : str, list[str], dict[{instrument}:list], None, optional
        the instrument modes to use, if None will use all known modes for the current instrument. See instruments for possible options
    steps : tuple(str), "all", optional
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
    configuration : dict[str:obj], str, list[str], dict[{instrument}:dict,str], optional
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

    isNone = {
        "modes": modes is None,
        "base_dir": base_dir is None,
        "input_dir": input_dir is None,
        "output_dir": output_dir is None,
    }

    # Loop over everything
    for j, i in enumerate(instrument):
        # settings: default settings of PyReduce
        # config: paramters for the current reduction
        # info: constant, instrument specific parameters
        config = load_config(configuration, i, j)

        # load default settings from settings_pyreduce.json
        if isNone["base_dir"]:
            base_dir = config["reduce.base_dir"]
        if isNone["input_dir"]:
            input_dir = config["reduce.input_dir"]
        if isNone["output_dir"]:
            output_dir = config["reduce.output_dir"]

        input_dir = join(base_dir, input_dir)
        output_dir = join(base_dir, output_dir)

        info = instruments.instrument_info.get_instrument_info(i)

        if isNone["modes"]:
            mode = info["modes"]
        elif isinstance(modes, dict):
            mode = modes[i]
        else:
            mode = modes

        for t in target:
            log_file = join(base_dir, "logs/%s.log" % t)
            util.start_logging(log_file)

            for n in night:
                for m in mode:
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
                            reducer = Reducer(
                                f[key],
                                output_dir,
                                t,
                                i,
                                m,
                                k,
                                config,
                                order_range=order_range,
                            )
                            reducer.run_steps(steps=steps)


def load_config(configuration, instrument, j):
    if configuration is None:
        config = "settings_%s.json" % instrument.upper()
    elif isinstance(configuration, dict):
        config = configuration[instrument]
    elif isinstance(configuration, list):
        config = configuration[j]
    elif isinstance(configuration, str):
        config = configuration

    if isinstance(config, str):
        if os.path.isfile(config):
            logging.info("Loading configuration from %s", config)
            with open(config) as f:
                config = json.load(f)
        else:
            logging.warning(
                "No configuration found at %s, using default values", config
            )
            config = {}

    settings = util.read_config()
    nparam1 = len(settings)
    settings.update(config)
    nparam2 = len(settings)
    if nparam2 > nparam1:
        logging.warning("New parameter(s) in instrument config, Check spelling!")

    config = settings
    return settings


class Reducer:

    step_order = {
        "bias": 0,
        "flat": 1,
        "orders": 2,
        "norm_flat": 3,
        "wavecal": 4,
        "shear": 5,
        "science": 6,
        "continuum": 7,
        "finalize": 8,
    }

    def __init__(
        self,
        files,
        output_dir,
        target,
        instrument,
        mode,
        night,
        config,
        order_range=None,
    ):
        """Reduce all observations from a single night and instrument mode

        Parameters
        ----------
        output_dir : str

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
        #:dict(str:str): Filenames sorted by usecase
        self.files = files
        #:str: Name of the observed target star
        self.target = target
        #:str: Name of the observing instrument
        self.instrument = instrument
        #:str: Observing mode of the instrument
        self.mode = mode
        #:str: Literal of the observing night
        self.night = night
        #:dict(str:obj): various configuration settings for the reduction
        self.config = config
        #:tuple(int, int): the upper and lower bound of the orders to reduce, defaults to all orders
        self.order_range = order_range

        self._mask = None
        self._spec_mask = np.ma.nomask
        self._output_dir = output_dir

        info = instruments.instrument_info.get_instrument_info(instrument)
        imode = util.find_first_index(info["modes"], mode)

        #:int: Fits File extension to use, as defined by the instrument settings
        self.extension = info["extension"][imode]

        # create output folder structure if necessary
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @property
    def output_dir(self):
        """str: output directory, may contain tags {instrument}, {night}, {target}, {mode}"""
        return self._output_dir.format(
            instrument=self.instrument,
            target=self.target,
            night=self.night,
            mode=self.mode,
        )

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = str(value)

    @property
    def prefix(self):
        """str: temporary file prefix"""
        i = self.instrument.lower()
        m = self.mode.lower()
        return f"{i}_{m}"

    @property
    def bias_file(self):
        """str: Name of master bias fits file"""
        return join(self.output_dir, self.prefix + ".bias.fits")

    @property
    def flat_file(self):
        """str: Name of master flat fits file"""
        return join(self.output_dir, self.prefix + ".flat.fits")

    @property
    def norm_flat_file(self):
        """str: Name of normalized flat file"""
        return join(self.output_dir, self.prefix + ".flat_norm.fits")

    @property
    def blaze_file(self):
        """str: Name of the blaze file"""
        return join(self.output_dir, self.prefix + ".ord_norm.npz")

    @property
    def order_file(self):
        """str: Name of the order tracing file"""
        return join(self.output_dir, self.prefix + ".ord_default.npz")

    @property
    def wave_file(self):
        """str: Name of the wavelength echelle file"""
        return join(self.output_dir, self.prefix + ".thar.ech")

    @property
    def shear_file(self):
        """str: Name of the shear echelle file"""
        return join(self.output_dir, self.prefix + ".shear.npz")

    def science_file(self, name):
        return util.swap_extension(name, ".science.ech", path=self.output_dir)

    def output_file(self, number):
        out = f"{self.instrument.upper()}.{self.night}_{number}.ech"
        return os.path.join(self.output_dir, out)

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.load_mask()
        return self._mask

    def load_mask(self):
        # Read mask
        # the mask is not stored with the data files (it is not supported by astropy)
        mask_dir = os.path.dirname(__file__)
        mask_dir = os.path.join(mask_dir, "masks")
        mask_file = join(
            mask_dir, "mask_%s_%s.fits.gz" % (self.instrument.lower(), self.mode)
        )

        mask, _ = util.load_fits(mask_file, self.instrument, self.mode, extension=0)
        mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks
        return mask

    def run_bias(self):
        logging.info("Creating master bias")
        bias, bhead = combine_bias(
            self.files["bias"],
            self.instrument,
            self.mode,
            mask=self.mask,
            extension=self.extension,
            plot=self.config["plot"],
        )
        fits.writeto(self.bias_file, data=bias.data, header=bhead, overwrite=True)
        return bias, bhead

    def load_bias(self):
        if not os.path.exists(self.bias_file):
            logging.error("Bias file not found, run with bias step")
            raise FileNotFoundError("Bias file not found, run with bias step")

        logging.info("Loading master bias")
        bias = fits.open(self.bias_file)[0]
        bias, bhead = bias.data, bias.header
        bias = np.ma.masked_array(bias, mask=self.mask)
        return bias, bhead

    def run_flat(self, bias):
        logging.info("Creating master flat")
        flat, fhead = combine_flat(
            self.files["flat"],
            self.instrument,
            self.mode,
            mask=self.mask,
            extension=self.extension,
            bias=bias,
            plot=self.config["plot"],
        )
        fits.writeto(self.flat_file, data=flat.data, header=fhead, overwrite=True)
        return flat, fhead

    def load_flat(self):
        if not os.path.exists(self.flat_file):
            logging.error("Flat file not found, run with flat step")
            raise FileNotFoundError("Flat file not found, run with flat step")

        logging.info("Loading master flat")
        flat = fits.open(self.flat_file)[0]
        flat, fhead = flat.data, flat.header
        flat = np.ma.masked_array(flat, mask=self.mask)
        return flat, fhead

    def run_orders(self):
        logging.info("Tracing orders")

        file = self.files["order"][0]
        order_img, _ = util.load_fits(
            file, self.instrument, self.mode, self.extension, mask=self.mask
        )

        orders, column_range = mark_orders(
            order_img,
            min_cluster=self.config["orders.min_cluster"],
            filter_size=self.config["orders.filter_size"],
            noise=self.config["orders.noise"],
            opower=self.config["orders.fit_degree"],
            border_width=self.config["orders.border_width"],
            manual=self.config["orders.manual"],
            plot=self.config["plot"],
        )

        # Save results
        np.savez(self.order_file, orders=orders, column_range=column_range)

        return orders, column_range

    def load_orders(self):
        logging.info("Loading order tracing data")
        data = np.load(self.order_file)
        orders = data["orders"]
        column_range = data["column_range"]
        return orders, column_range

    def run_extraction_width(self, flat, orders, column_range):
        extraction_width = estimate_extraction_width(flat, orders, column_range)

        self.config["normflat.extraction_width"] = extraction_width

        return extraction_width

    def run_norm_flat(self, flat, fhead, orders, column_range):
        logging.info("Normalizing flat field")

        norm, blaze = normalize_flat(
            flat,
            orders,
            gain=fhead["e_gain"],
            readnoise=fhead["e_readn"],
            dark=fhead["e_drk"],
            column_range=column_range,
            order_range=self.order_range,
            extraction_width=self.config["normflat.extraction_width"],
            scatter_degree=self.config["normflat.scatter_degree"],
            threshold=self.config["normflat.threshold"],
            lambda_sf=self.config["normflat.smooth_slitfunction"],
            lambda_sp=self.config["normflat.smooth_spectrum"],
            swath_width=self.config["normflat.swath_width"],
            osample=self.config["normflat.oversampling"],
            plot=self.config["plot"],
        )

        blaze = np.ma.filled(blaze, 0)
        # Fix column ranges
        for i in range(blaze.shape[0]):
            j = i + self.order_range[0]
            column_range[j] = np.where(blaze[i] != 0)[0][[0, -1]]
        self._spec_mask = np.full(blaze.shape, True)
        for i, cr in enumerate(column_range[self.order_range[0] : self.order_range[1]]):
            self._spec_mask[i, cr[0] : cr[1]] = False

        blaze = np.ma.masked_array(blaze, mask=self._spec_mask)

        # Save data
        np.savez(self.blaze_file, blaze=blaze, column_range=column_range)
        fits.writeto(self.norm_flat_file, data=norm.data, header=fhead, overwrite=True)
        return norm, blaze, column_range

    def load_norm_flat(self):
        logging.info("Loading normalized flat field")
        flat = fits.open(self.norm_flat_file)[0]
        flat, _ = flat.data, flat.header
        flat = np.ma.masked_array(flat, mask=self.mask)

        data = np.load(self.blaze_file)
        blaze = data["blaze"]
        column_range = data["column_range"]

        self._spec_mask = np.full(blaze.shape, True)
        for i, cr in enumerate(column_range[self.order_range[0] : self.order_range[1]]):
            self._spec_mask[i, cr[0] : cr[1]] = False
        # self._spec_mask = blaze == 0

        blaze = np.ma.masked_array(blaze, mask=self._spec_mask)

        return flat, blaze, column_range

    def run_wavecal(self, orders, column_range):
        logging.info("Creating wavelength calibration")

        if len(self.files["wave"]) == 0:
            raise AttributeError("No wavecal files given")

        f = self.files["wave"][-1]

        if len(self.files["wave"]) > 1:
            # TODO: Give the user the option to select one?
            logging.warning(
                "More than one wavelength calibration file found. Will use: %s", f
            )

        # Load wavecal image
        thar, thead = util.load_fits(
            f, self.instrument, self.mode, self.extension, mask=self.mask
        )

        # Extract wavecal spectrum
        thar, _, _, _ = extract(
            thar,
            orders,
            gain=thead["e_gain"],
            readnoise=thead["e_readn"],
            dark=thead["e_drk"],
            extraction_type="arc",
            column_range=column_range,
            order_range=self.order_range,
            extraction_width=self.config["wavecal.extraction_width"],
            osample=self.config["wavecal.oversampling"],
            plot=self.config["plot"],
        )
        base_order = 0 if self.order_range is None else self.order_range[0]
        thead["obase"] = (base_order, "base order number")

        # Create wavelength calibration fit
        reference = instruments.instrument_info.get_wavecal_filename(
            thead, self.instrument, self.mode
        )
        reference = np.load(reference)
        linelist = reference["cs_lines"]

        module = WavelengthCalibration(
            plot=self.config["plot"],
            manual=self.config["wavecal.manual"],
            degree=(self.config["wavecal.degree.x"], self.config["wavecal.degree.y"]),
            threshold=self.config["wavecal.threshold"],
            iterations=self.config["wavecal.iterations"],
            mode=self.config["wavecal.mode"],
            shift_window=self.config["wavecal.shift_window"],
        )
        wave = module.execute(thar, linelist)

        wave = np.ma.masked_array(wave, mask=self._spec_mask)
        thar = np.ma.masked_array(thar, mask=self._spec_mask)

        echelle.save(self.wave_file, thead, spec=thar, wave=wave)
        return wave, thar

    def load_wavecal(self):
        thar = echelle.read(self.wave_file, raw=True)
        wave = thar["wave"]
        wave = np.ma.masked_array(wave, mask=self._spec_mask)
        thar = thar["spec"]
        thar = np.ma.masked_array(thar, mask=self._spec_mask)
        return wave, thar

    def run_shear(self, orders, column_range, thar):
        logging.info("Determine shear of slit curvature")

        # TODO: Pick best image / combine images ?
        f = self.files["wave"][-1]
        orig, thead = util.load_fits(
            f, self.instrument, self.mode, self.extension, mask=self.mask
        )
        tilt, shear = make_shear(
            thar,
            orig,
            orders,
            extraction_width=self.config["wavecal.extraction_width"],
            column_range=column_range,
            order_range=self.order_range,
            plot=self.config.get("plot", True),
        )
        tilt = np.ma.masked_array(tilt, mask=self._spec_mask)
        shear = np.ma.masked_array(shear, mask=self._spec_mask)

        np.savez(self.shear_file, tilt=tilt, shear=shear)
        return tilt, shear

    def load_shear(self):
        fname = self.shear_file
        data = np.load(fname)
        tilt = data["tilt"]
        shear = data["shear"]
        tilt = np.ma.masked_array(tilt, mask=self._spec_mask)
        shear = np.ma.masked_array(shear, mask=self._spec_mask)
        return tilt, shear

    def run_science(self, bias, flat, orders, tilt, shear, column_range):
        logging.info("Extracting science spectra")
        heads, specs, sigmas = [], [], []
        for f in self.files["spec"]:
            im, head = util.load_fits(
                f,
                self.instrument,
                self.mode,
                self.extension,
                mask=self.mask,
                dtype=np.float32,
            )
            # Correct for bias and flat field
            im -= bias
            im /= flat

            # Optimally extract science spectrum
            spec, sigma, _, column_range = extract(
                im,
                orders,
                tilt=tilt,
                shear=shear,
                gain=head["e_gain"],
                readnoise=head["e_readn"],
                dark=head["e_drk"],
                column_range=column_range,
                order_range=self.order_range,
                extraction_width=self.config["science.extraction_width"],
                lambda_sf=self.config["science.smooth_slitfunction"],
                lambda_sp=self.config["science.smooth_spectrum"],
                osample=self.config["science.oversampling"],
                swath_width=self.config["science.swath_width"],
                plot=self.config["plot"],
            )
            # head["obase"] = (self.order_range[0], " base order number")
            # Replace current column range mask with new mask by reference
            self._spec_mask[:] = spec.mask
            spec = np.ma.masked_array(spec.data, mask=self._spec_mask)
            sigma = np.ma.masked_array(sigma.data, mask=self._spec_mask)

            # save spectrum to disk
            nameout = self.science_file(f)
            echelle.save(nameout, head, spec=spec, sig=sigma, columns=column_range)
            heads.append(head)
            specs.append(spec)
            sigmas.append(sigma)

        return heads, specs, sigmas

    def load_science(self):
        heads, specs, sigmas = [], [], []
        for f in self.files["spec"]:
            fname = self.science_file(f)
            science = echelle.read(
                fname,
                continuum_normalization=False,
                barycentric_correction=False,
                radial_velociy_correction=False,
            )
            self._spec_mask[:] = science["mask"]
            spec = np.ma.masked_array(science["spec"].data, mask=self._spec_mask)
            sig = np.ma.masked_array(science["sig"].data, mask=self._spec_mask)

            heads.append(science.header)
            specs.append(spec)
            sigmas.append(sig)
        return heads, specs, sigmas

    def run_continuum(self, specs, sigmas, wave, blaze):
        logging.info("Continuum normalization")
        conts = [None for _ in specs]
        for j, (spec, sigma) in enumerate(zip(specs, sigmas)):
            logging.info("Splicing orders")
            specs[j], wave, blaze, sigmas[j] = splice_orders(
                spec, wave, blaze, sigma, scaling=True, plot=self.config["plot"]
            )
            logging.info("Normalizing continuum")
            conts[j] = continuum_normalize(
                specs[j], wave, blaze, sigmas[j], plot=self.config["plot"]
            )
        return specs, sigmas, wave, conts

    def run_finalize(self, specs, heads, wave, conts, sigmas, column_range):
        # Combine science with wavecal and continuum
        for i, (head, spec, sigma, blaze) in enumerate(
            zip(heads, specs, sigmas, conts)
        ):
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

            if self.config["plot"]:
                for j in range(spec.shape[0]):
                    plt.plot(wave[j], spec[j] / blaze[j])
                plt.show()

            out_file = self.output_file(i)
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

    def run_steps(self, steps="all"):

        last_step = "finalize"
        if steps != "all":
            new_steps = list(steps)
            new_steps.sort(key=lambda x: self.step_order[x])
            last_step = new_steps[-1]

        # TODO some logic that will stop the run after all requested steps are done
        if "bias" in steps or steps == "all":
            bias, _ = self.run_bias()
        else:
            bias, _ = self.load_bias()
        if last_step == "bias":
            return

        if "flat" in steps or steps == "all":
            flat, fhead = self.run_flat(bias)
        else:
            flat, fhead = self.load_flat()
        if last_step == "flat":
            return

        if "orders" in steps or steps == "all":
            orders, column_range = self.run_orders()
        else:
            orders, column_range = self.load_orders()
        if last_step == "orders":
            return

        # extraction_width = self.run_extraction_width(flat, orders, column_range)

        if "norm_flat" in steps or steps == "all":
            norm, blaze, column_range = self.run_norm_flat(
                flat, fhead, orders, column_range
            )
        else:
            norm, blaze, column_range = self.load_norm_flat()
        if last_step == "norm_flat":
            return

        if "wavecal" in steps or steps == "all":
            wave, thar = self.run_wavecal(orders, column_range)
        else:
            wave, thar = self.load_wavecal()
        if last_step == "wavecal":
            return

        if "shear" in steps or steps == "all":
            tilt, shear = self.run_shear(orders, column_range, thar)
        else:
            tilt, shear = self.load_shear()
        if last_step == "shear":
            return

        if "science" in steps or steps == "all":
            heads, specs, sigmas = self.run_science(
                bias, norm, orders, tilt, shear, column_range
            )
        else:
            heads, specs, sigmas = self.load_science()
        if last_step == "science":
            return

        if "continuum" in steps or steps == "all":
            specs, sigmas, wave, conts = self.run_continuum(specs, sigmas, wave, blaze)
        else:
            conts = [blaze for _ in specs]
        if last_step == "continuum":
            return

        if "finalize" in steps or steps == "all":
            self.run_finalize(specs, heads, wave, conts, sigmas, column_range)

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
