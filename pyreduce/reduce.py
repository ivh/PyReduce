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
import joblib

from . import echelle, instruments, util

# PyReduce subpackages
from .configuration import load_config
from .combine_frames import combine_bias, combine_flat
from .continuum_normalization import continuum_normalize, splice_orders
from .extract import extract
from .make_shear import Curvature as CurvatureModule
from .normalize_flat import normalize_flat
from .trace_orders import mark_orders
from .wavelength_calibration import WavelengthCalibration as WavelengthCalibrationModule

from .extraction_width import estimate_extraction_width

# TODO turn dicts into numpy structured array
# TODO Naming of functions and modules
# TODO License

# TODO automatic determination of the extraction width


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
            base_dir = config["reduce"]["base_dir"]
        if isNone["input_dir"]:
            input_dir = config["reduce"]["input_dir"]
        if isNone["output_dir"]:
            output_dir = config["reduce"]["output_dir"]

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
                        input_dir, t, n, i, m, **config["instrument"]
                    )
                    if len(files) == 0:
                        logging.warning("No files found for instrument:%s, target:%s, night:%s, mode:%s", i, t, n, m)
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
                            logging.debug("Wavecal files:\n%s", f[key]["wavecal"])
                            logging.debug("Orderdef files:\n%s", f[key]["orders"])
                            logging.debug("Science files:\n%s", f[key]["science"])
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


class Step:
    def __init__(
        self,
        instrument,
        mode,
        extension,
        target,
        night,
        output_dir,
        order_range,
        **config,
    ):
        self._dependsOn = []
        self._loadDependsOn = []
        self.instrument = instrument
        self.mode = mode
        self.extension = extension
        self.target = target
        self.night = night
        self.order_range = order_range
        self.plot = config.get("plot", False)
        self._output_dir = output_dir

    def run(self, files, *args):
        raise NotImplementedError

    def save(self, *args):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    @property
    def dependsOn(self):
        return self._dependsOn

    @property
    def loadDependsOn(self):
        return self._loadDependsOn

    @property
    def output_dir(self):
        """str: output directory, may contain tags {instrument}, {night}, {target}, {mode}"""
        return self._output_dir.format(
            instrument=self.instrument,
            target=self.target,
            night=self.night,
            mode=self.mode,
        )

    @property
    def prefix(self):
        """str: temporary file prefix"""
        i = self.instrument.lower()
        m = self.mode.lower()
        return f"{i}_{m}"


class Mask(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self.extension = 0
        self._mask_dir = config["directory"]

    @property
    def mask_dir(self):
        this = os.path.dirname(__file__)
        return self._mask_dir.format(reduce=this)

    @property
    def mask_file(self):
        i = self.instrument.lower()
        m = self.mode
        return f"mask_{i}_{m}.fits.gz"

    def run(self):
        return self.load()

    def save(self, mask):
        mask_file = join(self.mask_dir, self.mask_file)
        fits.writeto(mask_file, data=(~mask).astype(int))

    def load(self):
        mask_file = join(self.mask_dir, self.mask_file)
        mask, _ = util.load_fits(
            mask_file, self.instrument, self.mode, extension=self.extension
        )
        mask = ~mask.data.astype(bool)  # REDUCE mask are inverse to numpy masks
        return mask


class Bias(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["mask"]
        self._loadDependsOn += ["mask"]

    @property
    def savefile(self):
        """str: Name of master bias fits file"""
        return join(self.output_dir, self.prefix + ".bias.fits")

    def save(self, bias, bhead):
        bias = np.asarray(bias, dtype=np.float32)
        fits.writeto(
            self.savefile,
            data=bias,
            header=bhead,
            overwrite=True,
            output_verify="fix+warn",
        )

    def run(self, files, mask):
        bias, bhead = combine_bias(
            files,
            self.instrument,
            self.mode,
            mask=mask,
            extension=self.extension,
            plot=self.plot,
        )

        self.save(bias.data, bhead)

        return bias, bhead

    def load(self, mask):
        bias = fits.open(self.savefile)[0]
        bias, bhead = bias.data, bias.header
        bias = np.ma.masked_array(bias, mask=mask)
        return bias, bhead


class Flat(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["mask", "bias"]
        self._loadDependsOn += ["mask"]

    @property
    def savefile(self):
        """str: Name of master bias fits file"""
        return join(self.output_dir, self.prefix + ".flat.fits")

    def save(self, flat, fhead):
        flat = np.asarray(flat, dtype=np.float32)
        fits.writeto(
            self.savefile,
            data=flat,
            header=fhead,
            overwrite=True,
            output_verify="fix+warn",
        )

    def run(self, files, bias, mask):
        bias, bhead = bias
        flat, fhead = combine_flat(
            files,
            self.instrument,
            self.mode,
            mask=mask,
            extension=self.extension,
            bias=bias,
            plot=self.plot,
        )

        self.save(flat.data, fhead)
        return flat, fhead

    def load(self, mask):
        flat = fits.open(self.savefile)[0]
        flat, fhead = flat.data, flat.header
        flat = np.ma.masked_array(flat, mask=mask)
        return flat, fhead


class OrderTracing(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["mask"]

        self.min_cluster = config["min_cluster"]
        self.filter_size = config["filter_size"]
        self.noise = config["noise"]
        self.fit_degree = config["degree"]
        self.border_width = config["border_width"]
        self.manual = config["manual"]

    @property
    def savefile(self):
        """str: Name of the order tracing file"""
        return join(self.output_dir, self.prefix + ".ord_default.npz")

    def run(self, files, mask):
        order_img, _ = util.load_fits(
            files[0], self.instrument, self.mode, self.extension, mask=mask
        )

        orders, column_range = mark_orders(
            order_img,
            min_cluster=self.min_cluster,
            filter_size=self.filter_size,
            noise=self.noise,
            opower=self.fit_degree,
            border_width=self.border_width,
            manual=self.manual,
            plot=self.plot,
        )

        self.save(orders, column_range)

        return orders, column_range

    def save(self, orders, column_range):
        np.savez(self.savefile, orders=orders, column_range=column_range, allow_pickle=True)

    def load(self):
        data = np.load(self.savefile, allow_pickle=True)
        orders = data["orders"]
        column_range = data["column_range"]
        return orders, column_range


class NormalizeFlatField(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["flat", "orders"]

        self.extraction_method = config["extraction_method"]
        if self.extraction_method == "normalize":
            self.extraction_kwargs = {
                "extraction_width": config["extraction_width"],
                "lambda_sf": config["smooth_slitfunction"],
                "lambda_sp": config["smooth_spectrum"],
                "osample": config["oversampling"],
                "swath_width": config["swath_width"]
            }
        else:
            raise ValueError(f"Extraction method {self.extraction_method} not supported for step 'norm_flat'")

        self.scatter_degree = config["scatter_degree"]
        self.threshold = config["threshold"]

    @property
    def savefile(self):
        """str: Name of the blaze file"""
        return join(self.output_dir, self.prefix + ".flat_norm.npz")

    def run(self, flat, orders):
        flat, fhead = flat
        orders, column_range = orders

        norm, blaze = normalize_flat(
            flat,
            orders,
            gain=fhead["e_gain"],
            readnoise=fhead["e_readn"],
            dark=fhead["e_drk"],
            column_range=column_range,
            order_range=self.order_range,
            scatter_degree=self.scatter_degree,
            threshold=self.threshold,
            plot=self.plot,
            **self.extraction_kwargs
        )

        blaze = np.ma.filled(blaze, 0)

        # Save data
        self.save(norm, blaze)

        return norm, blaze

    def save(self, norm, blaze):
        np.savez(self.savefile, blaze=blaze, norm=norm, allow_pickle=True)

    def load(self):
        logging.info("Loading normalized flat field")
        data = np.load(self.savefile, allow_pickle=True)
        blaze = data["blaze"]
        norm = data["norm"]
        return norm, blaze


class WavelengthCalibration(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["mask", "orders"]

        self.extraction_method = config["extraction_method"]

        if self.extraction_method == "arc":
            self.extraction_kwargs = {"extraction_width": config["extraction_width"]}
        elif self.extraction_method == "optimal":
            self.extraction_kwargs = {
                "extraction_width": config["extraction_width"],
                "lambda_sf": config["smooth_slitfunction"],
                "lambda_sp": config["smooth_spectrum"],
                "osample": config["oversampling"],
                "swath_width": config["swath_width"]
            }
        else:
            raise ValueError(f"Extraction method {self.extraction_method} not supported for step 'wavecal'")

        self.degree = config["degree"]
        self.manual = config["manual"]
        self.threshold = config["threshold"]
        self.iterations = config["iterations"]
        self.wavecal_mode = config["dimensionality"]
        self.shift_window = config["shift_window"]

    @property
    def savefile(self):
        """str: Name of the wavelength echelle file"""
        return join(self.output_dir, self.prefix + ".thar.npz")

    def run(self, files, orders, mask):
        orders, column_range = orders

        if len(files) == 0:
            raise FileNotFoundError("No wavecal files given")
        f = files[0]
        if len(files) > 1:
            # TODO: Give the user the option to select one?
            logging.warning(
                "More than one wavelength calibration file found. Will use: %s", f
            )

        # Load wavecal image
        thar, thead = util.load_fits(
            f, self.instrument, self.mode, self.extension, mask=mask
        )

        # Extract wavecal spectrum
        thar, _, _, _ = extract(
            thar,
            orders,
            gain=thead["e_gain"],
            readnoise=thead["e_readn"],
            dark=thead["e_drk"],
            column_range=column_range,
            extraction_type=self.extraction_method,
            order_range=self.order_range,
            plot=self.plot,
            **self.extraction_kwargs,
        )

        # load reference linelist
        reference = instruments.instrument_info.get_wavecal_filename(
            thead, self.instrument, self.mode
        )
        reference = np.load(reference, allow_pickle=True)
        linelist = reference["cs_lines"]

        module = WavelengthCalibrationModule(
            plot=self.plot,
            manual=self.manual,
            degree=self.degree,
            threshold=self.threshold,
            iterations=self.iterations,
            mode=self.wavecal_mode,
            shift_window=self.shift_window,
        )
        wave, coef = module.execute(thar, linelist)

        self.save(wave, thar, coef, linelist)

        return wave, thar, coef, linelist

    def save(self, wave, thar, coef, linelist):
        np.savez(self.savefile, wave=wave, thar=thar, coef=coef, linelist=linelist, allow_pickle=True)

    def load(self):
        data = np.load(self.savefile, allow_pickle=True)
        wave = data["wave"]
        thar = data["thar"]
        coef = data["coef"]
        linelist = data["linelist"]
        return wave, thar, coef, linelist


# TODO somehow this is part of the wavelength calibration, and not its own step
class LaserFrequencyComb(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["wavecal", "orders", "mask"]
        self._loadDependsOn += ["wavecal"]

        self.extraction_method = config["extraction_method"]

        if self.extraction_method == "arc":
            self.extraction_kwargs = {"extraction_width": config["extraction_width"]}
        elif self.extraction_method == "optimal":
            self.extraction_kwargs = {
                "extraction_width": config["extraction_width"],
                "lambda_sf": config["smooth_slitfunction"],
                "lambda_sp": config["smooth_spectrum"],
                "osample": config["oversampling"],
                "swath_width": config["swath_width"]
            }
        else:
            raise ValueError(f"Extraction method {self.extraction_method} not supported for step 'freq_comb'")

        self.degree = config["degree"]
        self.extraction_width = config["extraction_width"]
        self.threshold = config["threshold"]
        self.wavecal_mode = config["dimensionality"]
        self.lfc_peak_width = config["peak_width"]

    @property
    def savefile(self):
        """str: Name of the wavelength echelle file"""
        return join(self.output_dir, self.prefix + ".comb.npz")

    def run(self, files, wavecal, orders, mask):
        wave, thar, coef, linelist = wavecal
        orders, column_range = orders

        f = files[0]
        comb, chead = util.load_fits(
            f, self.instrument, self.mode, self.extension, mask=mask
        )

        comb, _, _, _ = extract(
            comb,
            orders,
            gain=chead["e_gain"],
            readnoise=chead["e_readn"],
            dark=chead["e_drk"],
            extraction_type=self.extraction_method,
            column_range=column_range,
            order_range=self.order_range,
            plot=self.plot,
            **self.extraction_kwargs
        )

        # for i in range(len(comb)):
        #     comb[i] -= comb[i][comb[i] > 0].min()
        #     comb[i] /= blaze[i] * comb[i].max() / blaze[i].max()

        module = WavelengthCalibrationModule(
            plot=self.plot,
            degree=self.degree,
            threshold=self.threshold,
            mode=self.wavecal_mode,
            lfc_peak_width=self.lfc_peak_width,
        )
        wave = module.frequency_comb(comb, wave, linelist)

        self.save(wave, comb)

        return wave, comb

    def save(self, wave, comb):
        np.savez(self.savefile, wave=wave, comb=comb, allow_pickle=True)

    def load(self, wavecal):
        try:
            data = np.load(self.savefile, allow_pickle=True)
        except FileNotFoundError:
            logging.warning(
                "No data for Laser Frequency Comb found, using regular wavelength calibration instead"
            )
            wave, thar, coef, linelist = wavecal
            data = {"wave": wave, "comb": None}
        wave = data["wave"]
        comb = data["comb"]
        return wave, comb


class SlitCurvatureDetermination(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["orders", "wavecal", "mask"]

        self.extraction_width = config["extraction_width"]
        self.fit_degree = config["degree"]
        self.max_iter = config["iterations"]
        self.sigma_cutoff = config["sigma_cutoff"]
        self.curvature_mode = config["dimensionality"]

    @property
    def savefile(self):
        """str: Name of the tilt/shear save file"""
        return join(self.output_dir, self.prefix + ".shear.npz")

    def run(self, files, orders, wavecal, mask):
        orders, column_range = orders
        wave, thar, coef, linelist = wavecal

        # TODO: Pick best image / combine images ?
        f = files[0]
        orig, _ = util.load_fits(
            f, self.instrument, self.mode, self.extension, mask=mask
        )

        module = CurvatureModule(
            orders,
            column_range=column_range,
            extraction_width=self.extraction_width,
            order_range=self.order_range,
            fit_degree=self.fit_degree,
            max_iter=self.max_iter,
            sigma_cutoff=self.sigma_cutoff,
            mode=self.curvature_mode,
            plot=self.plot,
        )
        tilt, shear = module.execute(thar, orig)
        self.save(tilt, shear)

        return tilt, shear

    def save(self, tilt, shear):
        np.savez(self.savefile, tilt=tilt, shear=shear, allow_pickle=True)

    def load(self):
        try:
            data = np.load(self.savefile, allow_pickle=True)
        except FileNotFoundError:
            logging.warning("No data for slit curvature found, setting it to 0.")
            data = {"tilt": None, "shear": None}

        tilt = data["tilt"]
        shear = data["shear"]
        return tilt, shear


class ScienceExtraction(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["mask", "bias", "orders", "norm_flat", "curvature"]
        self._loadDependsOn += []

        self.extraction_method = config["extraction_method"]
        if self.extraction_method == "arc":
            self.extraction_kwargs = {"extraction_width": config["extraction_width"]}
        elif self.extraction_method == "optimal":
            self.extraction_kwargs = {
                "extraction_width": config["extraction_width"],
                "lambda_sf": config["smooth_slitfunction"],
                "lambda_sp": config["smooth_spectrum"],
                "osample": config["oversampling"],
                "swath_width": config["swath_width"]
            }
        else:
            raise ValueError(f"Extraction method {self.extraction_method} not supported for step 'science'")

    def science_file(self, name):
        return util.swap_extension(name, ".science.ech", path=self.output_dir)

    def run(self, files, bias, orders, norm_flat, curvature, mask):
        bias, bhead = bias
        norm, blaze = norm_flat
        orders, column_range = orders
        tilt, shear = curvature

        heads, specs, sigmas, columns = [], [], [], []
        for fname in files:
            im, head = util.load_fits(
                fname,
                self.instrument,
                self.mode,
                self.extension,
                mask=mask,
                dtype=np.floating,
            )
            # Correct for bias and flat field
            im -= bias
            im /= norm

            # Optimally extract science spectrum
            spec, sigma, _, column_range = extract(
                im,
                orders,
                tilt=tilt,
                shear=shear,
                gain=head["e_gain"],
                readnoise=head["e_readn"],
                dark=head["e_drk"],
                extraction_type=self.extraction_method,
                column_range=column_range,
                order_range=self.order_range,
                plot=self.plot,
                **self.extraction_kwargs
            )

            # save spectrum to disk
            self.save(fname, head, spec, sigma, column_range)
            heads.append(head)
            specs.append(spec)
            sigmas.append(sigma)
            columns.append(column_range)

        return heads, specs, sigmas, columns

    def save(self, fname, head, spec, sigma, column_range):
        nameout = self.science_file(fname)
        echelle.save(nameout, head, spec=spec, sig=sigma, columns=column_range)

    def load(self):
        files = [s for s in os.listdir(self.output_dir) if s.endswith(".science.ech")]

        heads, specs, sigmas, columns = [], [], [], []
        for fname in files:
            fname = join(self.output_dir, fname)
            science = echelle.read(
                fname,
                continuum_normalization=False,
                barycentric_correction=False,
                radial_velociy_correction=False,
            )
            heads.append(science.header)
            specs.append(science["spec"])
            sigmas.append(science["sig"])
            columns.append(science["columns"])

        return heads, specs, sigmas, columns


class ContinuumNormalization(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["science", "freq_comb", "norm_flat"]

    @property
    def savefile(self):
        return join(self.output_dir, self.prefix + ".cont.npz")

    def run(self, science, freq_comb, norm_flat):
        wave, comb = freq_comb
        heads, specs, sigmas, columns = science
        norm, blaze = norm_flat

        logging.info("Continuum normalization")
        conts = [None for _ in specs]
        for j, (spec, sigma) in enumerate(zip(specs, sigmas)):
            logging.info("Splicing orders")
            specs[j], wave, blaze, sigmas[j] = splice_orders(
                spec, wave, blaze, sigma, scaling=True, plot=self.plot
            )
            logging.info("Normalizing continuum")
            conts[j] = continuum_normalize(
                specs[j], wave, blaze, sigmas[j], plot=self.plot
            )

        self.save(heads, specs, sigmas, conts, columns)
        return heads, specs, sigmas, conts, columns

    def save(self, heads, specs, sigmas, conts, columns):
        value = {
            "heads": heads,
            "specs": specs,
            "sigmas": sigmas,
            "conts": conts,
            "columns": columns,
        }
        joblib.dump(value, self.savefile)

    def load(self):
        data = joblib.load(self.savefile)
        heads = data["heads"]
        specs = data["specs"]
        sigmas = data["sigmas"]
        conts = data["conts"]
        columns = data["columns"]
        return heads, specs, sigmas, conts, columns


class Finalize(Step):
    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["continuum", "freq_comb"]

    def output_file(self, number):
        out = f"{self.instrument.upper()}.{self.night}_{number}.ech"
        return os.path.join(self.output_dir, out)

    def run(self, continuum, freq_comb):
        heads, specs, sigmas, conts, columns = continuum
        wave, comb = freq_comb

        # Combine science with wavecal and continuum
        for i, (head, spec, sigma, blaze) in enumerate(
            zip(heads, specs, sigmas, conts)
        ):
            head["e_erscle"] = ("absolute", "error scale")

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

            if self.plot:
                for j in range(spec.shape[0]):
                    plt.plot(wave[j], spec[j] / blaze[j])
                plt.show()

            fname = self.save(i, head, spec, sigma, blaze, wave, columns[i])
            logging.info("science file: %s", os.path.basename(fname))

    def save(self, i, head, spec, sigma, cont, wave, columns):
        out_file = self.output_file(i)
        echelle.save(
            out_file, head, spec=spec, sig=sigma, cont=cont, wave=wave, columns=columns
        )
        return out_file


class Reducer:

    step_order = {
        "bias": 10,
        "flat": 20,
        "orders": 30,
        "norm_flat": 40,
        "wavecal": 50,
        "freq_comb": 60,
        "curvature": 70,
        "science": 80,
        "continuum": 90,
        "finalize": 100,
    }

    modules = {
        "mask": Mask,
        "bias": Bias,
        "flat": Flat,
        "orders": OrderTracing,
        "norm_flat": NormalizeFlatField,
        "wavecal": WavelengthCalibration,
        "freq_comb": LaserFrequencyComb,
        "curvature": SlitCurvatureDetermination,
        "science": ScienceExtraction,
        "continuum": ContinuumNormalization,
        "finalize": Finalize,
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
        files: dict{str:str}
            Data files for each step
        output_dir : str
            directory to place output files in
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
        """
        #:dict(str:str): Filenames sorted by usecase
        self.files = files
        self.output_dir = output_dir.format(
            instrument=instrument, target=target, night=night, mode=mode
        )

        info = instruments.instrument_info.get_instrument_info(instrument)
        imode = util.find_first_index(info["modes"], mode)
        extension = info["extension"][imode]

        self.data = {}
        self.inputs = (
            instrument,
            mode,
            extension,
            target,
            night,
            output_dir,
            order_range,
        )
        self.config = config

    def run_module(self, step, load=False):
        # The Module this step is based on (An object of the Step class)
        module = self.modules[step](*self.inputs, **self.config.get(step, {}))

        # Load the dependencies necessary for loading/running this step
        dependencies = module.dependsOn if not load else module.loadDependsOn
        for dependency in dependencies:
            if dependency not in self.data.keys():
                self.data[dependency] = self.run_module(dependency, load=True)
        args = {d: self.data[d] for d in dependencies}

        # Try to load the data, if the step is not specifically given as necessary
        # If the intermediate data is not available, run it normally instead
        # But give a warning
        if load:
            try:
                logging.info("Loading data from step '%s'", step)
                data = module.load(**args)
            except FileNotFoundError:
                logging.warning(
                    "Intermediate File(s) for loading step {step} not found. Running it instead."
                )
                data = self.run_module(step, load=False)
        else:
            logging.info("Running step '%s'", step)
            if step in self.files.keys():
                args["files"] = self.files[step]
            data = module.run(**args)

        self.data[step] = data
        return data

    def prepare_output_dir(self):
        # create output folder structure if necessary
        output_dir = self.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def run_steps(self, steps="all"):
        """
        Execute the steps as required

        Parameters
        ----------
        steps : {tuple(str), "all"}, optional
            which steps of the reduction process to perform
            the possible steps are: "bias", "flat", "orders", "norm_flat", "wavecal", "freq_comb", 
            "curvature", "science", "continuum", "finalize"
            alternatively set steps to "all", which is equivalent to setting all steps
        """
        self.prepare_output_dir()

        if steps == "all":
            steps = list(self.step_order.keys())

        steps = list(steps)
        steps.sort(key=lambda x: self.step_order[x])

        for step in steps:
            self.run_module(step)

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
