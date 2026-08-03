"""Continuum normalization and final output steps."""

import logging
import os
from datetime import UTC, datetime
from os.path import join

import joblib
import matplotlib.pyplot as plt
import numpy as np

# PyReduce subpackages
from .. import util
from ..continuum_normalization import continuum_normalize, splice_orders
from ..spectra import Spectra, Spectrum
from .base import (
    Step,
    wavelengths_from_traces,
)

logger = logging.getLogger(__name__)


class ContinuumNormalization(Step):
    """Determine the continuum to each observation"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["science", "norm_flat", "trace"]
        self._loadDependsOn += ["norm_flat", "science"]

    @property
    def savefile(self):
        """str: savefile name"""
        return join(self.output_dir, self.prefix + ".cont.npz")

    def run(self, science, norm_flat, trace: list):
        """Determine the continuum to each observation
        Also splices the orders together

        Parameters
        ----------
        science : tuple
            results from science step: (heads, list[list[Spectrum]])
        norm_flat : tuple
            results from the normalized flatfield step
        trace : list[TraceData]
            Trace objects with wavelength polynomials

        Returns
        -------
        heads : list(FITS header)
            FITS headers of each observation
        specs : list(array of shape (ntrace, ncol))
            extracted spectra
        sigmas : list(array of shape (ntrace, ncol))
            uncertainties of the extracted spectra
        conts : list(array of shape (ntrace, ncol))
            continuum for each spectrum
        columns : list(array of shape (ntrace, 2))
            column ranges for each spectra
        """
        norm, blaze, *_ = norm_flat

        # Select same traces as science step (fiber/group selection + trace_range)
        selected = self._select_traces(trace, "science")
        trace_list = [t for traces in selected.values() for t in traces]
        if self.trace_range is not None:
            trace_list = trace_list[self.trace_range[0] : self.trace_range[1]]

        # Handle both old format (5 elements from load) and new format (2 elements from run)
        if len(science) == 2:
            # New Spectrum-based format from science.run()
            heads, spectra_lists = science
            specs = []
            sigmas = []
            columns = []
            for spectra in spectra_lists:
                specs.append(np.ma.masked_invalid([s.spec for s in spectra]))
                sigmas.append(np.ma.masked_invalid([s.sig for s in spectra]))
                columns.append(np.array([[0, len(s.spec)] for s in spectra]))
        else:
            # Old array format from science.load()
            heads, specs, sigmas, _, columns = science

        nspec = specs[0].shape[0]

        # Filter out traces that extraction marked invalid
        valid = [t for t in trace_list if not t.invalid]
        if len(valid) == nspec:
            trace_list = valid
        wave = wavelengths_from_traces(trace_list)

        if wave is None:
            raise ValueError(
                "Continuum normalization requires wavelength data. "
                "Run wavecal or freq_comb steps first."
            )

        # Align all arrays to the smallest count (norm_flat may skip edge traces)
        nmin = min(nspec, len(blaze), len(wave) if wave is not None else nspec)
        if nspec > nmin:
            specs = [s[nspec - nmin :] for s in specs]
            sigmas = [s[nspec - nmin :] for s in sigmas]
            columns = [c[nspec - nmin :] for c in columns]
            nspec = nmin
        if wave is not None and len(wave) > nmin:
            wave = wave[len(wave) - nmin :]
        if len(blaze) > nmin:
            blaze = blaze[len(blaze) - nmin :]

        logger.info("Continuum normalization")
        conts = [None for _ in specs]
        for j, (spec, sigma) in enumerate(zip(specs, sigmas, strict=False)):
            logger.info("Splicing orders")
            specs[j], wave, blaze, sigmas[j] = splice_orders(
                spec,
                wave,
                blaze,
                sigma,
                scaling=True,
                plot=self.plot,
                plot_title=self.plot_title,
            )
            logger.info("Normalizing continuum")
            conts[j] = continuum_normalize(
                specs[j],
                wave,
                blaze,
                sigmas[j],
                plot=self.plot,
                plot_title=self.plot_title,
            )

        for head in heads:
            head["e_cont"] = (True, "CONT is a fitted continuum, orders spliced")

        self.save(heads, specs, sigmas, conts, columns)
        return heads, specs, sigmas, conts, columns

    def save(self, heads, specs, sigmas, conts, columns):
        """Save the results from the continuum normalization

        Parameters
        ----------
        heads : list(FITS header)
            FITS headers of each observation
        specs : list(array of shape (ntrace, ncol))
            extracted spectra
        sigmas : list(array of shape (ntrace, ncol))
            uncertainties of the extracted spectra
        conts : list(array of shape (ntrace, ncol))
            continuum for each spectrum
        columns : list(array of shape (ntrace, 2))
            column ranges for each spectra
        """
        value = {
            "heads": heads,
            "specs": specs,
            "sigmas": sigmas,
            "conts": conts,
            "columns": columns,
        }
        joblib.dump(value, self.savefile)
        logger.info("Created continuum normalization file: %s", self.savefile)

    def load(self, norm_flat, science):
        """Load the results from the continuum normalization

        Returns
        -------
        heads : list(FITS header)
            FITS headers of each observation
        specs : list(array of shape (ntrace, ncol))
            extracted spectra
        sigmas : list(array of shape (ntrace, ncol))
            uncertainties of the extracted spectra
        conts : list(array of shape (ntrace, ncol))
            continuum for each spectrum
        columns : list(array of shape (ntrace, 2))
            column ranges for each spectra
        """
        try:
            data = joblib.load(self.savefile)
            logger.info("Continuum normalization file: %s", self.savefile)
        except FileNotFoundError:
            # Use science files instead
            logger.warning(
                "No continuum normalized data found. Using unnormalized results instead."
            )
            heads, specs, sigmas, columns = science
            norm, blaze, *_ = norm_flat
            conts = [blaze for _ in specs]
            for head in heads:
                head["e_cont"] = (False, "CONT is the blaze, not a fitted continuum")
            data = {
                "heads": heads,
                "specs": specs,
                "sigmas": sigmas,
                "conts": conts,
                "columns": columns,
            }
        heads = data["heads"]
        specs = data["specs"]
        sigmas = data["sigmas"]
        conts = data["conts"]
        columns = data["columns"]
        return heads, specs, sigmas, conts, columns


class Finalize(Step):
    """Create the final output files"""

    def __init__(self, *args, **config):
        super().__init__(*args, **config)
        self._dependsOn += ["continuum", "trace", "config"]
        self.filename = config["filename"]

    def output_file(self, number, name):
        """str: output file name"""
        out = self.filename.format(
            instrument=self.instrument.name,
            night=self.night,
            channel=self.channel,
            number=number,
            input=name,
        )
        return join(self.output_dir, out)

    def save_config_to_header(self, head, config, prefix="PR"):
        for key, value in config.items():
            if isinstance(value, dict):
                head = self.save_config_to_header(
                    head, value, prefix=f"{prefix} {key.upper()}"
                )
            else:
                if key in ["plot", "$schema", "__skip_existing__"]:
                    # Skip values that are not relevant to the file product
                    continue
                if value is None:
                    value = "null"
                elif not np.isscalar(value):
                    value = str(value)
                head[f"HIERARCH {prefix} {key.upper()}"] = value
        return head

    def run(self, continuum, trace: list, config):
        """Create the final output files

        this is includes:
         - heliocentric corrections
         - creating one echelle file

        Parameters
        ----------
        continuum : tuple
            results from the continuum normalization
        trace : list[TraceData]
            Trace objects with wavelength polynomials
        config : dict
            Pipeline configuration
        """
        heads, specs, sigmas, conts, columns = continuum

        # Select same traces as science/continuum steps
        selected = self._select_traces(trace, "science")
        trace_list = [t for traces in selected.values() for t in traces]
        if self.trace_range is not None:
            trace_list = trace_list[self.trace_range[0] : self.trace_range[1]]
        valid = [t for t in trace_list if not t.invalid]
        nspec = specs[0].shape[0]
        if len(valid) == nspec:
            trace_list = valid
        wave = wavelengths_from_traces(trace_list)
        if wave is not None and len(wave) > nspec:
            wave = wave[len(wave) - nspec :]

        fnames = []
        # Combine science with wavecal and continuum
        for i, (head, spec, sigma, blaze, column) in enumerate(
            zip(heads, specs, sigmas, conts, columns, strict=False)
        ):
            head["e_erscle"] = ("absolute", "error scale")

            # Add heliocentric correction
            try:
                rv_corr, bjd = util.helcorr(
                    head["e_obslon"],
                    head["e_obslat"],
                    head["e_obsalt"],
                    head["e_ra"],
                    head["e_dec"],
                    head["e_jd"],
                )

                logger.debug("Heliocentric correction: %f km/s", rv_corr)
                logger.debug("Heliocentric Julian Date: %s", str(bjd))
            except KeyError:
                logger.warning("Could not calculate heliocentric correction")
                # logger.warning("Telescope is in space?")
                rv_corr = 0
                bjd = head["e_jd"]

            head["barycorr"] = rv_corr
            head["e_jd"] = bjd
            head["DATE"] = (
                datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
                "UTC timestamp of the reduction",
            )

            head = self.save_config_to_header(head, config)

            if self.plot:
                plt.figure()
                plt.plot(wave.T, (spec / blaze).T)
                if self.plot_title is not None:
                    plt.title(self.plot_title)
                util.show_or_save(f"finalize_{i}")

            fname = self.save(i, head, spec, sigma, blaze, wave, column)
            fnames.append(fname)
        return fnames

    def save(self, i, head, spec, sigma, cont, wave, columns):
        """Save one output spectrum to disk

        Parameters
        ----------
        i : int
            individual number of each file
        head : FITS header
            FITS header
        spec : array of shape (ntrace, ncol)
            final spectrum
        sigma : array of shape (ntrace, ncol)
            final uncertainties
        cont : array of shape (ntrace, ncol)
            final continuum scales
        wave : array of shape (ntrace, ncol)
            wavelength solution
        columns : array of shape (ntrace, 2)
            columns that carry signal

        Returns
        -------
        out_file : str
            name of the output file
        """
        original_name = os.path.splitext(head["e_input"])[0]
        out_file = self.output_file(i, original_name)

        ntrace = spec.shape[0]

        # Convert arrays to list[Spectrum], masking outside column range with NaN
        spectra_list = []
        for j in range(ntrace):
            spec_row = np.array(spec[j], dtype=np.float32)
            sig_row = np.array(sigma[j], dtype=np.float32)
            wave_row = np.array(wave[j], dtype=np.float64) if wave is not None else None
            cont_row = np.array(cont[j], dtype=np.float32) if cont is not None else None

            # Apply column mask as NaN
            if columns is not None:
                spec_row[: columns[j, 0]] = np.nan
                spec_row[columns[j, 1] :] = np.nan
                sig_row[: columns[j, 0]] = np.nan
                sig_row[columns[j, 1] :] = np.nan

            spectra_list.append(
                Spectrum(
                    m=j,
                    spec=spec_row,
                    sig=sig_row,
                    wave=wave_row,
                    cont=cont_row,
                )
            )

        spectra = Spectra(header=head, data=spectra_list)
        spectra.save(out_file, steps=["finalize"])
        logger.info("Final science file: %s", out_file)
        return out_file
