"""
with our powers combined we increase snr
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from spectres import spectres
from tqdm import tqdm

from .. import echelle, util
from ..spectra import Spectra, Spectrum

logger = logging.getLogger(__name__)


def combine(files, output, plot=None):
    """Combine multiple spectra into a single higher-SNR spectrum.

    Parameters
    ----------
    files : list[str]
        Input spectrum files (FITS format).
    output : str
        Output file path.
    plot : int, optional
        If provided, plot this trace index for diagnostic visualization.
    """
    # Create a wavelength grid based on the first file
    first_spec = _load_spectrum(files[0])
    ntrace = first_spec["ntrace"]
    ncol = first_spec["ncol"]

    # Prepare storage arrays
    nfiles = len(files)
    waves = np.zeros((nfiles, ntrace, ncol))
    specs = np.zeros((nfiles, ntrace, ncol))
    sigms = np.zeros((nfiles, ntrace, ncol))
    conts = np.zeros((nfiles, ntrace, ncol))

    mask = np.full(nfiles, True)

    # Load all data and resample onto shared grid
    for k, file in tqdm(enumerate(files), desc="File", total=nfiles):
        try:
            data = _load_spectrum(file)
            specs[k] = np.nan_to_num(data["spec"], nan=0)
            waves[k] = np.nan_to_num(data["wave"], nan=0)
            sigms[k] = np.nan_to_num(data["sig"], nan=1)
            conts[k] = np.nan_to_num(data["cont"], nan=0)
        except ValueError as ex:
            logger.warning("Error in loading file %s. %s", file, ex)
            mask[k] = False

    waves = waves[mask]
    specs = specs[mask]
    sigms = sigms[mask]
    conts = conts[mask]

    wnew = np.copy(waves[0])

    for k in tqdm(range(specs.shape[0]), desc="File"):
        for i in tqdm(range(ntrace), desc="Trace", leave=False):
            conts[k, i], _ = spectres(wnew[i], waves[k, i], conts[k, i], sigms[k, i])
            specs[k, i], sigms[k, i] = spectres(
                wnew[i], waves[k, i], specs[k, i], sigms[k, i]
            )

    # These are just for plotting
    if plot:
        sold = np.copy(specs[:, plot])
        cold = np.copy(conts[:, plot])

    # Median and MAD
    arr = specs / conts
    mean = np.nanmedian(arr, axis=0)
    std = np.nanmedian(np.abs(arr - mean), axis=[0, 2])[:, None]
    vmin, vmax = mean - 5 * std, mean + 5 * std

    # Disregard all values outside of 5 * MAD
    where = (arr < vmin) | (arr > vmax)
    specs[where] = 0
    conts[where] = 1
    sigms[where] = np.nan
    weights = 1 / sigms
    weights[np.isposinf(weights)] = np.sqrt(2)
    weights[where] = 0

    w2 = np.sum(weights, axis=0) == 0
    weights[:, w2] = 1

    # Take weighted average
    snew = np.average(specs, weights=weights, axis=0)
    cnew = np.average(conts, weights=weights, axis=0)
    snew = np.nan_to_num(snew, copy=False)

    # Uncertainty from scatter
    unew = np.sqrt(np.nansum(weights * (arr - snew) ** 2, axis=0) / len(files))
    unew[unew == 0] = np.nansum(sigms, axis=0)[unew == 0]

    snew /= cnew
    cnew = np.ones_like(snew)

    if plot:
        plt.figure()
        for i in range(sold.shape[0]):
            plt.plot(wnew[plot], sold[i] / cold[i])
        plt.plot(wnew[plot], snew[plot], "--")
        plt.fill_between(wnew[plot], vmin[plot], vmax[plot], alpha=0.5)
        util.show_or_save("combine_spectra")

    # Save using new Spectra format
    _save_combined(output, first_spec, snew, unew, wnew, cnew)
    logger.info("Created combined file: %s", output)


def _load_spectrum(file):
    """Load spectrum from FITS file, supporting both new and legacy formats.

    Returns
    -------
    dict
        Dictionary with keys: spec, sig, wave, cont, ntrace, ncol, header,
        m_values, fiber_values
    """
    try:
        # Try new Spectra format first
        spectra = Spectra.read(file, continuum_normalization=False)
        ntrace = spectra.ntrace
        ncol = spectra.ncol

        spec = np.array([s.spec for s in spectra.data])
        sig = np.array([s.sig for s in spectra.data])
        wave = np.array(
            [s.wave if s.wave is not None else np.zeros(ncol) for s in spectra.data]
        )
        cont = np.array(
            [s.cont if s.cont is not None else np.ones(ncol) for s in spectra.data]
        )
        m_values = [s.m for s in spectra.data]
        fiber_values = [s.fiber for s in spectra.data]

        return {
            "spec": spec,
            "sig": sig,
            "wave": wave,
            "cont": cont,
            "ntrace": ntrace,
            "ncol": ncol,
            "header": spectra.header,
            "m_values": m_values,
            "fiber_values": fiber_values,
        }
    except Exception:
        # Fall back to legacy Echelle format
        e = echelle.read(file, continuum_normalization=False)
        ntrace, ncol = e.spec.shape

        return {
            "spec": np.ma.filled(e.spec, 0),
            "sig": np.ma.filled(e.sig, 1),
            "wave": np.ma.getdata(e.wave),
            "cont": np.ma.filled(e.cont, 0),
            "ntrace": ntrace,
            "ncol": ncol,
            "header": e.header,
            "m_values": list(range(ntrace)),
            "fiber_values": [0] * ntrace,
        }


def _save_combined(output, first_spec, spec, sig, wave, cont):
    """Save combined spectrum in new Spectra format."""
    ntrace = first_spec["ntrace"]
    header = first_spec["header"].copy()
    header["barycorr"] = 0.0

    # Build Spectrum objects
    spectra_list = []
    for i in range(ntrace):
        # Use NaN where data is zero (no signal)
        spec_i = spec[i].copy()
        sig_i = sig[i].copy()
        spec_i[spec_i == 0] = np.nan
        sig_i[sig_i == 0] = np.nan

        spectra_list.append(
            Spectrum(
                m=first_spec["m_values"][i],
                fiber=first_spec["fiber_values"][i],
                spec=spec_i,
                sig=sig_i,
                wave=wave[i],
                cont=cont[i],
            )
        )

    spectra = Spectra(header=header, data=spectra_list)
    spectra.save(output, steps=["combine"])
