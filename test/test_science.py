import numpy as np
import pytest

from pyreduce.combine_frames import combine_calibrate
from pyreduce.extract import extract

pytestmark = [pytest.mark.instrument, pytest.mark.downloads]


@pytest.mark.slow
def test_science(
    files,
    instr,
    instrument,
    channel,
    mask,
    bias,
    normflat,
    traces,
    settings,
    trace_range,
):
    if len(files["science"]) == 0:
        pytest.skip(f"No science files found for instrument {instrument}")

    flat, blaze = normflat
    bias, bhead = bias
    settings = settings["science"]

    # Apply trace_range
    traces_subset = traces[trace_range[0] : trace_range[1]]

    f = files["science"][0]

    im, head = combine_calibrate(
        [f],
        instr,
        channel,
        mask=mask,
        bias=bias,
        bhead=bhead,
        norm=flat,
        bias_scaling=settings["bias_scaling"],
    )

    # Optimally extract science spectrum
    spectra = extract(
        im,
        traces_subset,
        gain=head["e_gain"],
        readnoise=head["e_readn"],
        dark=head["e_drk"],
        extraction_type=settings["extraction_method"],
        extraction_height=settings["extraction_height"],
        lambda_sf=settings["smooth_slitfunction"],
        lambda_sp=settings["smooth_spectrum"],
        osample=settings["oversampling"],
        swath_width=settings["swath_width"],
        plot=False,
    )

    # Convert Spectrum objects to masked arrays for assertions
    spec = np.ma.array([s.spec for s in spectra])
    sigma = np.ma.array([s.sig for s in spectra])

    # Mask NaN values
    spec = np.ma.masked_invalid(spec)
    sigma = np.ma.masked_invalid(sigma)

    assert isinstance(spec, np.ma.masked_array)
    assert spec.ndim == 2
    assert spec.shape[0] == trace_range[1] - trace_range[0]
    assert spec.shape[1] == im.shape[1]
    assert np.issubdtype(spec.dtype, np.floating)
    assert not np.any(np.isnan(np.ma.filled(spec, 0)))
    assert not np.all(np.all(spec.mask, axis=0))

    assert isinstance(sigma, np.ma.masked_array)
    assert sigma.ndim == 2
    assert sigma.shape[0] == trace_range[1] - trace_range[0]
    assert sigma.shape[1] == im.shape[1]
    assert np.issubdtype(sigma.dtype, np.floating)
    assert not np.any(np.isnan(np.ma.filled(sigma, 0)))
    assert not np.all(np.all(sigma.mask, axis=0))
