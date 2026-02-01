import numpy as np
import pytest

from pyreduce.extract import extract_normalize

pytestmark = [pytest.mark.instrument, pytest.mark.downloads]


@pytest.mark.slow
def test_normflat(flat, traces, settings, trace_range, scatter, instrument):
    flat_img, fhead = flat
    settings = settings["norm_flat"]

    if flat_img is None:
        pytest.skip(f"No flat exists for instrument {instrument}")

    # Apply trace_range
    traces_subset = traces[trace_range[0] : trace_range[1]]

    norm, _, blaze, _, extracted_column_range = extract_normalize(
        flat_img,
        traces_subset,
        scatter=scatter,
        gain=fhead["e_gain"],
        readnoise=fhead["e_readn"],
        dark=fhead["e_drk"],
        extraction_height=settings["extraction_height"],
        threshold=settings["threshold"],
        lambda_sf=settings["smooth_slitfunction"],
        lambda_sp=settings["smooth_spectrum"],
        swath_width=settings["swath_width"],
        plot=False,
    )

    assert isinstance(norm, np.ndarray)
    assert norm.ndim == flat_img.ndim
    assert norm.shape[0] == flat_img.shape[0]
    assert norm.shape[1] == flat_img.shape[1]
    assert norm.dtype == flat_img.dtype
    assert np.ma.min(norm) > 0
    assert not np.any(np.isnan(norm))

    assert isinstance(blaze, np.ndarray)
    assert blaze.ndim == 2
    assert blaze.shape[0] == len(extracted_column_range)
    assert blaze.shape[1] == flat_img.shape[1]
    assert np.issubdtype(blaze.dtype, np.floating)
    assert not np.any(np.isnan(blaze))

    for i, cr in enumerate(extracted_column_range):
        assert np.all(blaze[i, : cr[0]].mask)
        assert np.all(blaze[i, cr[1] :].mask)
