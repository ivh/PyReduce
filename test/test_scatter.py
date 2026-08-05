import numpy as np
import pytest

from pyreduce.estimate_background_scatter import (
    ScatterModel,
    as_scatter_coeff,
    estimate_background_scatter,
)
from pyreduce.trace_model import Trace

pytestmark = [pytest.mark.instrument, pytest.mark.downloads]


@pytest.mark.slow
def test_scatter(flat, traces, settings):
    # The background scatter step in reduce possibly uses a
    # different set of files for the image
    # However it should still be able to create a scatter fit
    # from the flat image as is done here
    img, _ = flat
    settings = settings["scatter"]
    settings["sigma_cutoff"] = settings["scatter_cutoff"]
    del settings["scatter_cutoff"]
    del settings["bias_scaling"]
    del settings["norm_scaling"]

    if img is None:
        pytest.skip("Need flat")

    scatter = estimate_background_scatter(img, traces, **settings)

    degree = settings["scatter_degree"]
    if np.isscalar(degree):
        degree = [degree, degree]

    assert isinstance(scatter, np.ndarray)
    assert scatter.ndim == 2
    assert scatter.shape[0] == degree[0] + 1
    assert scatter.shape[1] == degree[1] + 1


def test_simple():
    img = np.full((100, 100), 10.0)
    traces = [
        Trace(m=0, group=0, pos=np.array([25.0, 0.0]), column_range=(0, 100)),
        Trace(m=1, group=0, pos=np.array([50.0, 0.0]), column_range=(0, 100)),
        Trace(m=2, group=0, pos=np.array([75.0, 0.0]), column_range=(0, 100)),
    ]

    scatter = estimate_background_scatter(img, traces, scatter_degree=0, plot=False)

    assert isinstance(scatter, np.ndarray)
    assert scatter.ndim == 2
    assert scatter.shape[0] == 1
    assert scatter.shape[1] == 1

    assert np.allclose(scatter[0, 0], 10.0)


def test_scatter_degree():
    img = np.full((100, 100), 10.0)
    traces = [
        Trace(m=0, group=0, pos=np.array([25.0, 0.0]), column_range=(0, 100)),
        Trace(m=1, group=0, pos=np.array([75.0, 0.0]), column_range=(0, 100)),
    ]

    estimate_background_scatter(img, traces, scatter_degree=0)

    with pytest.raises(ValueError):
        estimate_background_scatter(img, traces, scatter_degree=-1)

    estimate_background_scatter(img, traces, scatter_degree=(2, 2))

    with pytest.raises(AssertionError):
        estimate_background_scatter(img, traces, scatter_degree=(1,))

    with pytest.raises(AssertionError):
        estimate_background_scatter(img, traces, scatter_degree=(3, 2, 1))

    with pytest.raises(ValueError):
        estimate_background_scatter(img, traces, scatter_degree=(2, -1))


def _traces():
    return [
        Trace(m=0, group=0, pos=np.array([25.0, 0.0]), column_range=(0, 100)),
        Trace(m=1, group=0, pos=np.array([50.0, 0.0]), column_range=(0, 100)),
        Trace(m=2, group=0, pos=np.array([75.0, 0.0]), column_range=(0, 100)),
    ]


@pytest.mark.unit
def test_refit_uses_the_frame_it_is_given():
    """A model measured on a bright frame must not be reused on a faint one."""
    traces = _traces()
    params = {"scatter_degree": 0, "extraction_height": 0.1, "border_width": 0}

    flat = np.full((100, 100), 4000.0)  # e.g. 10 summed lamp exposures
    science = np.full((100, 100), 50.0)  # a single short exposure

    coeff = estimate_background_scatter(flat, traces, **params)
    model = ScatterModel(coeff=coeff, params=params, reference="flat")

    assert np.allclose(model.coeff[0, 0], 4000.0)
    assert np.allclose(model.refit(science, traces)[0, 0], 50.0)


@pytest.mark.unit
def test_as_scatter_coeff_refits_a_model_and_passes_arrays_through():
    traces = _traces()
    params = {"scatter_degree": 0, "extraction_height": 0.1, "border_width": 0}
    flat = np.full((100, 100), 4000.0)
    science = np.full((100, 100), 50.0)

    model = ScatterModel(
        coeff=estimate_background_scatter(flat, traces, **params),
        params=params,
        reference="flat",
    )
    # a model is re-estimated on the frame being corrected ...
    assert np.allclose(as_scatter_coeff(model, science, traces)[0, 0], 50.0)
    # ... a bare coefficient array carries no scale, so it is used unchanged ...
    assert np.allclose(as_scatter_coeff(model.coeff, science, traces)[0, 0], 4000.0)
    # ... and None stays None.
    assert as_scatter_coeff(None, science, traces) is None


@pytest.mark.unit
def test_masks_traces_whose_valid_columns_are_not_contiguous():
    """A trace bowing out of the frame mid-detector is in bounds at both edges.

    The aperture then fits for two disjoint runs of columns, which used to be
    indexed as one span from the first to the last valid column and raised an
    IndexError.
    """
    nrow, ncol = 100, 400
    img = np.full((nrow, ncol), 10.0)
    # parabola peaking above the top edge in the middle, inside it at both ends
    x = np.array([0, ncol // 2, ncol - 1], dtype=float)
    y = np.array([50.0, 130.0, 50.0])
    pos = np.polyfit(x, y, 2)
    traces = [
        Trace(m=0, group=0, pos=pos, column_range=(0, ncol)),
        Trace(m=1, group=0, pos=np.array([20.0, 0.0]), column_range=(0, ncol)),
    ]

    coeff = estimate_background_scatter(
        img, traces, extraction_height=20, scatter_degree=0, border_width=0
    )
    assert np.allclose(coeff[0, 0], 10.0)
