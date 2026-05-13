"""Unit tests for MOSAIC instrument-specific logic."""

import numpy as np
import pytest
from astropy.io import fits

from pyreduce.instruments.MOSAIC import MOSAIC
from pyreduce.trace_model import Trace as TraceData


def _make_fits_with_fibre_table(path, n_fibers, n_dead, bundle_size):
    """Write a minimal FITS with a FIBRE_TABLE extension matching the MOSAIC schema."""
    indices = np.arange(n_fibers, dtype=np.int32)
    bundles = (indices // bundle_size).astype(np.int32)
    ifbs = (indices % bundle_size).astype(np.int32)
    health = np.ones(n_fibers, dtype=np.int32)
    # Kill a few fibers
    rng = np.random.default_rng(0)
    dead_idx = rng.choice(n_fibers, size=n_dead, replace=False)
    health[dead_idx] = 0
    fibreids = np.array([f"F{i}-G{n_fibers - 1 - i}" for i in indices])

    cols = fits.ColDefs(
        [
            fits.Column(name="INDEX", format="J", array=indices),
            fits.Column(name="FIBREID", format="10A", array=fibreids),
            fits.Column(name="HEALTH", format="J", array=health),
            fits.Column(name="BUNDLE", format="J", array=bundles),
            fits.Column(name="IFB", format="J", array=ifbs),
        ]
    )
    tbl = fits.BinTableHDU.from_columns(cols, name="FIBRE_TABLE")
    primary = fits.PrimaryHDU(data=np.zeros((10, 10), dtype=np.float32))
    fits.HDUList([primary, tbl]).writeto(path, overwrite=True)


@pytest.mark.unit
def test_assign_bundles_top_half_pairs_in_order(tmp_path):
    """VIS3 (slit_half=top) takes the first N live FIBRE_TABLE rows and pairs by y."""
    path = tmp_path / "flat.fits"
    _make_fits_with_fibre_table(path, n_fibers=140, n_dead=4, bundle_size=7)

    inst = MOSAIC()

    # Simulate N detected traces in some scrambled order, but with known y.
    # The hook should pair them with the FIRST N live rows in y order.
    n_detected = 12
    rng = np.random.default_rng(42)
    ys = np.sort(rng.uniform(50, 950, size=n_detected))
    rng.shuffle(ys)  # detection order is not y-sorted
    traces = [
        TraceData(m=None, pos=np.array([0.0, 0.0, y]), column_range=(0, 1000))
        for y in ys
    ]

    centers = inst.assign_bundles(traces, [str(path)], header=None, channel="VIS3")
    assert isinstance(centers, dict)

    # All traces got assigned, none have m
    assert all(t.bundle is not None and t.fiber_idx is not None for t in traces)
    assert all(t.m is None for t in traces)

    # Pairing should follow y-order against the first n_detected live rows
    with fits.open(path) as h:
        live = h["FIBRE_TABLE"].data
        live = live[live["HEALTH"] == 1][:n_detected]

    traces_sorted = sorted(traces, key=lambda t: t.y_at_x(500))
    for tr, row in zip(traces_sorted, live, strict=True):
        assert tr.bundle == int(row["BUNDLE"])
        assert tr.fiber_idx == int(row["IFB"]) + 1  # 1-based

    # bundle_centers covers every unique assigned bundle
    assert set(centers) == {t.bundle for t in traces}


@pytest.mark.unit
def test_assign_bundles_bottom_half_takes_last_rows(tmp_path):
    """VIS1 (slit_half=bottom) takes the LAST N live rows."""
    path = tmp_path / "flat.fits"
    _make_fits_with_fibre_table(path, n_fibers=140, n_dead=2, bundle_size=7)

    inst = MOSAIC()
    n_detected = 10
    ys = np.linspace(50, 950, n_detected)
    traces = [
        TraceData(m=None, pos=np.array([0.0, 0.0, y]), column_range=(0, 1000))
        for y in ys
    ]

    centers = inst.assign_bundles(traces, [str(path)], header=None, channel="VIS1")
    assert isinstance(centers, dict)

    with fits.open(path) as h:
        live = h["FIBRE_TABLE"].data
        live = live[live["HEALTH"] == 1][-n_detected:]
    for tr, row in zip(traces, live, strict=True):
        assert tr.bundle == int(row["BUNDLE"])


@pytest.mark.unit
def test_assign_bundles_count_mismatch_raises(tmp_path):
    """Bail loudly when detected count != live FIBRE_TABLE count for the slit half."""
    path = tmp_path / "flat.fits"
    _make_fits_with_fibre_table(path, n_fibers=20, n_dead=0, bundle_size=7)

    inst = MOSAIC()
    # Only 5 detected, but slit_half=top would still try to take the first 5.
    # To force a mismatch use the unbundled NIR fallback (no slit_half), which
    # would take all 20 live rows.
    traces = [
        TraceData(m=None, pos=np.array([0.0, 0.0, float(i)]), column_range=(0, 1000))
        for i in range(5)
    ]
    with pytest.raises(ValueError, match="off-by-one"):
        inst.assign_bundles(traces, [str(path)], header=None, channel="NIR")


@pytest.mark.unit
def test_assign_bundles_no_fibre_table_returns_none(tmp_path):
    """Files without a FIBRE_TABLE extension are a no-op (fall back to nearest-y)."""
    path = tmp_path / "flat.fits"
    fits.PrimaryHDU(data=np.zeros((10, 10), dtype=np.float32)).writeto(path)

    inst = MOSAIC()
    traces = [TraceData(m=None, pos=np.array([0.0, 0.0, 1.0]), column_range=(0, 1000))]
    assert inst.assign_bundles(traces, [str(path)], header=None, channel="VIS3") is None
    # Nothing got assigned
    assert traces[0].bundle is None


@pytest.mark.unit
def test_assign_bundles_centers_use_middle_fiber_y(tmp_path):
    """bundle_centers should be the y of the geometric-middle fiber (IFB=3)
    for a 7-fiber bundle, not the mean of all detected y (which is biased
    when the bundle's internal spacing is non-uniform).
    """
    path = tmp_path / "flat.fits"
    _make_fits_with_fibre_table(path, n_fibers=7, n_dead=0, bundle_size=7)

    inst = MOSAIC()
    # Non-uniform fiber spacing: tight, tight, GAP, tight, tight, tight
    # (mimics the bundle's two-slitlet internal layout we see in MOSAIC).
    ys = [100.0, 107.5, 115.0, 130.0, 137.5, 145.0, 152.5]
    traces = [
        TraceData(m=None, pos=np.array([0.0, 0.0, y]), column_range=(0, 1000))
        for y in ys
    ]
    centers = inst.assign_bundles(traces, [str(path)], header=None, channel="NIR")
    # Geometric middle fiber (IFB=3, 4th from bottom) sits at y=130, not the
    # arithmetic mean of all 7 (which would be ~126.8).
    assert centers[0] == pytest.approx(130.0)
