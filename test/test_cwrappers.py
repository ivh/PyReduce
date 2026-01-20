import numpy as np
import pytest

from pyreduce.cwrappers import slitfunc, slitfunc_curved

pytestmark = pytest.mark.unit


@pytest.mark.skip(reason="Not a unit test - requires instrument data")
@pytest.mark.instrument
@pytest.mark.downloads
def test_1d_vs_2d_extraction(flat, orders):
    """Compare 1D (vertical) and 2D (curved with p1=p2=0) extraction on real UVES data."""
    import os
    from pathlib import Path

    from pyreduce.util import make_index

    flat_img, flat_head = flat
    if flat_img is None:
        pytest.skip("No flat data available")

    # Setup debug output directory
    reduce_data = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))
    debug_dir = Path(reduce_data) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    traces, column_range = orders

    # Pick a middle order and a swath in the middle of it
    nord = traces.shape[0]
    order_idx = nord // 2
    trace = traces[order_idx]
    cr = column_range[order_idx]

    # Swath parameters
    swath_width = 300
    extraction_height = 50
    xlow = (cr[0] + cr[1]) // 2  # middle of order
    xhigh = xlow + swath_width
    if xhigh > cr[1]:
        xhigh = cr[1]
        xlow = xhigh - swath_width

    # Get ycen for this swath
    x = np.arange(xlow, xhigh)
    ycen_full = np.polyval(trace, x)
    ycen_int = np.floor(ycen_full).astype(int)
    ycen = ycen_full - ycen_int  # fractional part for slitfunc

    # Cut out swath (zero=xlow to get relative x indices 0..swath_width)
    ylow = yhigh = extraction_height
    index = make_index(ycen_int - ylow, ycen_int + yhigh, xlow, xhigh, zero=xlow)
    swath_img = flat_img[index].astype(float)

    # Common parameters
    lambda_sp = 0
    lambda_sf = 0.1
    osample = 1

    # 1D extraction
    sp1, sl1, model1, unc1, mask1 = slitfunc(
        swath_img, ycen, lambda_sp, lambda_sf, osample
    )

    # 2D extraction with zero curvature
    yrange = (ylow, yhigh)
    sp2, sl2, model2, unc2, mask2, info = slitfunc_curved(
        swath_img,
        ycen,
        p1=0,
        p2=0,
        lambda_sp=lambda_sp,
        lambda_sf=lambda_sf,
        osample=osample,
        yrange=yrange,
    )

    # Compare results
    assert sp1.shape == sp2.shape, f"Spectrum shapes differ: {sp1.shape} vs {sp2.shape}"
    assert sl1.shape == sl2.shape, f"Slitfunc shapes differ: {sl1.shape} vs {sl2.shape}"

    # Report differences
    sp_rel_diff = np.abs(sp1 - sp2) / np.maximum(sp1, 1)
    sl_rel_diff = np.abs(sl1 - sl2) / np.maximum(sl1, 1e-10)
    print(
        f"\nSpectrum: max rel diff = {sp_rel_diff.max():.4f}, mean = {sp_rel_diff.mean():.4f}"
    )
    print(
        f"Slitfunc: max rel diff = {sl_rel_diff.max():.4f}, mean = {sl_rel_diff.mean():.4f}"
    )

    # Save results to debug directory
    outfile = debug_dir / "1d_vs_2d_extraction.npz"
    np.savez(
        outfile,
        swath_img=swath_img,
        ycen=ycen,
        sp_1d=sp1,
        sl_1d=sl1,
        model_1d=model1,
        unc_1d=unc1,
        mask_1d=mask1,
        sp_2d=sp2,
        sl_2d=sl2,
        model_2d=model2,
        unc_2d=unc2,
        mask_2d=mask2,
        info_2d=info,
    )
    print(f"Saved results to {outfile}")

    # Allow some tolerance since algorithms differ slightly
    # Note: with larger swaths, the slit functions show a 1-element offset between 1D and 2D
    np.testing.assert_allclose(
        sp1, sp2, rtol=0.05, err_msg="Spectra differ significantly"
    )


def test_slitfunc_vert():
    img = np.full((100, 100), 1, dtype=float)
    ycen = np.full(100, 0, dtype=float)
    lambda_sp = 0
    lambda_sf = 0.1
    osample = 1

    # Run it once the way it is supposed to
    slitfunc(img, ycen, lambda_sp, lambda_sf, osample)

    # Then try different incompatible inputs, which have to be caught before going to the C code
    with pytest.raises(AssertionError):
        slitfunc(None, ycen, lambda_sp, lambda_sf, osample)
    with pytest.raises(ValueError):
        slitfunc("bla", ycen, lambda_sp, lambda_sf, osample)

    with pytest.raises(AssertionError):
        slitfunc(img, None, lambda_sp, lambda_sf, osample)
    with pytest.raises(ValueError):
        slitfunc(img, "blub", lambda_sp, lambda_sf, osample)

    with pytest.raises(TypeError):
        slitfunc(img, ycen, None, lambda_sf, osample)
    with pytest.raises(ValueError):
        slitfunc(img, ycen, "bla", lambda_sf, osample)
    with pytest.raises(TypeError):
        slitfunc(img, ycen, lambda_sp, None, osample)
    with pytest.raises(ValueError):
        slitfunc(img, ycen, lambda_sp, "bla", osample)
    with pytest.raises(TypeError):
        slitfunc(img, ycen, lambda_sp, lambda_sf, None)
    with pytest.raises(ValueError):
        slitfunc(img, ycen, lambda_sp, lambda_sf, "bla")

    # Then try different sizes for img and ycen
    with pytest.raises(AssertionError):
        ycen_bad = np.full(50, 1, dtype=float)
        slitfunc(img, ycen_bad, lambda_sp, lambda_sf, osample)

    with pytest.raises(AssertionError):
        slitfunc(img, ycen, lambda_sp, lambda_sf, osample=0)
    with pytest.raises(AssertionError):
        slitfunc(img, ycen, lambda_sp, -1, osample)
    with pytest.raises(AssertionError):
        slitfunc(img, ycen, -1, lambda_sf, osample)


def test_slitfunc_curved():
    img = np.full((100, 100), 1)
    ycen = np.full(100, 50)
    p1 = np.full(100, 0)
    p2 = np.full(100, 0)
    lambda_sp = 0
    lambda_sf = 0.1
    osample = 1
    yrange = (49, 50)

    # Run it once the way it is supposed to
    slitfunc_curved(img, ycen, p1, p2, lambda_sp, lambda_sf, osample, yrange)
    slitfunc_curved(img, ycen, 1, 0.01, lambda_sp, lambda_sf, osample, yrange)

    # Then try different incompatible inputs, which have to be caught before going to the C code
    with pytest.raises(AssertionError):
        slitfunc_curved(None, ycen, p1, p2, lambda_sp, lambda_sf, osample, yrange)
    with pytest.raises(ValueError):
        slitfunc_curved("bla", ycen, p1, p2, lambda_sp, lambda_sf, osample, yrange)

    with pytest.raises(AssertionError):
        slitfunc_curved(img, None, p1, p2, lambda_sp, lambda_sf, osample, yrange)
    with pytest.raises(ValueError):
        slitfunc_curved(img, "blub", p1, p2, lambda_sp, lambda_sf, osample, yrange)

    with pytest.raises(AssertionError):
        slitfunc_curved(img, ycen, p1, None, lambda_sp, lambda_sf, osample, yrange)
    with pytest.raises(ValueError):
        slitfunc_curved(img, ycen, p1, "boo", lambda_sp, lambda_sf, osample, yrange)

    with pytest.raises(TypeError):
        slitfunc_curved(img, ycen, p1, p2, None, lambda_sf, osample, yrange)
    with pytest.raises(ValueError):
        slitfunc_curved(img, ycen, p1, p2, "bla", lambda_sf, osample, yrange)
    with pytest.raises(TypeError):
        slitfunc_curved(img, ycen, p1, p2, lambda_sp, None, osample, yrange)
    with pytest.raises(ValueError):
        slitfunc_curved(img, ycen, p1, p2, lambda_sp, "bla", osample, yrange)
    with pytest.raises(TypeError):
        slitfunc_curved(img, ycen, p1, p2, lambda_sp, lambda_sf, None, yrange)
    with pytest.raises(ValueError):
        slitfunc_curved(img, ycen, p1, p2, lambda_sp, lambda_sf, "bla", yrange)

    # Then try different sizes for img and ycen
    with pytest.raises(AssertionError):
        ycen_bad = np.full(50, 0, dtype=float)
        slitfunc_curved(img, ycen_bad, p1, p2, lambda_sp, lambda_sf, osample, yrange)

    with pytest.raises(AssertionError):
        p1_bad = np.full(50, 0, dtype=float)
        slitfunc_curved(img, ycen, p1_bad, p2, lambda_sp, lambda_sf, osample, yrange)

    with pytest.raises(AssertionError):
        p2_bad = np.full(50, 0, dtype=float)
        slitfunc_curved(img, ycen, p1, p2_bad, lambda_sp, lambda_sf, osample, yrange)

    with pytest.raises(AssertionError):
        slitfunc_curved(img, ycen, p1, p2, lambda_sp, lambda_sf, 0, yrange)
    with pytest.raises(AssertionError):
        slitfunc_curved(img, ycen, p1, p2, lambda_sp, -1, osample, yrange)
    with pytest.raises(AssertionError):
        slitfunc_curved(img, ycen, p1, p2, -1, lambda_sf, osample, yrange)
