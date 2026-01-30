# Plan: Use charslit for Extraction with Arbitrary Curvature

## Goal
Replace PyReduce's CFFI extraction with `charslit` package, which supports arbitrary polynomial degrees (1-5) and slitdeltas.

## Status

### ✅ Done
- [x] Branch: `charslit`
- [x] Added `charslit>=0.1.0` to dependencies with `[tool.uv.sources]` pointing to `../CharSlit.git`
- [x] Created `pyreduce/curvature_model.py` with SlitCurvature dataclass
- [x] Updated `pyreduce/slit_curve.py` for degree 1-5, returns SlitCurvature
- [x] Updated `pyreduce/reduce.py` save/load to use SlitCurvature
- [x] Updated `pyreduce/extract.py` to use `charslit.slitdec()` with curvature array
- [x] Updated `pyreduce/instruments/defaults/schema.json` curve_degree max: 5
- [x] Updated tests for new curvature format
- [x] **Fixed CharSlit wrapper** (`slitdec_wrapper.cpp`): stride 3 → 6 coefficients
- [x] Unit tests pass
- [x] **Backend switching**: `PYREDUCE_USE_CHARSLIT=1` enables charslit, default is CFFI
- [x] **Outlier rejection**: Added `kappa` parameter to charslit (RMS-based, matching CFFI)
- [x] Pass `extraction_reject` setting as `kappa` to charslit

### ⚠️ Partially Done
- [ ] slitdeltas computation in slit_curve.py (structure exists, not computed yet)

### ❌ Not Started
- [ ] Add `compute_slitdeltas` option to schema.json
- [ ] Remove/cleanup old CFFI code in `pyreduce/cwrappers.py`
- [ ] Verification with UVES example
- [ ] Comparison of extraction results with old method

### 🐛 Known Issues
- CharSlit fails with very sparse data (many zero columns, e.g., pure LFC frames)
  - Workaround: combine LFC with flat-field frames for continuum (done in ANDES example)

## charslit Interface

```python
import charslit

result = charslit.slitdec(
    im,           # (nrows, ncols) - 2D image
    pix_unc,      # (nrows, ncols) - pixel uncertainties
    mask,         # (nrows, ncols) - uint8, 0=bad, 1=good
    ycen,         # (ncols,) - trace center (fractional row)
    slitcurve,    # (ncols, n_coeffs) - polynomial coeffs, n_coeffs <= 6
    slitdeltas,   # (nrows,) - per-row residual offsets (interpolated internally)
    osample=6,
    lambda_sP=0.0,
    lambda_sL=0.1,
    maxiter=20,
    kappa=10.0,   # outlier rejection threshold in sigma (0 to disable)
)
# Returns: {spectrum, slitfunction, model, uncertainty, mask, return_code, info}
```

## Data Structure

```python
@dataclass
class SlitCurvature:
    coeffs: np.ndarray       # (ntrace, ncol, degree+1) - c0 always 0
    slitdeltas: np.ndarray   # (ntrace, nrow) or None
    degree: int              # 1-5
```

## Commits Made

1. `afe7590` - Use charslit for extraction with polynomial curvature up to degree 5
2. `ad566c7` - ANDES example: combine LFC with flats for continuum
3. `06f4823` - Add charslit dependency and update ANDES_RIZ curve_degree to 5
4. `8b042df` - Add PYREDUCE_USE_CHARSLIT env var to switch extraction backends
5. `912000d` - Pass extraction_reject as kappa to charslit for outlier rejection

## Files Modified

| File | Status | Changes |
|------|--------|---------|
| `pyreduce/curvature_model.py` | ✅ NEW | SlitCurvature dataclass, save/load functions |
| `pyreduce/slit_curve.py` | ✅ | Degree 1-5 support, returns SlitCurvature |
| `pyreduce/reduce.py` | ✅ | Updated save/load, curvature→coeffs extraction |
| `pyreduce/extract.py` | ✅ | Uses `charslit.slitdec()`, backend switching, kappa |
| `pyreduce/instruments/defaults/schema.json` | ✅ | curve_degree max: 5 |
| `pyreduce/cwrappers.py` | ❌ | Old CFFI code still present (unused when charslit active) |
| `test/*.py` | ✅ | Updated for new curvature format |
| `examples/andes_riz.py` | ✅ | Combines LFC+flats for extraction |

## Next Steps

1. **Test with UVES example** to verify extraction works on non-ANDES data
2. **Implement slitdeltas** computation in slit_curve.py (residuals after polynomial fit)
3. **Clean up cwrappers.py** - remove unused CFFI extraction code
4. **Compare results** with old extraction method on same data
