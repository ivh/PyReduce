# Trace/Spectrum Refactoring Status

## Summary

Refactoring complete. `Trace` and `Spectrum` objects are the standard interface.

## Completed

- `extract()` takes `list[Trace]`, returns `list[Spectrum]`
- `Trace.run()` and `Trace.load()` return `list[Trace]` directly
- Pipeline stores `list[Trace]` in `_data["trace"]`
- All step `run()` methods use `trace: list[Trace]` parameter
- Eliminated `.wavecal.npz` - data now in traces.fits + .linelist.npz
- Added `Trace.wlen(x)` method to evaluate wavelength polynomial
- Removed `echelle.py` - `Spectra` class handles all spectrum I/O
- `Finalize` step uses `Spectra` format (not legacy echelle)
- Added `docs/output_formats.md` documenting v1 vs v2 FITS formats
- All unit tests pass
- Removed `traces_to_arrays()` - all modules use `list[Trace]` directly
- Curvature step updates traces in-place (no separate curvature parameter needed)

## File formats

| File | Format |
|------|--------|
| `*.traces.fits` | FITS binary table, one row per trace |
| `*.science.fits` | Spectra v2 format (E_FMTVER=2) |
| `*.final.fits` | Spectra v2 format (E_FMTVER=2) |
| `*.linelist.npz` | Wavelength calibration line list |

Legacy formats (NPZ traces, echelle v1) are read but not written.

## Naming conventions

| Name | Type | Meaning |
|------|------|---------|
| `trace` | `list[Trace]` | Step run() parameter |
| `Trace.wave` | `np.ndarray` | Wavelength polynomial coefficients |
| `Trace.wlen(x)` | method | Evaluate wavelength at column positions |
| `wlen` | `np.ndarray` | Evaluated wavelength array |
| `wavecal_spec` | `np.ndarray` | Extracted calibration spectrum |

## Future work

Remaining array-based internal code:
- `wavelength_calibration` (uses `nord`/`iord` internally)
- `continuum_normalization` (uses `nord` internally)

These work with 2D arrays internally but could be updated to use Trace objects.

## Branch

`tracespec`
