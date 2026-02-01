# Trace/Spectrum Refactoring Status

## Summary

Refactoring complete. `Trace` objects are now the standard interface throughout the pipeline.

## Completed

- `extract()` takes `list[Trace]`, returns `list[Spectrum]`
- `Trace.run()` and `Trace.load()` return `list[Trace]` directly
- Pipeline stores `list[Trace]` in `_data["trace"]`
- All step `run()` methods use `trace: list[Trace]` parameter
- Eliminated `.wavecal.npz` - data now in traces.fits + .linelist.npz
- Added `Trace.wlen(x)` method to evaluate wavelength polynomial
- Fixed `LaserFrequencyCombMaster.run()` bug
- All 549 unit tests pass

## Naming conventions

| Name | Type | Meaning |
|------|------|---------|
| `trace` | `list[Trace]` | Step run() parameter |
| `Trace.wave` | `np.ndarray` | Wavelength polynomial coefficients |
| `Trace.wlen(x)` | method | Evaluate wavelength at column positions |
| `wlen` | `np.ndarray` | Evaluated wavelength array |
| `wavecal_spec` | `np.ndarray` | Extracted calibration spectrum |

## Future work

Internal `traces_to_arrays()` calls remain in steps that use legacy algorithms:
- `estimate_background_scatter`
- `CurvatureModule`
- `combine_calibrate` plotting
- `rectify_image`

These could be updated to work with Trace objects directly.

## Branch

`tracespec`
