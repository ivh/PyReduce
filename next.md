# Remaining steps for preset_slitfunc feature

## Current state
- C code has `use_preset_slitfunc` flag to skip sL iteration
- `slitfunc_curved()` accepts `preset_slitfunc` parameter
- `extract()` threads `preset_slitfunc` through to per-order extraction
- `_adapt_slitfunc()` helper exists for resampling (untested)

## Blocking issue
Slitfunc size mismatch when `extraction_height` is fractional (e.g., 0.4):
- norm_flat computes yrange based on order spacing at extraction time
- ThAr extraction computes different yrange for same `extraction_height=0.4`
- Result: slitfunc has 1581 elements, extraction expects 201

## Fix options

### Option A: Use fixed integer extraction_height
- Set `extraction_height` to fixed pixel value (e.g., 10) in norm_flat settings
- Ensures consistent yrange across all extractions
- Simple, no code changes needed

### Option B: Store yrange in slitfunc_meta
- Save actual `yrange` used per order in norm_flat
- Pass yrange to ThAr extraction to force same geometry
- Requires changes to extract.py to accept explicit yrange

### Option C: Auto-adapt slitfunc in extract_spectrum
- Call `_adapt_slitfunc()` automatically when sizes don't match
- Resample slitfunc to target osample/yrange
- Already have the helper, just need to wire it up

## Recommended approach
Option A for now (simplest), then Option C for robustness.

## Next steps
1. Set `extraction_height: 10` (or similar fixed value) in MOSAIC settings
2. Re-run norm_flat to regenerate slitfuncs with consistent size
3. Test ThAr extraction with preset_slitfunc
4. (Later) Wire up `_adapt_slitfunc()` for automatic resampling
